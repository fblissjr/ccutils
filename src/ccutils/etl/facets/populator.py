"""Tier 2 facet populator: LLM-extracted facets via injected extractor.

Step 4 of the facet & cluster pipeline. Builds a SessionInputs from the
v0.15 facts already populated for the current staging session, calls
the injected `FacetExtractor`, and writes one
`fact_session_facets` row per FacetOutput via `lineage_upsert`.

Contract (matches `internal/plans/facet_extractor_protocol.md` §2):
  - Each session in `stg_log_entries` becomes one SessionInputs:
      session_id        -> stg_log_entries.session_id
      first_user_message -> first non-meta user entry's content text
      last_assistant_message -> final assistant entry's content text
      tool_mix_summary   -> top-5 tools by count, "Bash×5, Read×3, ..."
      model_used         -> most-frequent dim_model.model_name
      duration_seconds   -> dim_session.last - first
  - The extractor decides truncation (the FacetExtractor protocol owns
    the input contract; the populator passes raw text and lets the
    extractor's _truncate_prefix / _truncate_suffix do the cut).
  - One row per (session, FacetSpec) lands in fact_session_facets.
    is_fallback + extraction_metadata_json flow from FacetOutput.

Failure isolation: a single session's extraction failure (an exception
the FacetExtractor raises rather than absorbs) is logged but does not
abort the populator -- other sessions in staging still get processed.
"""

from __future__ import annotations

import logging

from ccutils.etl.facets.catalog import FACET_SPECS, facet_tier_scope_sql
from ccutils.etl.facets.extractor import (
    OUTPUT_TYPE_TO_COL,
    FacetExtractor,
    FacetOutput,
    OutputType,
    SessionInputs,
)
from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert
from ccutils.etl.utils import extract_text_from_content_json


_log = logging.getLogger(__name__)


_INBOUND = "_inbound_tier2_facets"

# Same payload-col rules as Tier 1 (see fact_session_facets.py). Adds
# is_fallback + extraction_metadata_json which Tier 1 leaves at their
# DDL defaults.
_PAYLOAD_COLS = [
    "facet_type_key",
    "prompt_version",
    "value_text",
    "value_json",
    "value_numeric",
    "value_bool",
    "is_fallback",
    "extraction_metadata_json",
    "extracted_at",
]

# extracted_at is lineage metadata; excluding it keeps no-op re-runs
# from churning last_updated_at. extraction_metadata_json varies per
# call (timestamps, latency) so it's also excluded -- otherwise every
# re-run would flip the hash even when the model's actual value is
# unchanged.
_HASH_COLS = [
    c for c in _PAYLOAD_COLS
    if c not in ("extracted_at", "extraction_metadata_json")
]


def populate_tier2_facets(
    conn,
    *,
    run: EtlRun,
    extractor: FacetExtractor,
    include_thinking: bool = True,
) -> None:
    """Extract Tier 2 facets for every session currently in staging."""
    session_rows = _build_session_inputs(conn, include_thinking=include_thinking)
    if not session_rows:
        return

    _create_inbound(conn)

    # Catch every exception below: a single session's extractor failure
    # must not abort the whole batch. Logged + skipped so other sessions
    # in staging still process. BLE001 is the deliberate trade-off.
    inbound_rows: list[tuple] = []
    for inputs, first_timestamp in session_rows:
        try:
            outputs = extractor.extract(inputs, FACET_SPECS)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "Tier 2 extraction failed for session %s: %s",
                inputs.session_id, exc,
            )
            continue

        for spec in FACET_SPECS:
            output = outputs.get(spec.facet_id)
            if output is None:
                _log.warning(
                    "Extractor returned no output for %s on session %s",
                    spec.facet_id, inputs.session_id,
                )
                continue
            inbound_rows.append(
                _build_inbound_row(
                    inputs.session_id, first_timestamp,
                    spec.facet_id, spec.prompt_version,
                    spec.output_type, output,
                )
            )

    if inbound_rows:
        _bulk_insert_inbound(conn, inbound_rows)

    inbound_count = conn.execute(
        f"SELECT COUNT(*) FROM {_INBOUND}"
    ).fetchone()[0]
    if inbound_count == 0:
        # No rows accumulated (every session failed). Skip the
        # lineage_upsert sequence entirely -- it would no-op anyway,
        # but the cleanup is honest.
        conn.execute(f"DROP TABLE IF EXISTS {_INBOUND}")
        return

    lineage_upsert(
        conn, run=run,
        table="fact_session_facets",
        inbound_table=_INBOUND,
        natural_key="facet_row_key",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
        soft_delete_scope_sql=facet_tier_scope_sql(2),
    )


# ---------------------------------------------------------------------------
# SessionInputs construction
# ---------------------------------------------------------------------------


def _build_session_inputs(
    conn, *, include_thinking: bool = True,
) -> list[tuple[SessionInputs, str | None]]:
    """One (SessionInputs, first_timestamp) tuple per session in staging.

    Returns first_timestamp alongside so the inbound table can carry it
    into lineage_upsert's date_key / time_key derivation.
    """
    rows = conn.execute(
        """
        WITH scope AS (
            SELECT DISTINCT session_id FROM stg_log_entries
            WHERE session_id IS NOT NULL
        ),
        first_user AS (
            SELECT sle.session_id,
                   CAST(json_extract(sle.message_json, '$.content') AS VARCHAR)
                       AS content_json
            FROM stg_log_entries sle
            JOIN scope ss USING (session_id)
            WHERE sle.type = 'user'
              AND COALESCE(sle.is_meta, FALSE) = FALSE
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY sle.session_id
                ORDER BY sle.timestamp, sle.sequence_num
            ) = 1
        ),
        last_assistant AS (
            SELECT sle.session_id,
                   CAST(json_extract(sle.message_json, '$.content') AS VARCHAR)
                       AS content_json
            FROM stg_log_entries sle
            JOIN scope ss USING (session_id)
            WHERE sle.type = 'assistant'
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY sle.session_id
                ORDER BY sle.timestamp DESC, sle.sequence_num DESC
            ) = 1
        ),
        tool_counts AS (
            SELECT ftu.session_id, dt.tool_name, COUNT(*) AS ct
            FROM fact_tool_uses ftu
            JOIN dim_tool dt USING (tool_key)
            WHERE ftu.is_deleted = FALSE
              AND ftu.session_id IN (SELECT session_id FROM scope)
            GROUP BY ftu.session_id, dt.tool_name
        ),
        top5_tools AS (
            SELECT session_id, tool_name, ct
            FROM tool_counts
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY session_id ORDER BY ct DESC
            ) <= 5
        ),
        tool_mix AS (
            SELECT session_id,
                   string_agg(tool_name || '×' || CAST(ct AS VARCHAR),
                              ', ' ORDER BY ct DESC) AS tool_mix_summary
            FROM top5_tools
            GROUP BY session_id
        ),
        model_counts AS (
            -- Use a different alias from tool_counts above (`ftu`) so
            -- a future JOIN between them doesn't quietly collide.
            SELECT ftok.session_id, dm.model_name, COUNT(*) AS ct
            FROM fact_token_usage ftok
            JOIN dim_model dm USING (model_key)
            WHERE ftok.is_deleted = FALSE
              AND ftok.session_id IN (SELECT session_id FROM scope)
            GROUP BY ftok.session_id, dm.model_name
        ),
        model_used AS (
            SELECT session_id, model_name
            FROM model_counts
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY session_id ORDER BY ct DESC
            ) = 1
        )
        SELECT
            ss.session_id,
            fu.content_json AS first_user_content_json,
            la.content_json AS last_assistant_content_json,
            COALESCE(tm.tool_mix_summary, '') AS tool_mix_summary,
            mu.model_name AS model_used,
            CAST(EXTRACT(EPOCH FROM (ds.last_timestamp - ds.first_timestamp))
                 AS INTEGER) AS duration_seconds,
            ds.first_timestamp
        FROM scope ss
        LEFT JOIN first_user fu USING (session_id)
        LEFT JOIN last_assistant la USING (session_id)
        LEFT JOIN tool_mix tm USING (session_id)
        LEFT JOIN model_used mu USING (session_id)
        LEFT JOIN dim_session ds USING (session_id)
        """
    ).fetchall()

    out: list[tuple[SessionInputs, str | None]] = []
    for (
        session_id,
        first_user_json,
        last_assistant_json,
        tool_mix_summary,
        model_used,
        duration_seconds,
        first_timestamp,
    ) in rows:
        inputs = SessionInputs(
            session_id=session_id,
            first_user_message=extract_text_from_content_json(
                first_user_json, include_thinking=include_thinking,
            ),
            last_assistant_message=extract_text_from_content_json(
                last_assistant_json, include_thinking=include_thinking,
            ),
            tool_mix_summary=tool_mix_summary or "",
            model_used=model_used,
            duration_seconds=duration_seconds,
        )
        out.append((inputs, first_timestamp))
    return out


# ---------------------------------------------------------------------------
# Inbound table
# ---------------------------------------------------------------------------


def _create_inbound(conn) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {_INBOUND}")
    conn.execute(
        f"""
        CREATE TEMP TABLE {_INBOUND} (
            facet_row_key VARCHAR,
            session_id VARCHAR,
            facet_type_key VARCHAR,
            prompt_version VARCHAR,
            value_text VARCHAR,
            value_json JSON,
            value_numeric DOUBLE,
            value_bool BOOLEAN,
            is_fallback BOOLEAN,
            extraction_metadata_json JSON,
            timestamp TIMESTAMP,
            extracted_at TIMESTAMP DEFAULT current_timestamp
        )
        """
    )


def _route_value(
    output: FacetOutput, output_type: OutputType,
) -> tuple[str | None, str | None, float | None, bool | None]:
    """Choose which of (value_text, value_json, value_numeric, value_bool)
    carries the extracted value. The EAV shape on fact_session_facets is
    intentional: each row stores its value in the typed column that
    matches its dim_facet_type.output_type, so downstream queries can
    `SELECT value_numeric WHERE output_type='int'` without string-casting.

    The OUTPUT_TYPE_TO_COL map names the column; this routes the actual
    value with any necessary coercion.
    """
    if output.value is None:
        return None, None, None, None

    target = OUTPUT_TYPE_TO_COL[output_type]
    if target == "value_text":
        return output.value, None, None, None
    if target == "value_json":
        # The extractor stringifies JSON-output values via json.dumps;
        # pass through unchanged.
        return None, output.value, None, None
    if target == "value_numeric":
        try:
            return None, None, float(output.value), None
        except (TypeError, ValueError):
            # Couldn't cast to numeric -- fall back to text rather than
            # drop the value entirely.
            return output.value, None, None, None
    if target == "value_bool":
        return None, None, None, output.value.lower() in ("true", "1", "yes")
    return output.value, None, None, None  # safety net for unknown types


def _build_inbound_row(
    session_id: str,
    first_timestamp,
    facet_id: str,
    prompt_version: str,
    output_type: OutputType,
    output: FacetOutput,
) -> tuple:
    """Materialize the positional parameter tuple for one inbound row.
    Decoupled from the INSERT so the dispatch is testable on its own and
    the bulk-INSERT path doesn't need a Python loop of executes."""
    value_text, value_json, value_numeric, value_bool = _route_value(
        output, output_type,
    )
    return (
        session_id, facet_id, prompt_version,   # facet_row_key parts
        session_id,                              # session_id col
        facet_id, prompt_version,                # facet_type_key parts
        prompt_version,                          # prompt_version col
        value_text, value_json, value_numeric, value_bool,
        output.is_fallback, output.metadata_json,
        first_timestamp,
    )


def _bulk_insert_inbound(conn, rows: list[tuple]) -> None:
    """One DuckDB call covers every accumulated inbound row. Avoids one
    round-trip per (session × facet)."""
    conn.executemany(
        f"""
        INSERT INTO {_INBOUND}
            (facet_row_key, session_id, facet_type_key, prompt_version,
             value_text, value_json, value_numeric, value_bool,
             is_fallback, extraction_metadata_json, timestamp)
        VALUES (
            md5(? || '|' || ? || '|' || ?),
            ?,
            md5(? || '|' || ?),
            ?,
            ?, ?, ?, ?,
            ?, CAST(? AS JSON), ?
        )
        """,
        rows,
    )
