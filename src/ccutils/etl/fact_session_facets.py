"""Tier 1 facet populator: 19 facets per session, computed from v0.15 facts.

Step 2 of the facet & cluster pipeline (docs/FACET_CLUSTER_PIPELINE.md).
One row per (session, facet) into fact_session_facets via lineage_upsert.

Tier 1 facets are SQL-only and deterministic. No inference, no external
dependencies, no LLM. The catalog lives in
docs/FACET_CLUSTER_PIPELINE.md §3 (Tier 1, F01..F19); the registry seeds
in schemas/star/schema.py mirror it.

Source facts (all v0.15; all populated before this populator by the
orchestrator):
    dim_session (heuristic columns + first/last_timestamp + depth_level)
    dim_project (repo_slug)
    dim_model
    dim_file
    fact_messages
    fact_tool_uses, fact_tool_chain_steps
    fact_errors
    fact_file_operations
    fact_token_usage
    fact_agent_delegations, fact_pr_links, fact_plan_revisions

Ordering: F01..F04 read `intent`/`complexity`/`outcome`/`domain` from
`dim_session`, which `populate_dim_session_heuristics` writes earlier in
the orchestrator. Calling this populator standalone (without the
heuristic pass first) emits NULL for those four facets.

Per-session scoping: every per-session source-fact read is bounded by
stg_log_entries.session_id (which only carries the session currently
being ETL'd). Without this guard, the populator would rescan the whole
warehouse on every per-session call.

Graceful absence: bool facets (F17/F18/F19) emit FALSE for sessions
without subagents / PR links / plan revisions, rather than going
missing. JSON facets emit an empty array when source is empty
(F05/F06/F07/F09/F11).

Run AFTER all source-fact populators; BEFORE populate_fact_session_summary
so the summary populator stays last as the aggregate roll-up.
"""

from __future__ import annotations

from ccutils.etl.facets.catalog import facet_tier_scope_sql
from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert
from ccutils.etl.utils import fetch_scalar


_INBOUND = "_inbound_tier1_facets"
_SCOPE = "_t1_scope"

# CLAUDE.md rule: payload_cols must NOT include session_key (lineage_upsert
# derives + injects it via extra_keys).
_PAYLOAD_COLS = [
    "facet_type_key",
    "prompt_version",
    "value_text",
    "value_json",
    "value_numeric",
    "value_bool",
    "extracted_at",
]

# `extracted_at` is lineage metadata; excluding it keeps no-op re-runs from
# churning last_updated_at. Derive rather than re-list so the exclusion rule
# stays in one place.
_HASH_COLS = [c for c in _PAYLOAD_COLS if c != "extracted_at"]


def _facet_type_key_sql(facet_id: str) -> str:
    """Mirrors the Tier 1 seed formula in schemas/star/schema.py:
    md5(facet_id || '|' || '') when prompt_version is NULL. Tier 2's
    populator will compute the same shape with a real prompt_version."""
    return f"md5('{facet_id}' || '|' || '')"


def _facet_row_key_sql(facet_id: str) -> str:
    """Composite logical key (session_id, facet_id, prompt_version)
    collapsed to a single column so lineage_upsert can WHERE on it. Same
    pattern as fact_errors.error_id."""
    return f"md5(scope.session_id || '|' || '{facet_id}' || '|' || '')"


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
            timestamp TIMESTAMP,
            extracted_at TIMESTAMP DEFAULT current_timestamp
        )
        """
    )


def _create_scope(conn) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {_SCOPE}")
    conn.execute(
        f"""
        CREATE TEMP TABLE {_SCOPE} AS
        SELECT ds.session_id, ds.first_timestamp, ds.last_timestamp,
               ds.depth_level, ds.intent, ds.complexity, ds.outcome,
               ds.domain, ds.project_key
        FROM dim_session ds
        WHERE ds.session_id IN (
            SELECT DISTINCT session_id FROM stg_log_entries
            WHERE session_id IS NOT NULL
        )
        """
    )


def _insert_facet(conn, facet_id: str, value_col: str, value_expr: str) -> None:
    """Emit one row per scope session into the inbound table. `value_col` is
    one of value_text / value_json / value_numeric / value_bool; `value_expr`
    is the SQL expression that produces it (may use `scope.*` to access
    per-session attrs). JSON facets pass an expression that already evaluates
    to JSON — no implicit CAST."""
    conn.execute(
        f"""
        INSERT INTO {_INBOUND}
            (facet_row_key, session_id, facet_type_key, prompt_version,
             {value_col}, timestamp)
        SELECT
            {_facet_row_key_sql(facet_id)},
            scope.session_id,
            {_facet_type_key_sql(facet_id)},
            NULL,
            {value_expr},
            scope.first_timestamp
        FROM {_SCOPE} scope
        """
    )


def populate_tier1_facets(conn, *, run: EtlRun) -> None:
    """Compute Tier 1 facets for every session currently in staging."""
    _create_scope(conn)

    # Empty staging => empty scope => no work. Skip the 19 INSERTs +
    # lineage_upsert ALTER/UPDATE sequence entirely.
    if fetch_scalar(conn, f"SELECT COUNT(*) FROM {_SCOPE}") == 0:
        conn.execute(f"DROP TABLE IF EXISTS {_SCOPE}")
        return

    _create_inbound(conn)

    # F01..F04 -- heuristic columns already on dim_session.
    _insert_facet(conn, "F01", "value_text", "scope.intent")
    _insert_facet(conn, "F02", "value_text", "scope.complexity")
    _insert_facet(conn, "F03", "value_text", "scope.outcome")
    _insert_facet(conn, "F04", "value_text", "scope.domain")

    # F05 -- ordered list of error_types; empty array when no errors.
    _insert_facet(
        conn, "F05", "value_json",
        """COALESCE(
            (SELECT to_json(list(fe.error_type ORDER BY fe.timestamp))
             FROM fact_errors fe
             WHERE fe.is_deleted = FALSE
               AND fe.session_id = scope.session_id),
            to_json(CAST([] AS VARCHAR[]))
        )""",
    )

    # F06 -- tool_name -> count histogram as JSON array of {tool, count}.
    _insert_facet(
        conn, "F06", "value_json",
        """COALESCE(
            (SELECT to_json(list({'tool': tool_name, 'count': ct}))
             FROM (
                SELECT dt.tool_name, COUNT(*) AS ct
                FROM fact_tool_uses ftu
                JOIN dim_tool dt USING (tool_key)
                WHERE ftu.is_deleted = FALSE
                  AND ftu.session_id = scope.session_id
                GROUP BY dt.tool_name
             )),
            to_json(CAST([] AS VARCHAR[]))
        )""",
    )

    # F07 -- top 3 (prev -> next) tool pairs. Inner LIMIT 3 + ORDER BY does
    # the selection; outer list() preserves arrival order, no re-sort needed.
    _insert_facet(
        conn, "F07", "value_json",
        """COALESCE(
            (SELECT to_json(list(pair))
             FROM (
                SELECT
                    dt_curr.tool_name || '->' || dt_next.tool_name AS pair,
                    COUNT(*) AS ct
                FROM fact_tool_chain_steps fcs
                JOIN dim_tool dt_curr ON fcs.tool_key = dt_curr.tool_key
                JOIN dim_tool dt_next ON fcs.next_tool_key = dt_next.tool_key
                WHERE fcs.is_deleted = FALSE
                  AND fcs.session_id = scope.session_id
                  AND fcs.next_tool_key IS NOT NULL
                GROUP BY dt_curr.tool_name, dt_next.tool_name
                ORDER BY ct DESC
                LIMIT 3
             )),
            to_json(CAST([] AS VARCHAR[]))
        )""",
    )

    # F08 emits a proxy (count of modifying ops). The real LOC delta lives
    # inside fact_tool_results.edit_structured_patch_json and needs unpacking
    # into typed columns -- tracked as a follow-up (see dim_facet_type.notes).
    _insert_facet(
        conn, "F08", "value_numeric",
        """COALESCE(
            (SELECT COUNT(*)::DOUBLE
             FROM fact_file_operations ffo
             WHERE ffo.is_deleted = FALSE
               AND ffo.session_id = scope.session_id
               AND ffo.operation_type IN ('write', 'edit')),
            0.0
        )""",
    )

    _insert_facet(
        conn, "F09", "value_json",
        """COALESCE(
            (SELECT to_json(list(DISTINCT df.file_extension
                                 ORDER BY df.file_extension))
             FROM fact_file_operations ffo
             JOIN dim_file df USING (file_key)
             WHERE ffo.is_deleted = FALSE
               AND ffo.session_id = scope.session_id
               AND df.file_extension IS NOT NULL),
            to_json(CAST([] AS VARCHAR[]))
        )""",
    )

    _insert_facet(
        conn, "F10", "value_text",
        """(SELECT dp.project_name FROM dim_project dp
            WHERE dp.project_key = scope.project_key)""",
    )

    _insert_facet(
        conn, "F11", "value_json",
        """COALESCE(
            (SELECT to_json(list({'model': model_name, 'tokens': total_tokens}))
             FROM (
                SELECT dm.model_name,
                       SUM(COALESCE(ftu.input_tokens, 0)
                          + COALESCE(ftu.output_tokens, 0)) AS total_tokens
                FROM fact_token_usage ftu
                JOIN dim_model dm USING (model_key)
                WHERE ftu.is_deleted = FALSE
                  AND ftu.session_id = scope.session_id
                GROUP BY dm.model_name
             )),
            to_json(CAST([] AS VARCHAR[]))
        )""",
    )

    _insert_facet(
        conn, "F12", "value_numeric",
        """COALESCE(
            EXTRACT(EPOCH FROM (scope.last_timestamp - scope.first_timestamp)),
            0.0
        )""",
    )

    _insert_facet(conn, "F13", "value_numeric",
                  "COALESCE(scope.depth_level, 0)::DOUBLE")

    _insert_facet(
        conn, "F14", "value_numeric",
        """COALESCE(
            (SELECT COUNT(*)::DOUBLE
             FROM fact_messages fm
             WHERE fm.is_deleted = FALSE
               AND fm.message_type = 'user'
               AND COALESCE(fm.is_meta, FALSE) = FALSE
               AND fm.session_id = scope.session_id),
            0.0
        )""",
    )

    # F15 sources from fact_token_usage rather than fact_session_summary so
    # Tier 1 stays independent of summary populator order.
    _insert_facet(
        conn, "F15", "value_numeric",
        """COALESCE(
            (SELECT SUM(COALESCE(ftu.input_tokens, 0))::DOUBLE
             FROM fact_token_usage ftu
             WHERE ftu.is_deleted = FALSE
               AND ftu.session_id = scope.session_id),
            0.0
        )""",
    )

    _insert_facet(conn, "F16", "value_numeric",
                  "EXTRACT(hour FROM scope.first_timestamp)::DOUBLE")

    # F17 -- fact_agent_delegations.parent_session_key is the md5 of the
    # parent session_id (no plain parent_session_id column carried), so hash
    # inline. Asymmetric with F18/F19 which compare session_id directly.
    _insert_facet(
        conn, "F17", "value_bool",
        """EXISTS (
            SELECT 1 FROM fact_agent_delegations fad
            WHERE fad.is_deleted = FALSE
              AND fad.parent_session_key = md5(scope.session_id)
        )""",
    )
    _insert_facet(
        conn, "F18", "value_bool",
        """EXISTS (
            SELECT 1 FROM fact_pr_links fpl
            WHERE fpl.is_deleted = FALSE
              AND fpl.session_id = scope.session_id
        )""",
    )
    _insert_facet(
        conn, "F19", "value_bool",
        """EXISTS (
            SELECT 1 FROM fact_plan_revisions fpr
            WHERE fpr.is_deleted = FALSE
              AND fpr.session_id = scope.session_id
        )""",
    )

    # F30/F31 -- the behavioral pair. F15 (tokens_in) shipped without an
    # output counterpart, and thinking depth was reachable only through
    # fact_session_summary, which Tier 1 must not depend on (it populates
    # last). Both are emitted as raw counts, deliberately unnormalized and
    # unbucketed: any archetype thresholds belong in the analysis layer,
    # derived from the corpus distribution, not frozen into ETL.
    _insert_facet(
        conn, "F30", "value_numeric",
        """COALESCE(
            (SELECT SUM(COALESCE(ftu.output_tokens, 0))::DOUBLE
             FROM fact_token_usage ftu
             WHERE ftu.is_deleted = FALSE
               AND ftu.session_id = scope.session_id),
            0.0
        )""",
    )
    _insert_facet(
        conn, "F31", "value_numeric",
        """COALESCE(
            (SELECT COUNT(*)::DOUBLE
             FROM fact_messages fm
             WHERE fm.is_deleted = FALSE
               AND fm.has_thinking = TRUE
               AND fm.session_id = scope.session_id),
            0.0
        )""",
    )

    # Free scope before lineage_upsert mutates the inbound. If
    # lineage_upsert raises, we don't want the scope to leak across the
    # connection lifetime.
    conn.execute(f"DROP TABLE IF EXISTS {_SCOPE}")

    # Scope the soft-delete to Tier 1 facet_type_keys only. fact_session_facets
    # is written by both Tier 1 and Tier 2 populators; without scoping, this
    # populator would soft-delete every Tier 2 row in the session (and Tier 2
    # would soft-delete every Tier 1 row on its own pass).
    lineage_upsert(
        conn, run=run,
        table="fact_session_facets",
        inbound_table=_INBOUND,
        natural_key="facet_row_key",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
        soft_delete_scope_sql=facet_tier_scope_sql(1),
    )
