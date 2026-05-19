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

Per-session scoping: every per-session source-fact read is bounded by
stg_log_entries.session_id (which only carries the session currently
being ETL'd). Without this guard, the populator would rescan the whole
warehouse on every per-session call.

Graceful absence: bool facets (F17/F18/F19) emit FALSE for sessions
without subagents / PR links / plan revisions, rather than going
missing. Same shape for the JSON facets when the source is empty
(F05/F06/F07/F09 emit an empty JSON array; F11 emits an empty JSON
object).

Run AFTER all source-fact populators; BEFORE populate_fact_session_summary
so the summary populator stays last as the aggregate roll-up.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


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

# extracted_at is lineage metadata; excluding it from hash_cols keeps the
# no-op re-run path clean (otherwise last_updated_at would churn on every
# call).
_HASH_COLS = [
    "facet_type_key",
    "prompt_version",
    "value_text",
    "value_json",
    "value_numeric",
    "value_bool",
]


def _facet_type_key_sql(facet_id: str) -> str:
    """Inline-md5 form of the Tier 1 facet_type_key. Mirrors the seed
    formula in schemas/star/schema.py: md5(facet_id || '|' || '') for
    facets whose prompt_version is NULL."""
    return f"md5('{facet_id}' || '|' || '')"


def _facet_row_key_sql(facet_id: str) -> str:
    """Inline-md5 form of the per-row natural key. Composite logical key
    (session_id, facet_id, prompt_version) collapsed to one column so
    lineage_upsert can WHERE on it."""
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
    """One row per session currently in staging, with its first_timestamp.
    Reused by every facet INSERT below."""
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


def _insert_text(conn, facet_id: str, value_expr: str) -> None:
    """value_expr is a SQL expression returning VARCHAR for a row in scope."""
    conn.execute(
        f"""
        INSERT INTO {_INBOUND}
            (facet_row_key, session_id, facet_type_key, prompt_version,
             value_text, timestamp)
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


def _insert_numeric(conn, facet_id: str, value_expr: str) -> None:
    conn.execute(
        f"""
        INSERT INTO {_INBOUND}
            (facet_row_key, session_id, facet_type_key, prompt_version,
             value_numeric, timestamp)
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


def _insert_bool(conn, facet_id: str, value_expr: str) -> None:
    conn.execute(
        f"""
        INSERT INTO {_INBOUND}
            (facet_row_key, session_id, facet_type_key, prompt_version,
             value_bool, timestamp)
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


def _insert_json(conn, facet_id: str, value_expr: str) -> None:
    conn.execute(
        f"""
        INSERT INTO {_INBOUND}
            (facet_row_key, session_id, facet_type_key, prompt_version,
             value_json, timestamp)
        SELECT
            {_facet_row_key_sql(facet_id)},
            scope.session_id,
            {_facet_type_key_sql(facet_id)},
            NULL,
            CAST({value_expr} AS JSON),
            scope.first_timestamp
        FROM {_SCOPE} scope
        """
    )


def populate_tier1_facets(conn, *, run: EtlRun) -> None:
    """Compute Tier 1 facets for every session currently in staging."""
    _create_scope(conn)
    _create_inbound(conn)

    # F01..F04 -- heuristic columns already on dim_session (populated by
    # populate_dim_session_heuristics earlier in the orchestrator).
    _insert_text(conn, "F01", "scope.intent")
    _insert_text(conn, "F02", "scope.complexity")
    _insert_text(conn, "F03", "scope.outcome")
    _insert_text(conn, "F04", "scope.domain")

    # F05 error_signature -- ordered JSON array of error_types per session.
    # Empty array when the session has no errors (graceful absence).
    _insert_json(
        conn, "F05",
        """COALESCE(
            (SELECT to_json(list(fe.error_type ORDER BY fe.timestamp))
             FROM fact_errors fe
             WHERE fe.is_deleted = FALSE
               AND fe.session_id = scope.session_id),
            to_json(CAST([] AS VARCHAR[]))
        )""",
    )

    # F06 tool_mix -- histogram of tool_name -> count, as a JSON object of
    # {tool, count} entries. dim_tool join gives tool_name (degenerate would
    # also work but the FK path is canonical).
    _insert_json(
        conn, "F06",
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

    # F07 tool_bigram_top3 -- top 3 (prev -> next) tool pairs by frequency,
    # as JSON array of strings like "Read->Edit". fact_tool_chain_steps
    # already carries next_tool_key.
    _insert_json(
        conn, "F07",
        """COALESCE(
            (SELECT to_json(list(pair ORDER BY ct DESC))
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

    # F08 loc_delta -- v0.15 doesn't surface line-add/line-remove counts as
    # typed columns (the data lives inside fact_tool_results.edit_structured_patch_json
    # as opaque JSON). For step 2 we emit a count-of-modifying-operations
    # proxy and leave the literal-LOC computation as a follow-up once the
    # structured_patch JSON is unpacked into a typed shape.
    _insert_numeric(
        conn, "F08",
        """COALESCE(
            (SELECT COUNT(*)::DOUBLE
             FROM fact_file_operations ffo
             WHERE ffo.is_deleted = FALSE
               AND ffo.session_id = scope.session_id
               AND ffo.operation_type IN ('write', 'edit')),
            0.0
        )""",
    )

    # F09 file_extensions_touched -- distinct extensions per session, as
    # JSON array.
    _insert_json(
        conn, "F09",
        """COALESCE(
            (SELECT to_json(list(DISTINCT df.file_extension ORDER BY df.file_extension))
             FROM fact_file_operations ffo
             JOIN dim_file df USING (file_key)
             WHERE ffo.is_deleted = FALSE
               AND ffo.session_id = scope.session_id
               AND df.file_extension IS NOT NULL),
            to_json(CAST([] AS VARCHAR[]))
        )""",
    )

    # F10 repo_slug -- dim_project.project_name (stable per repo).
    _insert_text(
        conn, "F10",
        """(SELECT dp.project_name FROM dim_project dp
            WHERE dp.project_key = scope.project_key)""",
    )

    # F11 model_mix -- tokens per model as JSON object of {model, tokens}.
    _insert_json(
        conn, "F11",
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

    # F12 duration_seconds -- last_timestamp - first_timestamp.
    _insert_numeric(
        conn, "F12",
        """COALESCE(
            EXTRACT(EPOCH FROM (scope.last_timestamp - scope.first_timestamp)),
            0.0
        )""",
    )

    # F13 agent_depth -- 0 for primary sessions, >0 for subagents.
    _insert_numeric(conn, "F13", "COALESCE(scope.depth_level, 0)::DOUBLE")

    # F14 human_message_count -- user messages, excluding meta entries.
    _insert_numeric(
        conn, "F14",
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

    # F15 tokens_in -- sum of input_tokens from fact_token_usage. Sourcing
    # from the per-API-response fact (not fact_session_summary) keeps Tier 1
    # independent of the summary populator order.
    _insert_numeric(
        conn, "F15",
        """COALESCE(
            (SELECT SUM(COALESCE(ftu.input_tokens, 0))::DOUBLE
             FROM fact_token_usage ftu
             WHERE ftu.is_deleted = FALSE
               AND ftu.session_id = scope.session_id),
            0.0
        )""",
    )

    # F16 local_hour -- hour of session start. Source timestamps are UTC;
    # ccutils does not record the system TZ at capture, so "local" here is
    # UTC-hour. True-local-hour would require a Claude Code change.
    _insert_numeric(
        conn, "F16",
        "EXTRACT(hour FROM scope.first_timestamp)::DOUBLE",
    )

    # F17/F18/F19 -- bool presence flags. EXISTS subqueries; graceful FALSE
    # for sessions without subagents / PR links / plan revisions.
    _insert_bool(
        conn, "F17",
        """EXISTS (
            SELECT 1 FROM fact_agent_delegations fad
            WHERE fad.is_deleted = FALSE
              AND fad.parent_session_key = md5(scope.session_id)
        )""",
    )
    _insert_bool(
        conn, "F18",
        """EXISTS (
            SELECT 1 FROM fact_pr_links fpl
            WHERE fpl.is_deleted = FALSE
              AND fpl.session_id = scope.session_id
        )""",
    )
    _insert_bool(
        conn, "F19",
        """EXISTS (
            SELECT 1 FROM fact_plan_revisions fpr
            WHERE fpr.is_deleted = FALSE
              AND fpr.session_id = scope.session_id
        )""",
    )

    # Delegate to lineage_upsert. extracted_at on inbound is set by
    # CREATE TEMP TABLE's default (current_timestamp) at table creation
    # time -- which is fine; the column is in payload_cols but excluded
    # from hash_cols so no-op re-runs don't churn last_updated_at.
    lineage_upsert(
        conn, run=run,
        table="fact_session_facets",
        inbound_table=_INBOUND,
        natural_key="facet_row_key",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
    )

    # Scope cleanup; inbound is dropped by lineage_upsert itself.
    conn.execute(f"DROP TABLE IF EXISTS {_SCOPE}")
