"""Populate fact_agent_delegations from fact_tool_uses + fact_tool_results.

One row per Task tool_use (parent-side agent spawn). Captures the task
input (description, prompt, subagent_type) from fact_tool_uses.input_json
and the agent rollup metrics (status, totalDurationMs, totalTokens,
totalToolUseCount, wasInterrupted) from fact_tool_results.agent_*
columns -- the v0.15 R1 structured toolUseResult capture.

agent_session_key / parent_session_key are NULL for now. Cross-session
subagent linkage (reading .meta.json sidecars to mark dim_session
.is_agent / .parent_session_key) is a separate Phase D follow-up; the
parent-side rollup metrics already tell most of the story.

Run AFTER populate_fact_tool_uses + populate_fact_tool_results.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


# Tools that spawn subagents. Claude Code 2.x emits both names depending
# on version / context -- both shapes match what we extract.
_AGENT_TOOL_NAMES = ("Task", "Agent")


_PAYLOAD_COLS = [
    "tool_use_id",
    "parent_session_key", "agent_session_key",
    "timestamp", "delegation_timestamp", "completion_timestamp",
    "seconds_to_completion",
    "task_description", "task_prompt", "subagent_type",
    "agent_status", "agent_total_duration_ms",
    "agent_total_tokens", "agent_total_tool_use_count",
    "agent_was_interrupted", "agent_output_text",
]
_HASH_COLS = [
    "tool_use_id", "delegation_timestamp", "completion_timestamp",
    "task_description", "task_prompt", "subagent_type",
    "agent_status", "agent_total_duration_ms",
    "agent_total_tokens", "agent_total_tool_use_count",
    "agent_was_interrupted", "agent_output_text",
]


def populate_fact_agent_delegations(conn, *, run: EtlRun) -> None:
    """Derive one fact_agent_delegations row per Task tool_use."""
    tool_list = ", ".join(f"'{t}'" for t in _AGENT_TOOL_NAMES)
    conn.execute("DROP TABLE IF EXISTS _inbound_agent_delegations")
    conn.execute(
        f"""
        CREATE TEMP TABLE _inbound_agent_delegations AS
        SELECT
            md5(ftu.session_id || '|' || ftu.tool_use_id) AS delegation_key,
            ftu.tool_use_id,
            ftu.session_id,
            -- The PARENT is the session that's doing the delegating;
            -- session_key on this fact already points there. We surface
            -- it under the parent_session_key column as well for query
            -- ergonomics on subagent rollup analysis.
            md5(ftu.session_id) AS parent_session_key,
            -- agent_session_key resolves via dim_session.agent_id (which
            -- gets set during subagent_dim_session enrichment).
            (SELECT ds.session_key FROM dim_session ds
             WHERE ds.is_agent = TRUE AND ds.agent_id = ftr.agent_id
             LIMIT 1) AS agent_session_key,
            ftu.timestamp,
            ftu.timestamp AS delegation_timestamp,
            ftr.timestamp AS completion_timestamp,
            CASE WHEN ftr.timestamp IS NOT NULL THEN
                EXTRACT(EPOCH FROM (ftr.timestamp - ftu.timestamp))
            ELSE NULL END AS seconds_to_completion,
            json_extract_string(ftu.input_json, '$.description')
                AS task_description,
            json_extract_string(ftu.input_json, '$.prompt')
                AS task_prompt,
            COALESCE(
                json_extract_string(ftu.input_json, '$.subagent_type'),
                ftr.agent_subagent_type
            ) AS subagent_type,
            ftr.agent_status,
            ftr.agent_total_duration_ms,
            ftr.agent_total_tokens,
            ftr.agent_total_tool_use_count,
            ftr.agent_was_interrupted,
            -- Tool result content can be a list of blocks (Agent typically
            -- emits one text block); fall back to result_content_text when
            -- the parser flattened it to a plain string.
            COALESCE(
                ftr.result_content_text,
                CAST(json_extract(ftr.result_payload_json, '$.content') AS VARCHAR)
            ) AS agent_output_text
        FROM fact_tool_uses ftu
        LEFT JOIN fact_tool_results ftr USING (tool_use_id)
        WHERE ftu.is_deleted = FALSE
          AND ftu.tool_name IN ({tool_list})
          AND ftu.session_id IN (
              SELECT DISTINCT session_id FROM stg_log_entries
              WHERE session_id IS NOT NULL
          )
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_agent_delegations",
        inbound_table="_inbound_agent_delegations",
        natural_key="delegation_key",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
        timestamp_col="delegation_timestamp",
    )
