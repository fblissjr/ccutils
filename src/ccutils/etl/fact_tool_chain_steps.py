"""Populate fact_tool_chain_steps from fact_tool_uses + fact_tool_results.

One row per tool_use, ordered within its "chain". A chain is the
contiguous block of tool_uses that share an assistant message_id; one
assistant turn = one chain. The chain_id is the message_id itself.

For each step we record step_position (1-indexed within chain),
prev_tool_key / next_tool_key (for "after Read, do I usually Edit?"
queries without window functions), is_error (from fact_tool_results),
and time_since_prev_seconds (always 0 in practice because all tools in
one assistant turn share a timestamp -- kept for forward compatibility
with future per-tool timestamping).

Run AFTER populate_fact_tool_uses + populate_fact_tool_results.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


_PAYLOAD_COLS = [
    "tool_use_id", "tool_key", "chain_id", "step_position",
    "prev_tool_key", "next_tool_key",
    "is_error", "time_since_prev_seconds", "timestamp",
]
_HASH_COLS = [
    "tool_use_id", "tool_key", "chain_id", "step_position",
    "prev_tool_key", "next_tool_key",
    "is_error", "time_since_prev_seconds",
]


def populate_fact_tool_chain_steps(conn, *, run: EtlRun) -> None:
    """Derive one fact_tool_chain_steps row per fact_tool_uses row."""
    conn.execute("DROP TABLE IF EXISTS _inbound_tool_chain_steps")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_tool_chain_steps AS
        WITH ordered AS (
            SELECT
                ftu.tool_use_id,
                ftu.session_id,
                ftu.tool_key,
                ftu.message_id AS chain_id,
                ftu.timestamp,
                ftu.invoke_sequence_num,
                ROW_NUMBER() OVER (
                    PARTITION BY ftu.session_id, ftu.message_id
                    ORDER BY ftu.invoke_sequence_num, ftu.timestamp
                ) AS step_position,
                LAG(ftu.tool_key) OVER (
                    PARTITION BY ftu.session_id, ftu.message_id
                    ORDER BY ftu.invoke_sequence_num, ftu.timestamp
                ) AS prev_tool_key,
                LEAD(ftu.tool_key) OVER (
                    PARTITION BY ftu.session_id, ftu.message_id
                    ORDER BY ftu.invoke_sequence_num, ftu.timestamp
                ) AS next_tool_key,
                LAG(ftu.timestamp) OVER (
                    PARTITION BY ftu.session_id, ftu.message_id
                    ORDER BY ftu.invoke_sequence_num, ftu.timestamp
                ) AS prev_timestamp
            FROM fact_tool_uses ftu
            WHERE ftu.is_deleted = FALSE
              AND ftu.session_id IN (
                  SELECT DISTINCT session_id FROM stg_log_entries
                  WHERE session_id IS NOT NULL
              )
        )
        SELECT
            md5(o.session_id || '|' || o.tool_use_id) AS chain_step_id,
            o.tool_use_id,
            o.session_id,
            o.tool_key,
            o.chain_id,
            o.step_position,
            o.prev_tool_key,
            o.next_tool_key,
            ftr.is_error,
            CASE WHEN o.prev_timestamp IS NOT NULL THEN
                EXTRACT(EPOCH FROM (o.timestamp - o.prev_timestamp))
            ELSE NULL END AS time_since_prev_seconds,
            o.timestamp
        FROM ordered o
        LEFT JOIN fact_tool_results ftr USING (tool_use_id)
        """
    )
    lineage_upsert(
        conn, run=run,
        table="fact_tool_chain_steps",
        inbound_table="_inbound_tool_chain_steps",
        natural_key="chain_step_id",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
    )
