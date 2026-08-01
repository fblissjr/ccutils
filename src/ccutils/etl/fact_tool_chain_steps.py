"""Populate fact_tool_chain_steps from fact_tool_uses + fact_messages.

One row per tool_use, ordered within its "chain". A chain is an *agentic
run*: the contiguous block of tool_uses following one human turn, up to
the next human turn. A human turn is a user message that is not a
tool-result carrier and not meta, so the run is bounded by real human
requests rather than by transcript packing.

The chain_id is md5(session_id || '|' || run_index), where run_index is
the count of human turns at or before the tool use (0 for tool uses that
precede the first human turn -- resumed and agent transcripts can open
mid-flight, and those keep an honest run_index of 0 rather than being
dropped).

Why not the assistant message: Claude Code writes ONE content block per
assistant entry -- parallel tool calls become separate entries sharing an
API message id but carrying distinct uuids. `fact_tool_uses.message_id`
is that per-entry uuid, so partitioning on it puts every tool use in its
own chain of length 1. That shipped: on a 2,250-session corpus, 71,175 of
71,216 chain steps were step_position 1 with prev/next_tool_key NULL,
which left semantic_tool_patterns with 5 rows and facet F07
(tool_bigram_top3) empty for 99.5% of sessions. The old fixtures hid it
by packing several tool_use blocks into one entry, a shape real
transcripts never produce.

step_position is 1-indexed within the run; prev_tool_key / next_tool_key
support "after Read, do I usually Edit?" without window functions; and
time_since_prev_seconds is now genuine elapsed time between consecutive
tools in a run (median ~4.6s on the real corpus) rather than the always-
NULL column the per-entry grain produced.

Run AFTER populate_fact_messages + populate_fact_tool_uses +
populate_fact_tool_results.
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
        WITH staged AS (
            SELECT DISTINCT session_id FROM stg_log_entries
            WHERE session_id IS NOT NULL
        ),
        -- One row per (session, entry). fact_messages is NOT unique on
        -- message_id alone -- resumed transcripts replay earlier entries,
        -- so the same uuid recurs under a different session_id, and a
        -- handful of (session_id, message_id) pairs repeat as well.
        -- Aggregating here keeps the lookup 1:1; joining raw fans out and
        -- would emit duplicate chain_step_id rows into lineage_upsert.
        msg_seq AS (
            SELECT session_id, message_id, MIN(sequence_num) AS sequence_num
            FROM fact_messages
            WHERE is_deleted = FALSE
              AND session_id IN (SELECT session_id FROM staged)
            GROUP BY session_id, message_id
        ),
        -- A human turn: a user message carrying real input, not a
        -- tool_result envelope and not a meta entry.
        human_turns AS (
            SELECT DISTINCT session_id, sequence_num
            FROM fact_messages
            WHERE is_deleted = FALSE
              AND message_type = 'user'
              AND has_tool_result = FALSE
              AND is_meta = FALSE
              AND session_id IN (SELECT session_id FROM staged)
        ),
        tool_events AS (
            SELECT
                ftu.tool_use_id,
                ftu.session_id,
                ftu.tool_key,
                ftu.timestamp,
                ms.sequence_num,
                ftu.invoke_sequence_num
            FROM fact_tool_uses ftu
            JOIN msg_seq ms
              ON ms.session_id = ftu.session_id
             AND ms.message_id = ftu.message_id
            WHERE ftu.is_deleted = FALSE
              AND ftu.session_id IN (SELECT session_id FROM staged)
        ),
        -- Interleave human turns with tool uses on the session's entry
        -- order, then a running count of human turns labels each run.
        events AS (
            SELECT
                session_id, sequence_num,
                CAST(NULL AS TIMESTAMP) AS timestamp,
                1 AS is_human,
                CAST(NULL AS VARCHAR) AS tool_use_id,
                CAST(NULL AS VARCHAR) AS tool_key,
                0 AS invoke_sequence_num
            FROM human_turns
            UNION ALL
            SELECT
                session_id, sequence_num, timestamp,
                0 AS is_human,
                tool_use_id, tool_key, invoke_sequence_num
            FROM tool_events
        ),
        runs AS (
            SELECT
                *,
                SUM(is_human) OVER (
                    PARTITION BY session_id
                    -- is_human DESC so a human turn sharing a sequence_num
                    -- with a tool use opens the run containing it.
                    ORDER BY sequence_num, is_human DESC, invoke_sequence_num
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS run_index
            FROM events
        ),
        -- Collapse tool results to one row per (session, tool_use) before
        -- joining: the raw table carries a few duplicate pairs, and an
        -- unscoped USING (tool_use_id) join also ignored is_deleted. Either
        -- one fans out into duplicate chain_step_id rows. MAX() over the
        -- tri-state BOOLEAN keeps the right semantics -- TRUE if any result
        -- errored, FALSE if all succeeded, NULL only when all are unknown.
        results AS (
            SELECT session_id, tool_use_id, MAX(is_error) AS is_error
            FROM fact_tool_results
            WHERE is_deleted = FALSE
              AND session_id IN (SELECT session_id FROM staged)
            GROUP BY session_id, tool_use_id
        ),
        ordered AS (
            SELECT
                tool_use_id,
                session_id,
                tool_key,
                timestamp,
                md5(session_id || '|' || CAST(run_index AS VARCHAR)) AS chain_id,
                ROW_NUMBER() OVER w AS step_position,
                LAG(tool_key) OVER w AS prev_tool_key,
                LEAD(tool_key) OVER w AS next_tool_key,
                LAG(timestamp) OVER w AS prev_timestamp
            FROM runs
            WHERE tool_use_id IS NOT NULL
            WINDOW w AS (
                PARTITION BY session_id, run_index
                ORDER BY sequence_num, invoke_sequence_num
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
            r.is_error,
            CASE WHEN o.prev_timestamp IS NOT NULL THEN
                EXTRACT(EPOCH FROM (o.timestamp - o.prev_timestamp))
            ELSE NULL END AS time_since_prev_seconds,
            o.timestamp
        FROM ordered o
        LEFT JOIN results r
          ON r.session_id = o.session_id
         AND r.tool_use_id = o.tool_use_id
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
