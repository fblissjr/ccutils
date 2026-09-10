"""Populate fact_tool_calls from etl.log_entries: one row per tool_use_id.

The use (the assistant's tool_use block), its result (the tool_result block
plus the entry-level `toolUseResult` payload, typed per tool), and its
position in the agentic run (chain_id, step_position, prev/next tool) are
one row, because they are one grain. They used to be three tables keyed by
the same id -- fact_tool_uses, fact_tool_results, fact_tool_chain_steps --
plus fact_errors, a filter over the second; on the corpus every use had
exactly one result and one chain step, and every consumer joined them.

A result whose tool_use_id has no use in the same session is dropped: the
use is the row, the result decorates it. A use with no result (the session
ended mid-flight) keeps NULL result columns.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert
from ccutils.etl.utils import project_key_sql


# Columns copied into fact_tool_calls by the lineage upsert. EXCLUDES the
# natural key (tool_use_id), session_id, and the derived keys (session_key,
# date_key, time_key) which the helper handles.
_RESULT_TYPED_COLS = [
    "is_error", "result_content_text", "result_payload_json",
    "bash_stdout_bytes", "derived_exit_code", "derived_failure_kind", "sandbox_disabled",
    "edit_user_modified", "edit_replace_all", "edit_structured_patch_json",
    "read_num_lines", "read_total_lines", "read_file_path",
    "write_type",
    "glob_num_files", "glob_truncated",
    "grep_mode", "grep_num_files",
    "webfetch_http_code", "webfetch_bytes",
    "agent_status", "agent_total_duration_ms", "agent_total_tokens",
    "agent_total_tool_use_count", "agent_subagent_type",
    "agent_id", "agent_resolved_model", "agent_is_async",
]
_CHAIN_COLS = [
    "chain_id", "step_position", "prev_tool_key", "next_tool_key",
    "time_since_prev_seconds",
]
_PAYLOAD_COLS = [
    "entry_id", "message_id", "project_key", "tool_key",
    "tool_name", "invoke_sequence_num", "input_json", "input_summary", "timestamp",
    "result_entry_id", "result_message_id", "result_timestamp",
    *_RESULT_TYPED_COLS, *_CHAIN_COLS,
]
_HASH_COLS = [
    "tool_name", "invoke_sequence_num", "input_json", "input_summary", "timestamp",
    "result_timestamp", *_RESULT_TYPED_COLS, *_CHAIN_COLS,
]


# fact_tool_uses: explode every assistant message's content[*] for tool_use blocks.
_PROJECT_USES_SQL = """
WITH assistant_entries AS (
    SELECT
        sle.entry_id,
        sle.uuid AS message_id,
        sle.session_id,
        sle.source_path,
        TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
        sle.message_json
    FROM etl.log_entries sle
    WHERE sle.type = 'assistant'
      AND json_type(sle.message_json, '$.content') = 'ARRAY'
),
exploded AS (
    SELECT
        ae.*,
        b.block,
        b.block_idx
    FROM assistant_entries ae,
    LATERAL (
        SELECT unnest(json_extract(ae.message_json, '$.content')::JSON[]) AS block,
               generate_subscripts(
                   json_extract(ae.message_json, '$.content')::JSON[], 1
               ) AS block_idx
    ) b
    WHERE json_extract_string(b.block, '$.type') = 'tool_use'
)
SELECT
    -- entry_id of the source entry is used as the partition key for upsert;
    -- but multiple tool_uses can share an entry, so the natural key for
    -- this fact is tool_use_id. We still carry entry_id for lineage.
    entry_id,
    message_id,
    session_id,
    source_path,
    timestamp,
    json_extract_string(block, '$.id') AS tool_use_id,
    json_extract_string(block, '$.name') AS tool_name,
    block_idx AS invoke_sequence_num,
    CAST(json_extract(block, '$.input') AS VARCHAR) AS input_json
FROM exploded

-- One row per tool_use_id, which is the grain this fact DECLARES via
-- natural_key="tool_use_id". A real Claude Code session records a single
-- tool_use_id under two distinct entry uuids (29 keys on a 2,344-session
-- corpus), so unnesting blocks across entries would otherwise emit two rows
-- for one key and break the uniqueness every consumer joins on.
--
-- Collapsing is safe HERE and only here, because the justification is
-- specific to this data: the duplicate records are identical in every
-- column this projection emits -- `tool_name`, `input_json` and
-- `invoke_sequence_num` all match on 7 of 7 measured duplicate keys. Only
-- `entry_id` differs, which is by definition the source entry reference.
-- Earliest entry wins; entry_id breaks ties so a rebuild from identical
-- input is reproducible.
--
-- This used to be a generic collapse inside lineage_upsert, which applied
-- one fact's judgment to all 13 and resolved it silently. That helper now
-- ASSERTS this grain instead -- if this QUALIFY is removed, the ETL fails
-- loudly rather than silently double-counting.
-- NULL tool_use_id is left alone: SQL puts every NULL in ONE partition, so
-- collapsing here would silently drop all but one. lineage_upsert excludes
-- NULL from its uniqueness check for the same reason. (The column is NOT NULL
-- at the DDL level, so such a row fails loudly on insert -- which is the
-- correct outcome and must not be masked into a silent drop.)
QUALIFY tool_use_id IS NULL OR ROW_NUMBER() OVER (
    PARTITION BY tool_use_id ORDER BY timestamp, entry_id
) = 1
"""


# fact_tool_results: explode user messages' content[*] for tool_result blocks,
# then JOIN to the user entry's toolUseResult payload.
_PROJECT_RESULTS_SQL = """
WITH user_entries AS (
    SELECT
        sle.entry_id,
        sle.uuid AS message_id,
        sle.session_id,
        sle.source_path,
        TRY_CAST(sle.timestamp AS TIMESTAMP) AS timestamp,
        sle.message_json,
        sle.tool_use_result_json
    FROM etl.log_entries sle
    WHERE sle.type = 'user'
      AND json_type(sle.message_json, '$.content') = 'ARRAY'
),
exploded AS (
    SELECT
        ue.*,
        b.block
    FROM user_entries ue,
    LATERAL (
        SELECT unnest(json_extract(ue.message_json, '$.content')::JSON[]) AS block
    ) b
    WHERE json_extract_string(b.block, '$.type') = 'tool_result'
),
tool_name_map AS (
    -- Resolve tool_name by joining tool_use_id back to the assistant's
    -- tool_use blocks in staging. Scoped to sessions present in the
    -- current inbound batch so the scan stays O(batch), not O(archive).
    SELECT
        json_extract_string(b.block, '$.id') AS tool_use_id,
        json_extract_string(b.block, '$.name') AS tool_name
    FROM etl.log_entries sle,
    LATERAL (
        SELECT unnest(json_extract(sle.message_json, '$.content')::JSON[]) AS block
    ) b
    WHERE sle.type = 'assistant'
      AND json_type(sle.message_json, '$.content') = 'ARRAY'
      AND json_extract_string(b.block, '$.type') = 'tool_use'
      AND sle.session_id IN (SELECT DISTINCT session_id FROM user_entries)
),
with_tool_name AS (
    SELECT
        e.*,
        json_extract_string(e.block, '$.tool_use_id') AS tool_use_id,
        tnm.tool_name AS tool_name
    FROM exploded e
    LEFT JOIN tool_name_map tnm
      ON tnm.tool_use_id = json_extract_string(e.block, '$.tool_use_id')
)
SELECT
    entry_id, message_id, session_id, source_path, timestamp,
    tool_use_id, tool_name,

    -- is_error tri-state (R16): NULL when the block omits it
    CASE
        WHEN json_type(block, '$.is_error') = 'NULL' THEN NULL
        ELSE json_extract(block, '$.is_error')::BOOLEAN
    END AS is_error,

    -- result_content_text: the tool_result block's `content` (often truncated
    -- preview of the structured payload).
    CASE
        WHEN json_type(block, '$.content') = 'VARCHAR'
        THEN json_extract_string(block, '$.content')
        WHEN json_type(block, '$.content') = 'ARRAY'
        THEN list_aggregate(
            list_filter(
                list_transform(
                    COALESCE(json_extract(block, '$.content[*]')::JSON[],
                             CAST([] AS JSON[])),
                    bb -> CASE WHEN json_extract_string(bb, '$.type') = 'text'
                               THEN json_extract_string(bb, '$.text') END
                ),
                t -> t IS NOT NULL
            ),
            'string_agg', ' '
        )
        ELSE NULL
    END AS result_content_text,

    -- result_payload_json: the entry-level toolUseResult (full untruncated,
    -- structured payload). For string-shape toolUseResult (errors), the
    -- string is preserved as JSON.
    tool_use_result_json AS result_payload_json,

    -- Per-tool typed projections. NULL when the tool doesn't match.
    -- Bash / BashOutput. The payload states stdout, stderr, interrupted,
    -- isImage, noOutputExpected and (newer) dangerouslyDisableSandbox --
    -- never an exit code or a duration (0 of 52,974 real results). Those
    -- are derived below from the stated failure signal.
    CASE WHEN tool_name IN ('Bash', 'BashOutput')
         THEN length(json_extract_string(tool_use_result_json, '$.stdout')) END
        AS bash_stdout_bytes,

    -- Edit / MultiEdit
    CASE WHEN tool_name IN ('Edit', 'MultiEdit')
         THEN json_extract(tool_use_result_json, '$.userModified')::BOOLEAN END
        AS edit_user_modified,
    CASE WHEN tool_name IN ('Edit', 'MultiEdit')
         THEN json_extract(tool_use_result_json, '$.replaceAll')::BOOLEAN END
        AS edit_replace_all,
    CASE WHEN tool_name IN ('Edit', 'MultiEdit')
         THEN CAST(json_extract(tool_use_result_json, '$.structuredPatch') AS VARCHAR) END
        AS edit_structured_patch_json,

    -- Read
    CASE WHEN tool_name = 'Read'
         THEN json_extract(tool_use_result_json, '$.file.numLines')::INTEGER END
        AS read_num_lines,
    CASE WHEN tool_name = 'Read'
         THEN json_extract(tool_use_result_json, '$.file.totalLines')::INTEGER END
        AS read_total_lines,
    CASE WHEN tool_name = 'Read'
         THEN json_extract_string(tool_use_result_json, '$.file.filePath') END
        AS read_file_path,

    -- Write
    CASE WHEN tool_name = 'Write'
         THEN json_extract_string(tool_use_result_json, '$.type') END
        AS write_type,

    -- Glob
    CASE WHEN tool_name = 'Glob'
         THEN json_extract(tool_use_result_json, '$.numFiles')::INTEGER END
        AS glob_num_files,
    CASE WHEN tool_name = 'Glob'
         THEN json_extract(tool_use_result_json, '$.truncated')::BOOLEAN END
        AS glob_truncated,

    -- Grep
    CASE WHEN tool_name = 'Grep'
         THEN json_extract_string(tool_use_result_json, '$.mode') END
        AS grep_mode,
    CASE WHEN tool_name = 'Grep'
         THEN json_extract(tool_use_result_json, '$.numFiles')::INTEGER END
        AS grep_num_files,

    -- WebFetch
    CASE WHEN tool_name = 'WebFetch'
         THEN json_extract(tool_use_result_json, '$.code')::INTEGER END
        AS webfetch_http_code,
    CASE WHEN tool_name = 'WebFetch'
         THEN json_extract(tool_use_result_json, '$.bytes')::INTEGER END
        AS webfetch_bytes,

    -- Agent / Task subagent rollup
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract_string(tool_use_result_json, '$.status') END
        AS agent_status,
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract(tool_use_result_json, '$.totalDurationMs')::FLOAT END
        AS agent_total_duration_ms,
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract(tool_use_result_json, '$.totalTokens')::INTEGER END
        AS agent_total_tokens,
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract(tool_use_result_json, '$.totalToolUseCount')::INTEGER END
        AS agent_total_tool_use_count,
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract_string(tool_use_result_json, '$.agentType') END
        AS agent_subagent_type,
    -- agent_id from the toolUseResult is the link to dim_session.agent_id
    -- for the subagent that ran this delegation.
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract_string(tool_use_result_json, '$.agentId') END
        AS agent_id,
    -- resolvedModel is the model the subagent ACTUALLY ran on, and the
    -- parent's delegation row is the only place it is stated: 894 of 2,046
    -- agent sessions on the real corpus have no ingestible transcript of
    -- their own, so their model is otherwise unknowable.
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN json_extract_string(tool_use_result_json, '$.resolvedModel') END
        AS agent_resolved_model,
    -- isAsync is STATED in the payload (and, like is_error, only ever
    -- written as true -- absence means synchronous). Since Claude Code
    -- v2.1.198+ subagents run in the background by default, so the result
    -- returned at spawn time is a launch acknowledgment, not an outcome.
    -- Downstream uses this to refuse to present that acknowledgment as a
    -- completion.
    CASE WHEN tool_name IN ('Agent', 'Task', 'TaskCreate')
         THEN COALESCE(
             json_extract(tool_use_result_json, '$.isAsync')::BOOLEAN, FALSE)
         END
        AS agent_is_async
FROM with_tool_name

-- One row per tool_use_id, which is the grain this fact DECLARES via
-- natural_key="tool_use_id". A real Claude Code session records a single
-- tool_use_id under two distinct entry uuids (29 keys on a 2,344-session
-- corpus), so unnesting blocks across entries would otherwise emit two rows
-- for one key and break the uniqueness every consumer joins on.
--
-- Collapsing is safe HERE and only here, because the justification is
-- specific to this data: the duplicate records are content-identical on
-- every extracted column (is_error, result_content_text, timestamp,
-- tool_name, and every typed bash_*/edit_*/read_*/agent_* field). Only the
-- raw result_payload_json ever differs, on 8 of the 29. Earliest entry wins;
-- entry_id breaks ties so a rebuild from identical input is reproducible.
--
-- This used to be a generic collapse inside lineage_upsert, which applied
-- one fact's judgment to all 13 and resolved it silently. That helper now
-- ASSERTS this grain instead -- if this QUALIFY is removed, the ETL fails
-- loudly rather than silently double-counting.
-- NULL tool_use_id is left alone: SQL puts every NULL in ONE partition, so
-- collapsing here would silently drop all but one. lineage_upsert excludes
-- NULL from its uniqueness check for the same reason. (The column is NOT NULL
-- at the DDL level, so such a row fails loudly on insert -- which is the
-- correct outcome and must not be masked into a silent drop.)
QUALIFY tool_use_id IS NULL OR ROW_NUMBER() OVER (
    PARTITION BY tool_use_id ORDER BY timestamp, entry_id
) = 1
"""


# Derived from the STATED failure signal. `is_error` is what the source
# says; the error text carries the reason. 1,281 of 1,608 failed Bash calls
# on the corpus start with "Exit code N"; the rest are hook blocks,
# permission denials, user rejections, sibling failures, timeouts,
# interruptions, or a model outage. Tool-agnostic on purpose: an Edit whose
# string was not found classifies as 'other', not NULL.
_DERIVED_RESULT_SQL = """
    CASE WHEN is_error IS TRUE
         THEN TRY_CAST(regexp_extract(result_content_text, '^\\s*Exit code (\\d+)', 1) AS INTEGER)
    END AS derived_exit_code,
    CASE WHEN is_error IS NOT TRUE THEN NULL
         WHEN regexp_matches(result_content_text, '^\\s*Exit code \\d+') THEN 'nonzero_exit'
         WHEN result_content_text ILIKE '%Sibling tool call errored%' THEN 'sibling_errored'
         WHEN result_content_text ILIKE '%Blocked%' OR result_content_text ILIKE '%hook%' THEN 'blocked_by_hook'
         WHEN result_content_text ILIKE '%Permission%denied%'
           OR result_content_text ILIKE '%requires approval%'
           OR result_content_text ILIKE '%auto-denied%' THEN 'permission_denied'
         WHEN result_content_text ILIKE '%rejected%' OR result_content_text ILIKE '%doesn''t want%' THEN 'user_rejected'
         WHEN result_content_text ILIKE '%timed out%' THEN 'timeout'
         WHEN result_content_text ILIKE '%interrupted%' OR result_content_text ILIKE '%cancel%' THEN 'interrupted'
         WHEN result_content_text ILIKE '%temporarily unavailable%' THEN 'model_unavailable'
         ELSE 'other'
    END AS derived_failure_kind,
    COALESCE(TRY_CAST(json_extract(result_payload_json, '$.dangerouslyDisableSandbox') AS BOOLEAN), FALSE)
        AS sandbox_disabled
"""


# One chain per agentic run: the contiguous block of tool uses following one
# human turn, up to the next human turn. A human turn is a user message that
# carries real input, not a tool_result envelope and not meta. Partitioning
# on the assistant message instead put every tool use in its own chain of
# length 1 (Claude Code writes ONE content block per assistant entry, so
# parallel calls are separate entries) -- 71,175 of 71,216 steps at position
# 1 on a 2,250-session corpus. run_index 0 is honest for transcripts that
# open mid-flight. Reads the inbound uses/results, and fact_messages for
# entry order (populated earlier in the same run).
_CHAIN_SQL = """
WITH staged AS (
    SELECT DISTINCT session_id FROM etl.log_entries WHERE session_id IS NOT NULL
),
msg_seq AS (
    SELECT session_id, message_id, MIN(sequence_num) AS sequence_num
    FROM fact_messages
    WHERE is_deleted = FALSE AND session_id IN (SELECT session_id FROM staged)
    GROUP BY session_id, message_id
),
human_turns AS (
    SELECT DISTINCT session_id, sequence_num
    FROM fact_messages
    WHERE is_deleted = FALSE AND message_type = 'user'
      AND has_tool_result = FALSE AND is_meta = FALSE
      AND session_id IN (SELECT session_id FROM staged)
),
tool_events AS (
    SELECT u.tool_use_id, u.session_id, u.tool_key, u.timestamp,
           ms.sequence_num, u.invoke_sequence_num
    FROM _inbound_tool_uses u
    JOIN msg_seq ms ON ms.session_id = u.session_id AND ms.message_id = u.message_id
),
events AS (
    SELECT session_id, sequence_num, CAST(NULL AS TIMESTAMP) AS timestamp,
           1 AS is_human, CAST(NULL AS VARCHAR) AS tool_use_id,
           CAST(NULL AS VARCHAR) AS tool_key, 0 AS invoke_sequence_num
    FROM human_turns
    UNION ALL
    SELECT session_id, sequence_num, timestamp, 0, tool_use_id, tool_key, invoke_sequence_num
    FROM tool_events
),
runs AS (
    SELECT *, SUM(is_human) OVER (
        PARTITION BY session_id
        ORDER BY sequence_num, is_human DESC, invoke_sequence_num
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS run_index
    FROM events
),
ordered AS (
    SELECT tool_use_id, session_id, timestamp,
           md5(session_id || '|' || CAST(run_index AS VARCHAR)) AS chain_id,
           ROW_NUMBER() OVER w AS step_position,
           LAG(tool_key) OVER w AS prev_tool_key,
           LEAD(tool_key) OVER w AS next_tool_key,
           LAG(timestamp) OVER w AS prev_timestamp
    FROM runs
    WHERE tool_use_id IS NOT NULL
    WINDOW w AS (PARTITION BY session_id, run_index ORDER BY sequence_num, invoke_sequence_num)
)
SELECT tool_use_id, session_id, chain_id, step_position, prev_tool_key, next_tool_key,
       CASE WHEN prev_timestamp IS NOT NULL
            THEN EXTRACT(EPOCH FROM (timestamp - prev_timestamp)) END AS time_since_prev_seconds
FROM ordered
"""


def populate_fact_tool_calls(conn, *, run: EtlRun) -> None:
    """Project staging into one row per tool_use_id: use + result + chain."""
    conn.execute("DROP TABLE IF EXISTS _inbound_tool_uses")
    conn.execute(f"CREATE TEMP TABLE _inbound_tool_uses AS {_PROJECT_USES_SQL}")
    conn.execute("ALTER TABLE _inbound_tool_uses ADD COLUMN input_summary VARCHAR")
    conn.execute("ALTER TABLE _inbound_tool_uses ADD COLUMN project_key VARCHAR")
    conn.execute("ALTER TABLE _inbound_tool_uses ADD COLUMN tool_key VARCHAR")
    conn.execute("UPDATE _inbound_tool_uses SET input_summary = substring(input_json, 1, 200)")
    conn.execute("UPDATE _inbound_tool_uses SET tool_key = md5(tool_name)")
    conn.execute(
        "UPDATE _inbound_tool_uses "
        f"SET project_key = {project_key_sql('source_path')}"
    )

    conn.execute("DROP TABLE IF EXISTS _inbound_tool_results")
    conn.execute(
        "CREATE TEMP TABLE _inbound_tool_results AS "
        f"SELECT *, {_DERIVED_RESULT_SQL} FROM ({_PROJECT_RESULTS_SQL}) p"
    )

    conn.execute("DROP TABLE IF EXISTS _inbound_tool_chain")
    conn.execute(f"CREATE TEMP TABLE _inbound_tool_chain AS {_CHAIN_SQL}")

    typed = ", ".join(f"r.{c}" for c in _RESULT_TYPED_COLS)
    chain = ", ".join(f"c.{col}" for col in _CHAIN_COLS)
    conn.execute("DROP TABLE IF EXISTS _inbound_tool_calls")
    conn.execute(
        f"""
        CREATE TEMP TABLE _inbound_tool_calls AS
        SELECT
            u.tool_use_id, u.session_id, u.entry_id, u.message_id,
            u.project_key, u.tool_key, u.tool_name, u.invoke_sequence_num,
            u.input_json, u.input_summary, u.timestamp,
            r.entry_id AS result_entry_id,
            r.message_id AS result_message_id,
            r.timestamp AS result_timestamp,
            {typed},
            {chain}
        FROM _inbound_tool_uses u
        LEFT JOIN _inbound_tool_results r
               ON r.tool_use_id = u.tool_use_id AND r.session_id = u.session_id
        LEFT JOIN _inbound_tool_chain c
               ON c.tool_use_id = u.tool_use_id AND c.session_id = u.session_id
        """
    )

    lineage_upsert(
        conn,
        run=run,
        table="fact_tool_calls",
        inbound_table="_inbound_tool_calls",
        natural_key="tool_use_id",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
    )
