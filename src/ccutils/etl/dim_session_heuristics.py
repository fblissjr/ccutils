"""Enrich dim_session with heuristic classifications from v0.15 facts.

Reads first_user_message / last_assistant_message from stg_log_entries
(which has the raw content), and tool counts / message counts / error
counts / file extensions from the v0.15 fact + bridge tables. Runs the
classifiers in ccutils.etl.heuristics over each session and UPDATEs
dim_session in place.

Scope: only sessions currently in staging are enriched -- prior sessions
in dim_session retain their previous classification. Idempotent:
re-running on unchanged source produces identical results.

Run AFTER populate_fact_messages, populate_fact_tool_uses,
populate_fact_tool_results, populate_fact_file_operations, and
populate_bridge_session_file. (These supply the metrics + the file
extensions.)
"""

from __future__ import annotations

import json

from ccutils.etl.heuristics import (
    classify_complexity,
    classify_domain,
    classify_intent,
    classify_outcome,
)
from ccutils.etl.lineage import EtlRun


_MAX_MESSAGE_CHARS = 500


def _extract_text(content_json_raw: str | None) -> str:
    """Pull plain text out of a message.content payload.

    Claude Code emits user content as either a bare JSON string or a list
    of content-blocks; assistant content is always a list of blocks. We
    concatenate every text-bearing block (text / thinking). tool_result
    blocks are skipped -- they're tool output, not user intent / assistant
    conclusion.
    """
    if not content_json_raw:
        return ""
    try:
        content = json.loads(content_json_raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "thinking":
                parts.append(block.get("thinking", ""))
        return " ".join(p for p in parts if p)
    return ""


def populate_dim_session_heuristics(conn, *, run: EtlRun) -> None:
    """UPDATE dim_session with intent / complexity / outcome / domain
    + first_user_message + last_assistant_message for every session
    currently in staging."""
    # 1. Pull per-session inputs into Python. Limit to sessions in staging
    # so prior sessions' classifications stay intact (and aren't recomputed
    # against potentially different data).
    rows = conn.execute(
        """
        WITH staging_sessions AS (
            SELECT DISTINCT session_id FROM stg_log_entries
            WHERE session_id IS NOT NULL
        ),
        first_user AS (
            -- json_extract (not json_extract_string) so list content
            -- comes back as raw JSON the Python side can json.loads().
            SELECT sle.session_id,
                   CAST(json_extract(sle.message_json, '$.content') AS VARCHAR)
                       AS content_json
            FROM stg_log_entries sle
            JOIN staging_sessions ss USING (session_id)
            WHERE sle.type = 'user'
              AND COALESCE(sle.is_meta, FALSE) = FALSE
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY sle.session_id ORDER BY sle.timestamp, sle.sequence_num
            ) = 1
        ),
        last_assistant AS (
            SELECT sle.session_id,
                   CAST(json_extract(sle.message_json, '$.content') AS VARCHAR)
                       AS content_json
            FROM stg_log_entries sle
            JOIN staging_sessions ss USING (session_id)
            WHERE sle.type = 'assistant'
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY sle.session_id
                ORDER BY sle.timestamp DESC, sle.sequence_num DESC
            ) = 1
        ),
        tool_metrics AS (
            SELECT ftu.session_id,
                   COUNT(*) AS tool_count,
                   SUM(CASE WHEN ftr.is_error = TRUE THEN 1 ELSE 0 END)
                       AS error_count
            FROM fact_tool_uses ftu
            LEFT JOIN fact_tool_results ftr USING (tool_use_id)
            WHERE ftu.is_deleted = FALSE
              AND ftu.session_id IN (SELECT session_id FROM staging_sessions)
            GROUP BY ftu.session_id
        ),
        msg_metrics AS (
            SELECT session_id, COUNT(*) AS msg_count
            FROM fact_messages
            WHERE is_deleted = FALSE
              AND session_id IN (SELECT session_id FROM staging_sessions)
            GROUP BY session_id
        ),
        file_exts AS (
            SELECT bsf.session_id,
                   string_agg(DISTINCT df.file_extension, '|')
                       AS extensions_pipe
            FROM bridge_session_file bsf
            JOIN dim_file df USING (file_key)
            WHERE bsf.is_deleted = FALSE
              AND df.file_extension IS NOT NULL
              AND bsf.session_id IN (SELECT session_id FROM staging_sessions)
            GROUP BY bsf.session_id
        )
        SELECT
            ss.session_id,
            fu.content_json AS first_user_content_json,
            la.content_json AS last_assistant_content_json,
            COALESCE(tm.tool_count, 0) AS tool_count,
            COALESCE(mm.msg_count, 0) AS msg_count,
            COALESCE(tm.error_count, 0) AS error_count,
            fe.extensions_pipe
        FROM staging_sessions ss
        LEFT JOIN first_user fu USING (session_id)
        LEFT JOIN last_assistant la USING (session_id)
        LEFT JOIN tool_metrics tm USING (session_id)
        LEFT JOIN msg_metrics mm USING (session_id)
        LEFT JOIN file_exts fe USING (session_id)
        """
    ).fetchall()

    # 2. Apply classifiers in Python.
    updates = []
    for (
        session_id,
        first_user_content_json,
        last_assistant_content_json,
        tool_count,
        msg_count,
        error_count,
        extensions_pipe,
    ) in rows:
        first_user_text = _extract_text(first_user_content_json)
        last_assistant_text = _extract_text(last_assistant_content_json)
        error_rate = (error_count / tool_count) if tool_count else 0.0
        extensions = (
            [e for e in extensions_pipe.split("|") if e]
            if extensions_pipe
            else []
        )

        updates.append(
            (
                classify_intent(first_user_text),
                classify_complexity(tool_count, msg_count, 0, error_count),
                classify_outcome(last_assistant_text, error_rate=error_rate),
                classify_domain(extensions),
                first_user_text[:_MAX_MESSAGE_CHARS] if first_user_text else None,
                (
                    last_assistant_text[:_MAX_MESSAGE_CHARS]
                    if last_assistant_text
                    else None
                ),
                session_id,
            )
        )

    if not updates:
        return

    # 3. Batch UPDATE dim_session. Classifiers are deterministic over the
    # same inputs so re-runs produce identical results -- no hash_diff
    # guard needed.
    _ = run  # signature symmetry; classifiers don't stamp version_key
    conn.executemany(
        """
        UPDATE dim_session
        SET intent = ?,
            complexity = ?,
            outcome = ?,
            domain = ?,
            first_user_message = ?,
            last_assistant_message = ?
        WHERE session_id = ?
        """,
        updates,
    )
