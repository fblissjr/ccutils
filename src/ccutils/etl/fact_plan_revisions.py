"""Populate fact_plan_revisions from fact_tool_uses + fact_tool_results.

One row per ExitPlanMode tool_use. Outcome is classified with a
two-tier signal hierarchy:

1. STRUCTURAL (preferred): fact_tool_results.is_error tri-state nullable
   BOOLEAN (R16). When is_error=FALSE -> accepted, is_error=TRUE -> rejected.
2. CONTENT (fallback when is_error IS NULL): match the result_content_text
   against the documented Claude Code approval-signature phrase. Real
   sessions emit accepted plans without an explicit is_error field, so
   this fallback is necessary -- but unlike v0.14 we read the FULL
   content from fact_tool_results.result_content_text rather than the
   2000-char-truncated fact_tool_calls.output_text.

Classification (priority order):
- 'superseded' -- a later ExitPlanMode exists in the same session
- 'accepted'   -- is_error=FALSE OR content matches approval signature
- 'rejected'   -- is_error=TRUE
- 'pending'    -- no tool_result yet (session still in flight)
- 'unknown'    -- tool_result present but no usable signal

parent_revision_key chains revisions within a session by timestamp.
user_feedback_text captures the next user text message after a rejection.

Run AFTER populate_fact_tool_uses + populate_fact_tool_results.
"""

from __future__ import annotations

from ccutils.etl.lineage import EtlRun
from ccutils.etl.upsert import lineage_upsert


# Approval signature emitted by Claude Code on accepted ExitPlanMode
# results. Used only as fallback when is_error IS NULL (which is the
# common case in real sessions -- Claude Code typically omits the
# is_error field for accepted plans rather than emitting false).
_APPROVAL_SIGNATURE = "approved your plan"


_PAYLOAD_COLS = [
    "tool_use_id", "timestamp", "plan_timestamp", "resolved_timestamp",
    "seconds_to_resolution",
    "revision_number", "parent_revision_key",
    "plan_text", "plan_file_path", "plan_char_count",
    "outcome", "outcome_signal",
    "user_feedback_message_id", "user_feedback_text",
]
_HASH_COLS = [
    "tool_use_id", "plan_timestamp", "resolved_timestamp",
    "revision_number", "parent_revision_key",
    "plan_text", "plan_file_path", "outcome", "outcome_signal",
    "user_feedback_message_id", "user_feedback_text",
]


def populate_fact_plan_revisions(conn, *, run: EtlRun) -> None:
    """Derive one fact_plan_revisions row per ExitPlanMode tool_use."""
    conn.execute("DROP TABLE IF EXISTS _inbound_plan_revisions")
    approval_pattern = f"%{_APPROVAL_SIGNATURE}%"
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_plan_revisions AS
        WITH plan_calls AS (
            -- Every ExitPlanMode invocation in sessions currently in staging.
            SELECT
                ftu.tool_use_id,
                ftu.session_id,
                ftu.timestamp AS plan_timestamp,
                ftu.message_id AS invoke_message_id,
                json_extract_string(ftu.input_json, '$.plan') AS plan_text,
                json_extract_string(ftu.input_json, '$.planFilePath')
                    AS plan_file_path,
                ROW_NUMBER() OVER (
                    PARTITION BY ftu.session_id ORDER BY ftu.timestamp
                ) AS revision_number,
                COUNT(*) OVER (
                    PARTITION BY ftu.session_id
                ) AS revisions_in_session
            FROM fact_tool_uses ftu
            JOIN dim_tool dt USING (tool_key)
            WHERE ftu.is_deleted = FALSE
              AND dt.tool_name = 'ExitPlanMode'
              AND ftu.session_id IN (
                  SELECT DISTINCT session_id FROM stg_log_entries
                  WHERE session_id IS NOT NULL
              )
        ),
        with_results AS (
            -- Pair each plan_call to its tool_result (may be missing if
            -- the session ended mid-flight -> pending).
            SELECT
                pc.*,
                ftr.is_error,
                ftr.result_content_text,
                ftr.timestamp AS resolved_timestamp,
                ftr.message_id AS result_message_id
            FROM plan_calls pc
            LEFT JOIN fact_tool_results ftr USING (tool_use_id)
        ),
        with_outcome AS (
            SELECT
                wr.*,
                CASE
                    -- A later ExitPlanMode in the same session always wins;
                    -- this revision is superseded regardless of how its
                    -- own tool_result classified.
                    WHEN wr.revision_number < wr.revisions_in_session
                        THEN 'superseded'
                    WHEN wr.resolved_timestamp IS NULL THEN 'pending'
                    WHEN wr.is_error = FALSE THEN 'accepted'
                    WHEN wr.is_error = TRUE  THEN 'rejected'
                    -- is_error IS NULL: fall back to content signature.
                    WHEN wr.result_content_text LIKE ?
                        THEN 'accepted'
                    ELSE 'unknown'
                END AS outcome,
                CASE
                    WHEN wr.revision_number < wr.revisions_in_session
                        THEN 'later_plan_exists'
                    WHEN wr.resolved_timestamp IS NULL THEN 'no_resolution'
                    WHEN wr.is_error = FALSE THEN 'is_error=FALSE'
                    WHEN wr.is_error = TRUE  THEN 'is_error=TRUE'
                    WHEN wr.result_content_text LIKE ?
                        THEN 'approval_signature'
                    ELSE 'is_error_null'
                END AS outcome_signal
            FROM with_results wr
        ),
        -- For rejected revisions: capture the next user text message in
        -- the session as user_feedback_text. Subquery via fact_messages,
        -- matched by session + timestamp(next user-message-with-text).
        with_feedback AS (
            SELECT
                wo.*,
                -- Pick the first user message with text content whose
                -- timestamp is strictly after the tool_result.
                (SELECT fm.message_id
                 FROM fact_messages fm
                 WHERE fm.session_id = wo.session_id
                   AND fm.message_type = 'user'
                   AND fm.timestamp > wo.resolved_timestamp
                   AND fm.is_deleted = FALSE
                 ORDER BY fm.timestamp
                 LIMIT 1) AS user_feedback_message_id
            FROM with_outcome wo
            WHERE wo.outcome = 'rejected'
        ),
        -- Pull the feedback text from staging (fact_messages doesn't
        -- carry content; the v0.15 model keeps text in stg_log_entries).
        with_feedback_text AS (
            SELECT
                wf.*,
                json_extract_string(sle.message_json, '$.content')
                    AS user_feedback_text_candidate
            FROM with_feedback wf
            LEFT JOIN stg_log_entries sle
                ON sle.uuid = wf.user_feedback_message_id
        )
        SELECT
            -- revision_key = md5(session_id || '|' || tool_use_id) so it's
            -- stable across re-ETL.
            md5(wo.session_id || '|' || wo.tool_use_id) AS revision_key,
            wo.tool_use_id,
            wo.session_id,
            wo.plan_timestamp AS timestamp,
            wo.plan_timestamp,
            wo.resolved_timestamp,
            CASE WHEN wo.resolved_timestamp IS NOT NULL THEN
                EXTRACT(EPOCH FROM (wo.resolved_timestamp - wo.plan_timestamp))
            ELSE NULL END AS seconds_to_resolution,
            wo.revision_number,
            CASE WHEN wo.revision_number = 1 THEN NULL
                 ELSE md5(
                     wo.session_id || '|' ||
                     (SELECT inner_pc.tool_use_id
                      FROM plan_calls inner_pc
                      WHERE inner_pc.session_id = wo.session_id
                        AND inner_pc.revision_number = wo.revision_number - 1)
                 )
            END AS parent_revision_key,
            wo.plan_text,
            wo.plan_file_path,
            length(wo.plan_text) AS plan_char_count,
            wo.outcome,
            wo.outcome_signal,
            wft.user_feedback_message_id,
            wft.user_feedback_text_candidate AS user_feedback_text
        FROM with_outcome wo
        LEFT JOIN with_feedback_text wft USING (tool_use_id)
        """,
        [approval_pattern, approval_pattern],
    )
    lineage_upsert(
        conn, run=run,
        table="fact_plan_revisions",
        inbound_table="_inbound_plan_revisions",
        natural_key="revision_key",
        payload_cols=_PAYLOAD_COLS,
        hash_cols=_HASH_COLS,
        timestamp_col="plan_timestamp",
    )
