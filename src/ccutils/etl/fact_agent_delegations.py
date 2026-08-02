"""Populate fact_agent_delegations from fact_tool_uses + fact_tool_results.

One row per Task tool_use (parent-side agent spawn). Captures the task
input (description, prompt, subagent_type) from fact_tool_uses.input_json
and the agent rollup metrics (status, totalDurationMs, totalTokens,
totalToolUseCount, wasInterrupted) from fact_tool_results.agent_*
columns -- the v0.15 R1 structured toolUseResult capture.

parent_session_key is md5 of the parent session_id; agent_session_key is
md5('agent-' || agent_id), derived from the natural key rather than looked
up in dim_session so it does not depend on whether the agent's own
transcript has been ETL'd yet. Either may point at a session absent from
dim_session -- LEFT JOIN accordingly.

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
    "agent_resolved_model", "agent_is_async",
]
_HASH_COLS = [
    # agent_session_key IS hashed: it was previously excluded, so once a row
    # was written with a NULL key no later run could ever repair it (hash
    # unchanged -> no update). Including it lets a re-run heal old rows.
    "tool_use_id", "agent_session_key",
    "delegation_timestamp", "completion_timestamp",
    "task_description", "task_prompt", "subagent_type",
    "agent_status", "agent_total_duration_ms",
    "agent_total_tokens", "agent_total_tool_use_count",
    "agent_was_interrupted", "agent_output_text",
    "agent_resolved_model", "agent_is_async",
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
            -- agent_session_key is DERIVED, not looked up. session_key is
            -- md5(session_id) and an agent's session_id is
            -- 'agent-<agent_id>' (holds for all 2,046 agent sessions on the
            -- real corpus), so the key needs neither a join nor an ordering
            -- guarantee.
            --
            -- It used to resolve through a correlated subquery on
            -- dim_session.agent_id, which only finds a row if the agent's
            -- OWN transcript was already ETL'd. ETL is per-session and the
            -- parent is normally processed first, so on the real corpus
            -- this produced NULL for 941 of 941 delegations -- 826 of them
            -- carrying a subagent_type, 936 agent sessions left unlinked.
            -- agent_session_key is also excluded from _HASH_COLS, so a
            -- later run never repaired it: hash unchanged, no update.
            --
            -- The key may point at a session that was never ingested (30 of
            -- 880 on the real corpus). That is the normal degenerate-key
            -- situation -- consumers LEFT JOIN dim_session already.
            md5('agent-' || ftr.agent_id) AS agent_session_key,
            ftu.timestamp,
            ftu.timestamp AS delegation_timestamp,
            -- On a background launch the tool result is an acknowledgment
            -- that lands milliseconds after the spawn, so neither of these
            -- describes the agent finishing. Measured on the real corpus:
            -- median seconds_to_completion 2.05s across 719 async rows
            -- versus 102.45s across the 192 that actually completed, mixed
            -- together with nothing marking which was which. NULL is
            -- honest; a plausible wrong number silently poisons every
            -- aggregate, and biased toward under-reporting exactly the
            -- long-running delegations. Re-deriving the true values from
            -- the agent's own transcript is separate work -- see
            -- internal/plans/2026-08-01_agent_delegation_capture_gap.md.
            CASE WHEN ftr.agent_is_async IS TRUE THEN NULL
                 ELSE ftr.timestamp END AS completion_timestamp,
            CASE WHEN ftr.agent_is_async IS TRUE THEN NULL
                 WHEN ftr.timestamp IS NOT NULL THEN
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
            ftr.agent_resolved_model,
            ftr.agent_is_async,
            -- Tool result content can be a list of blocks (Agent typically
            -- emits one text block); fall back to result_content_text when
            -- the parser flattened it to a plain string.
            -- Same reasoning: on a background launch this is the string
            -- "Async agent launched successfully...", not the agent's work.
            CASE WHEN ftr.agent_is_async IS TRUE THEN NULL ELSE
                COALESCE(
                    ftr.result_content_text,
                    CAST(json_extract(ftr.result_payload_json, '$.content')
                         AS VARCHAR)
                )
            END AS agent_output_text
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


_COMPLETION_SQL = """
CREATE TEMP TABLE _inbound_delegation_completion AS
WITH agent_rollup AS (
    -- One row per AGENT session, from that agent's OWN fact rows. The agent
    -- transcript is ingested as its own session (session_id 'agent-<id>'),
    -- so nothing is re-read from disk and the backfill is retroactive.
    SELECT
        fm.session_key AS agent_session_key,
        SUM(COALESCE(fm.input_tokens, 0)
            + COALESCE(fm.output_tokens, 0))          AS tokens,
        MIN(fm.timestamp)                             AS first_ts,
        MAX(fm.timestamp)                             AS last_ts,
        -- Terminality is decided by comparing timestamps rather than with
        -- arg_max(stop_reason, timestamp): arg_max skips NULL values, so an
        -- unfinished agent whose EARLIER turn had a stop_reason would be
        -- reported as terminal. These two maxima are equal only when the
        -- last assistant record itself carries a stop_reason.
        MAX(fm.timestamp) FILTER (
            WHERE fm.message_type = 'assistant')      AS last_assistant_ts,
        MAX(fm.timestamp) FILTER (
            WHERE fm.message_type = 'assistant'
              AND fm.stop_reason IS NOT NULL)         AS last_terminal_ts
    FROM fact_messages fm
    WHERE fm.is_deleted = FALSE
    GROUP BY fm.session_key
),
agent_tools AS (
    SELECT session_key AS agent_session_key, COUNT(*) AS tool_uses
    FROM fact_tool_uses WHERE is_deleted = FALSE GROUP BY session_key
),
scored AS (
    SELECT
        -- EXCLUDE is load-bearing: the target already HAS a completion_state
        -- column, so a bare d.* would put two columns of that name in this
        -- CTE and the outer SELECT would silently read the stale NULL one
        -- instead of the value computed below.
        d.* EXCLUDE (completion_state),
        ar.tokens              AS ar_tokens,
        ar.last_assistant_ts   AS ar_last_ts,
        ar.last_terminal_ts    AS ar_terminal_ts,
        ar.first_ts            AS ar_first_ts,
        agt.tool_uses          AS ar_tool_uses,
        CASE
            -- No agent id at all AND no status AND the result is a STATED
            -- error: the spawn was refused, so no agent ever existed. 29 of
            -- the 30 NULL-status rows on the real corpus (fork-inside-fork,
            -- depth limit 3 of 3, cancellation, validation error, user
            -- rejection). Distinct from "finished but unmeasured" --
            -- collapsing them loses a real signal.
            --
            -- The is_error gate is load-bearing, not belt-and-braces. Without
            -- it the branch also swallowed the 30th row, a successful
            -- background launch reading "Fork started - processing in
            -- background" that merely carried no agentId. All 29 genuine
            -- failures state is_error: true; that one omits the field. Per
            -- the settled tri-state rule an omitted is_error means
            -- not-an-error, so IS TRUE separates them on a stated fact
            -- rather than on the result text.
            WHEN d.agent_status IS NULL AND d.agent_session_key IS NULL
                 AND tr.is_error IS TRUE
                THEN 'spawn_failed'
            -- Synchronous: the parent genuinely saw the outcome, and those
            -- rollups are already correct. Never recomputed.
            WHEN d.agent_is_async IS NOT TRUE AND d.agent_status IS NOT NULL
                THEN 'completed'
            -- Async but the agent's transcript is not in the warehouse. We
            -- cannot say what happened; NULL means "not reconciled", which
            -- is different from having observed it unfinished.
            WHEN ar.agent_session_key IS NULL THEN NULL
            WHEN ar.last_terminal_ts IS NOT NULL
             AND ar.last_terminal_ts = ar.last_assistant_ts THEN 'completed'
            -- The agent's transcript records no completion. That is ALL this
            -- says. It was called `in_flight_at_ingest`, which asserted the
            -- agent was still running -- a claim the transcript cannot
            -- support and which the corpus contradicts: of 101 rows carrying
            -- it, 98 ended mid-tool-loop a median of 15.7 DAYS before the ETL
            -- ran, and only 2 had a last message recent enough to still be
            -- live. Naming the observation rather than the inference also
            -- removes the reason `abandoned` was withheld: no staleness
            -- threshold is needed to state that nothing was recorded.
            --
            -- Known limit, measured: on 188 agents the parent SAW complete,
            -- 19 (10.1%) still fail the terminal test, so this state also
            -- absorbs a ~10% false-negative rate. Two alternative predicates
            -- (last non-null stop_reason terminal; any end_turn present) miss
            -- exactly the same 19, so the shortfall is in the transcripts,
            -- not the predicate. Erring toward "not recorded" keeps the
            -- rollups NULL rather than inventing them.
            ELSE 'no_completion_recorded'
        END AS completion_state
    FROM fact_agent_delegations d
    LEFT JOIN agent_rollup ar ON ar.agent_session_key = d.agent_session_key
    LEFT JOIN agent_tools  agt ON agt.agent_session_key = d.agent_session_key
    -- One row per (session, tool use) is guaranteed by the QUALIFY in
    -- _PROJECT_RESULTS_SQL, which makes fact_tool_results unique on
    -- tool_use_id by construction. It is NOT guaranteed by lineage_upsert,
    -- which only asserts the key and no longer collapses duplicates.
    LEFT JOIN fact_tool_results tr
           ON tr.tool_use_id = d.tool_use_id
          AND tr.session_id = d.session_id
          AND tr.is_deleted = FALSE
    WHERE d.is_deleted = FALSE
)
SELECT
    delegation_key, tool_use_id, session_id,
    parent_session_key, agent_session_key,
    timestamp, delegation_timestamp,
    task_description, task_prompt, subagent_type,
    agent_was_interrupted, agent_resolved_model, agent_is_async,
    agent_status, agent_output_text,
    completion_state,
    -- Rollups are written ONLY for a completed delegation. For in_flight the
    -- agent file holds a partial sum, which is indistinguishable from a fast
    -- agent and poisons any aggregate silently; NULL is the honest value.
    CASE WHEN completion_state = 'completed' AND agent_is_async IS TRUE
              THEN ar_last_ts
         WHEN completion_state = 'completed' THEN completion_timestamp
    END AS completion_timestamp,
    CASE WHEN completion_state = 'completed' AND agent_is_async IS TRUE
              THEN EXTRACT(EPOCH FROM (ar_last_ts - delegation_timestamp))
         WHEN completion_state = 'completed' THEN seconds_to_completion
    END AS seconds_to_completion,
    -- agent_total_tokens is the API's STATED number or nothing. The derived
    -- sum is NOT a substitute for it and must never be written here.
    --
    -- Ground truth, 188 synchronous delegations carrying both a stated
    -- rollup and an ingested agent transcript: the duration derivation
    -- matched 188/188 and the tool-count derivation 188/188, but tokens
    -- matched only 12/188 within 10%, per-row ratio p10 0.063 to p90 1.004.
    -- Nothing reconciles them -- not in+out, out alone, +cache_creation,
    -- +cache_read, total_uncached_equivalent, nor the 5m/1h splits. It is
    -- not a capture gap (median 23 assistant records, 23 carrying usage) and
    -- not a nested-agent rollup (all 188 spawned none). The two count
    -- different things, so a single column cannot hold both: merging them
    -- made async delegations read 3x cheaper than synchronous ones (median
    -- 19,444 vs 61,362) while measuring 2x longer.
    CASE WHEN completion_state = 'completed' AND agent_is_async IS NOT TRUE
              THEN agent_total_tokens
    END AS agent_total_tokens,
    -- The derived measurement, kept in its own column so it can be used
    -- without being mistaken for the API's. Written only for a completed
    -- delegation, on the same reasoning as every other rollup here: a
    -- partial sum is indistinguishable from a fast agent.
    CASE WHEN completion_state = 'completed' THEN ar_tokens
    END AS agent_derived_io_tokens,
    CASE WHEN completion_state = 'completed' AND agent_is_async IS TRUE
              THEN EXTRACT(EPOCH FROM (ar_last_ts - ar_first_ts)) * 1000
         WHEN completion_state = 'completed' THEN agent_total_duration_ms
    END AS agent_total_duration_ms,
    CASE WHEN completion_state = 'completed' AND agent_is_async IS TRUE
              THEN ar_tool_uses
         WHEN completion_state = 'completed' THEN agent_total_tool_use_count
    END AS agent_total_tool_use_count
FROM scored
"""


def populate_delegation_completion(conn, *, run) -> None:
    """Re-derive async delegation rollups from the AGENT's own transcript.

    Roadmap 0e step 2. Since Claude Code v2.1.198+ subagents run in the
    background by default, so the tool result the parent receives at spawn
    time is a launch acknowledgment, not an outcome -- 721 of 943 delegations
    on the real corpus carry no tokens and no duration. Everything needed to
    repair that is already in the warehouse: the agent's transcript is
    ingested as its own session and carries usage and stop_reason.

    This is a POST-LOOP pass, not a per-session populator, and that is
    load-bearing. ``run_v15_etl`` processes sessions in arbitrary order (and
    in parallel), so when a parent is ETL'd its agent's rows usually do not
    exist yet. Deriving these inside the per-session populator would rebuild
    exactly the ordering dependency that left ``agent_session_key`` NULL on
    941 of 941 rows. Call it once, after the session loop, from BOTH entry
    points -- ``local`` and ``all`` diverging on post-ETL steps is a bug this
    project has already shipped once.

    Scope is every non-deleted delegation rather than the staged sessions:
    staging is per-session and already cleared by the time this runs, and the
    whole point is cross-session. Inbound therefore covers every row, which
    also makes the soft-delete pass a no-op instead of deleting the rows a
    narrower scope would have omitted.
    """
    conn.execute("DROP TABLE IF EXISTS _inbound_delegation_completion")
    conn.execute(_COMPLETION_SQL)
    lineage_upsert(
        conn, run=run,
        table="fact_agent_delegations",
        inbound_table="_inbound_delegation_completion",
        natural_key="delegation_key",
        # agent_derived_io_tokens is appended here rather than added to
        # _PAYLOAD_COLS: that list is shared with the base populator, whose
        # inbound table has no such column.
        payload_cols=_PAYLOAD_COLS
        + ["completion_state", "agent_derived_io_tokens"],
        hash_cols=_HASH_COLS + ["completion_state", "agent_derived_io_tokens"],
        timestamp_col="delegation_timestamp",
    )
