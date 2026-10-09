# Worked query recipes

Copy and adapt. All assume `duckdb -readonly <db>`. Raw-fact queries already
include the `is_deleted = FALSE` filter where it matters.

## Cost and tokens

```sql
-- Sessions ranked by real (uncached-equivalent) token cost
SELECT ds.session_id, fss.total_input_tokens, fss.total_output_tokens,
       fss.total_cache_creation_5m_tokens, fss.total_cache_creation_1h_tokens,
       fss.total_cache_read_tokens, fss.total_uncached_equivalent_tokens
FROM semantic_session_summary fss
JOIN dim_session ds USING (session_key)
ORDER BY total_uncached_equivalent_tokens DESC LIMIT 20;

-- Token usage by model (view pre-filters soft deletes)
SELECT model_name, SUM(input_tokens) AS total_in, SUM(output_tokens) AS total_out,
       SUM(cache_read_tokens) AS total_cache_read
FROM semantic_token_usage GROUP BY model_name;

-- Cache efficiency per session (R11-corrected denominator)
SELECT session_id, cache_hit_rate_pct, total_uncached_equivalent_tokens
FROM semantic_cost_analysis ORDER BY total_uncached_equivalent_tokens DESC LIMIT 20;
```

## Sessions and projects

```sql
-- What kinds of sessions do I have?
SELECT intent, complexity, outcome, COUNT(*) AS sessions
FROM semantic_sessions GROUP BY ALL ORDER BY sessions DESC;

-- Catch up on one project
SELECT session_id, intent, complexity, outcome,
       first_user_message, last_assistant_message,
       total_messages, total_tool_uses, total_tool_errors
FROM semantic_project_context
WHERE project_name = 'my-project'
ORDER BY created_at DESC LIMIT 10;

-- Resumed chains: longest multi-session arcs
SELECT slug, session_count, total_duration_seconds
FROM semantic_session_chains ORDER BY session_count DESC LIMIT 10;
```

## Tools and errors

```sql
-- Bash invocations that failed or were interrupted
SELECT ftu.session_id,
       json_extract_string(ftu.input_json, '$.command') AS command,
       ftu.derived_exit_code, ftu.derived_failure_kind, ftu.result_timestamp
FROM fact_tool_calls ftu
WHERE ftu.tool_name = 'Bash'
  AND ftu.is_error
  AND ftu.is_deleted = FALSE
ORDER BY ftu.result_timestamp DESC LIMIT 20;

-- Failure taxonomy by tool
SELECT dt.tool_name, fe.derived_failure_kind, COUNT(*) AS n
FROM fact_tool_calls fe JOIN dim_tool dt USING (tool_key)
WHERE fe.is_deleted = FALSE AND fe.is_error
GROUP BY ALL ORDER BY n DESC LIMIT 20;

-- Tool→tool transitions most likely to error
SELECT t1.tool_name, t2.tool_name AS next_tool, COUNT(*) AS freq,
       SUM(CASE WHEN fcs.is_error THEN 1 ELSE 0 END) AS errors
FROM fact_tool_calls fcs
JOIN dim_tool t1 ON fcs.tool_key = t1.tool_key
JOIN dim_tool t2 ON fcs.next_tool_key = t2.tool_key
WHERE fcs.is_deleted = FALSE
GROUP BY ALL HAVING COUNT(*) >= 5 ORDER BY freq DESC LIMIT 20;
```

## Files

```sql
-- Most-modified files overall
SELECT df.file_path, SUM(bsf.write_count + bsf.edit_count) AS modifications
FROM semantic_session_files bsf JOIN dim_file df USING (file_key)
GROUP BY df.file_path ORDER BY modifications DESC LIMIT 20;

-- Files touched across the most sessions (hotspots)
SELECT file_path, session_count, total_edits
FROM semantic_file_evolution ORDER BY session_count DESC LIMIT 20;
```

## Plans and decisions

```sql
-- Sessions where the first plan was rejected and iterated
SELECT session_id, project_name, revision_number, outcome,
       seconds_to_resolution, SUBSTR(plan_text, 1, 80) AS plan_preview
FROM semantic_plan_revisions
WHERE revision_number > 1 ORDER BY plan_date DESC, revision_number;

-- Unified decision timeline for one session
SELECT timestamp, decision_type, decision_value, decision_detail
FROM semantic_decisions WHERE session_id = '<id>' ORDER BY timestamp;
```

## Subagents

```sql
-- Delegations with rollup metrics.
-- agent_* columns are what the parent's tool result STATED; derived_* columns
-- come from the agent's own transcript. No column holds both. On a
-- background launch (agent_is_async) the parent states no duration, tokens
-- or tool count, so read the derived_ columns there.
-- The two token columns are DIFFERENT MEASURES and must never be summed
-- together or compared: agent_total_tokens is the API's stated number;
-- derived_io_tokens is input+output summed from the agent's own usage.
-- completion_state is derived from the agent's transcript (completed /
-- no_completion_recorded / spawn_failed / NULL = transcript not in the
-- warehouse); every derived_ column is NULL unless it is 'completed'.
SELECT parent_session_id, subagent_type, task_description,
       agent_is_async,
       agent_status,              -- stated by the parent
       agent_total_duration_ms,   -- stated; NULL when agent_is_async
       agent_total_tokens,        -- stated; NULL when agent_is_async
       completion_state,          -- derived
       derived_duration_ms,
       derived_io_tokens          -- NOT comparable with agent_total_tokens
FROM semantic_agent_delegations ORDER BY delegation_timestamp DESC LIMIT 20;

-- Cost-style aggregate: pick ONE measure and say which.
SELECT subagent_type,
       COUNT(*) AS delegations,
       CAST(median(derived_io_tokens) AS BIGINT) AS median_io_tokens
FROM semantic_agent_delegations
WHERE completion_state = 'completed'
GROUP BY 1 ORDER BY delegations DESC;

-- How much each session delegated (from the summary, no join needed)
SELECT session_id, total_delegations, total_spawn_failures,
       delegated_output_tokens
FROM semantic_session_summary
WHERE total_delegations > 0 ORDER BY total_delegations DESC LIMIT 20;

-- Agent tree under a parent session
SELECT session_id, agent_type, agent_description, depth_level
FROM dim_session
WHERE parent_session_key = md5('<parent-session-uuid>')
ORDER BY first_timestamp;
```

## Facets (EAV shape)

One row per (session, facet, prompt_version); exactly ONE `value_*` column is
populated per row, chosen by `dim_facet_type.output_type` (text/enum→`value_text`,
json→`value_json`, int/float→`value_numeric`, bool→`value_bool`).

```sql
-- Tier 2 task descriptions (F20), skipping fallback rows
SELECT fsf.session_id, fsf.value_text AS task_description
FROM fact_session_facets fsf
JOIN dim_facet_type dft USING (facet_type_key)
WHERE dft.facet_id = 'F20' AND fsf.is_fallback = FALSE
  AND fsf.is_deleted = FALSE;

-- Tier 1 tool-mix histograms (F06, json-valued)
SELECT session_id, value_json FROM fact_session_facets fsf
JOIN dim_facet_type dft USING (facet_type_key)
WHERE dft.facet_id = 'F06' AND fsf.is_deleted = FALSE;
```

Facet id → meaning: `docs/FACET_CLUSTER_PIPELINE.md` §3.

## ETL observability

```sql
-- Per-session run health: status, duration, CDC window, row rollups
SELECT etl_run_id, batch_run_id, status, duration_ms,
       data_start_ts, data_end_ts, step_count, rows_inserted, rows_updated,
       ccutils_version
FROM semantic_etl_runs ORDER BY started_at DESC LIMIT 10;

-- What did each CLI invocation do (batch grain)
SELECT status, sessions_seen, sessions_succeeded, sessions_failed,
       rows_inserted, output_format
FROM etl.batch_runs ORDER BY started_at DESC LIMIT 5;
```

## Entry events and system events

```sql
-- Changes of permission mode in a session. NOT "every permission-mode row":
-- the source restates the current mode far more often than it changes it,
-- and writes no timestamp on those entries. semantic_decisions keeps only
-- the changes; order by COALESCE(timestamp, derived_timestamp).
SELECT COALESCE(timestamp, derived_timestamp) AS changed_at,
       decision_value AS new_mode
FROM semantic_decisions
WHERE decision_type = 'permission_mode_change' AND session_id = '<id>'
ORDER BY changed_at;

-- Every permission-mode entry as written, in file order. `timestamp` is NULL
-- on these rows (and on custom-title, agent-name, last-prompt): sequence_num
-- is the only order they have, and derived_timestamp places each one by the
-- timestamped entry before it.
SELECT sequence_num, derived_timestamp, value_text AS permission_mode
FROM fact_entry_events
WHERE entry_type = 'permission-mode' AND session_id = '<id>'
  AND is_deleted = FALSE
ORDER BY sequence_num;

-- What a session's non-message entries were, by kind. fact_entry_events
-- holds attachments, the permission-mode / custom-title / agent-name /
-- last-prompt series, file-history snapshots, queue operations and PR links:
-- entry_type says which, subtype and value_text carry what each states, and
-- payload_json holds the rest (e.g. a PR's number and repository).
SELECT entry_type, subtype, COUNT(*) AS entries
FROM fact_entry_events
WHERE session_id = '<id>' AND is_deleted = FALSE
GROUP BY ALL ORDER BY entries DESC;

-- PRs linked from sessions
SELECT session_id, value_text AS pr_url,
       json_extract(payload_json, '$.prNumber')::INTEGER AS pr_number,
       json_extract_string(payload_json, '$.prRepository') AS pr_repository
FROM fact_entry_events
WHERE entry_type = 'pr-link' AND is_deleted = FALSE
ORDER BY timestamp DESC LIMIT 20;

-- Compactions and API errors per session
SELECT session_id, subtype, COUNT(*) AS n
FROM fact_system_events
WHERE subtype IN ('compact_boundary', 'api_error') AND is_deleted = FALSE
GROUP BY ALL ORDER BY n DESC;
```

## Auto memory (what Claude has learned)

`dim_memory` is a Type 2 SCD -- one row per (memory file, content VERSION).
Filter `is_current` for present state, or use `semantic_memory`, which already
does. Query `dim_memory` directly only when you want history.

```sql
-- What Claude currently believes about each project, densest first
SELECT project_name, memory_type, memory_name, description, body_chars
FROM semantic_memory
WHERE NOT is_index
ORDER BY project_name, body_chars DESC;

-- Which memories have actually evolved, and how much
SELECT owner_key, file_name,
       COUNT(*)        AS versions,
       MIN(valid_from) AS first_seen,
       MAX(CASE WHEN is_current THEN modified_at END) AS last_written
FROM dim_memory
GROUP BY ALL HAVING COUNT(*) > 1
ORDER BY versions DESC;

-- What one memory used to say (point-in-time read)
SELECT version_num, valid_from, valid_to, body_text
FROM dim_memory
WHERE file_name = 'feedback_timeout_defaults.md'
ORDER BY version_num;

-- Memory volume by type and project
SELECT project_name, memory_type, COUNT(*) AS memories, SUM(body_chars) AS chars
FROM semantic_memory GROUP BY ALL ORDER BY chars DESC;

-- Which kinds of session produced memories (what teaches Claude something)
SELECT origin_intent, COUNT(*) AS memories_written
FROM semantic_memory
WHERE origin_session_id IS NOT NULL
GROUP BY ALL ORDER BY memories_written DESC;
```

The link graph carries two edge kinds -- keep them apart (see gotchas):

```sql
-- What the index catalogues, with the label it uses
SELECT project_name, link_text, target_file_name, is_resolved
FROM semantic_memory_links
WHERE source_is_index AND link_syntax = 'markdown'
ORDER BY project_name, ordinal;

-- Prose cross-references between topic files (not index entries)
SELECT source_name, target_resolved_name
FROM semantic_memory_links
WHERE link_syntax = 'wiki' AND is_resolved;

-- Referenced but never written -- memories Claude meant to record
SELECT DISTINCT owner_key, source_file_name, target_name
FROM semantic_memory_links WHERE NOT is_resolved;

-- Most-referenced memories (load-bearing knowledge)
SELECT target_resolved_name, COUNT(*) AS inbound
FROM semantic_memory_links
WHERE is_resolved GROUP BY ALL ORDER BY inbound DESC LIMIT 10;
```
