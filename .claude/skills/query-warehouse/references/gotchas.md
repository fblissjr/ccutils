# Query gotchas — read before joining raw facts

## Which tables are populated

Ask the warehouse, not this file: `SELECT * FROM etl.table_coverage` lists
every table and view with its status, writer and reason, and
`SELECT DISTINCT table_name FROM etl.steps` says what this file was actually
written by. There are no empty-by-design tables since 1.0.0, and no
conditional ones either (`--embed` and its table were removed).
Turn durations and stop events live in `fact_system_events` by `subtype`.

## Soft deletes

Facts are never hard-deleted (`is_deleted`/`deleted_at`). Views that already
filter: `semantic_sessions`, `semantic_decisions`,
`semantic_token_usage`, `semantic_cost_analysis`. **Every raw-fact query needs
`WHERE is_deleted = FALSE`**; for other views, check the view SQL in
`src/ccutils/schemas/star/schema.py` before trusting counts.

## Join traps

- `fact_token_usage` has **no `message_id`** — join to `fact_messages` via
  `entry_id`, or `session_key` + `timestamp`.
- `semantic_session_summary` has **no `unique_files_touched`** — count distinct
  files via `semantic_session_files` grouped by `session_key`.
- `fact_tool_calls.is_error` is stored **tri-state** (TRUE/FALSE/NULL), but
  NULL means **not an error**, not "unknown". Claude Code writes
  `is_error: true` on failure and encodes success either as `false` or by
  omitting the field; the API defines an absent `is_error` as false. Measured
  across 71,635 results: 2,331 TRUE / 31,269 FALSE / 38,035 NULL, and the
  per-tool split is bimodal — Bash writes the field on every result, most
  other tools write it only on failure. So **`is_error IS NOT TRUE` is the
  correct "succeeded" test**, and `is_error = FALSE` silently drops the 38,035
  succeeded-by-omission rows. Count errors with
  `SUM(CASE WHEN is_error THEN 1 ELSE 0 END)`, which routes NULL correctly.
  (Do not rely on "only Bash writes FALSE" — a `Workflow` result writes it too,
  which is harmless since FALSE and absent are equivalent.)
- **`semantic_etl_runs.rows_inserted` is 0 for the auto-memory run, and that is
  not a failure.** The rollup sums only `step_kind = 'upsert'` steps, which are
  fact populators; memory is a dimension, so counting it there would inflate
  every fact total. The real count lives one grain down --
  `SELECT rows_inserted FROM etl.steps WHERE step_name = 'dim_memory'`.
  Same applies to any future non-fact step. Filter `run_kind = 'global_source'`
  to find these runs.
- `dim_memory` is a **Type 2 SCD** — one row per (memory file, content
  version), not one per file. **Any query that is not about history must
  filter `WHERE is_current`**, or a memory edited five times counts five
  times. `semantic_memory` applies that filter for you; `dim_memory` does not.
  `memory_id` is the file's stable identity across versions, `memory_key` one
  version of it — group by `memory_id` to count memories, `memory_key` to
  count versions.
- `bridge_memory_link` mixes two link kinds and **`link_syntax` is not
  cosmetic**: `markdown` rows are `MEMORY.md` index entries (the index
  catalogues a topic file), `wiki` rows are prose cross-references between
  topic files. Counting both as one edge type conflates "is catalogued here"
  with "argues against that". Unresolved rows are kept deliberately
  (`is_resolved = FALSE`, NULL target) — a link to a memory that was never
  written is real signal, so filter it out explicitly rather than assuming
  every row resolves.
- Most facts carry `session_id` as a degenerate dimension — filter on it
  directly instead of joining `dim_session` when you only need scoping.
- Surrogate keys are `md5(natural key)` — `session_key = md5(session_id)`
  lets you skip a dim join.
- **`tool_use_id` is not unique.** Join `fact_file_operations`,
  `fact_plan_revisions` or anything else to `fact_tool_calls` on
  `tool_call_key`, or on `tool_use_id` AND `session_id`. On the id alone a
  replayed call matches every transcript that holds a copy.

## Replayed tool calls

A resumed, backgrounded or forked session replays earlier history into its
own file under the same `tool_use` ids, so one call can be several rows of
`fact_tool_calls` and `fact_file_operations`, one per transcript holding it.
Only one of them is the session that made the call.

- `semantic_tool_call_replay` has one row per live tool call with
  `is_replay`. Every `semantic_` view that counts tool calls, file
  operations or delegations already leaves replays out. **A query against
  the raw facts must do it itself**, or it counts a call once per copy and
  credits a fork with its spawner's work:

```sql
SELECT tc.session_id, COUNT(*) AS own_calls
FROM fact_tool_calls tc
JOIN semantic_tool_call_replay rp USING (tool_call_key)
WHERE tc.is_deleted = FALSE AND NOT rp.is_replay
GROUP BY ALL ORDER BY own_calls DESC LIMIT 20;
```

- `replay_source` says how sure the mark is. `fork_prefix` is stated by the
  fork's sidecar. `later_copy` is a rule (among the holders of a call, the
  session that ended first is taken as the original), and it can pick the
  wrong one of two sessions that share a prefix and both continued.
- **A fork's facets, heuristics and plan revisions describe its own work.**
  `dim_session.fork_cut_timestamp` is where a fork's own history starts
  (NULL for anything that is not a fork), and `fact_session_facets`, the
  `dim_session` heuristics (`first_user_message`, `intent`, ...) and
  `fact_plan_revisions` are computed from what came after it. So are the
  `derived_*` figures a delegation takes from a fork child.
- **A fork's message and token totals are its own too.** The message and
  token columns of `semantic_session_summary` (and so of
  `semantic_cost_analysis`, `semantic_sessions` and
  `semantic_session_behavior`) count what came after the cut, and agree
  with the fork's `F15` / `F30`. The per-response views,
  `semantic_token_usage` and `semantic_context_growth`, keep a fork's
  inherited rows and mark them `in_fork_prefix`: add `NOT in_fork_prefix`
  to total a session's own usage from them. A query on raw
  `fact_token_usage` or `fact_messages` must apply the cut itself
  (`timestamp > fork_cut_timestamp` where the session has one).
- The same holds for a fork's events and timeline: the summary's turn,
  compaction, API-error, progress and attachment counts, its
  `permission_mode_transition_count` and `session_duration_seconds`, and
  `semantic_decisions` all start at the cut. A mode change the fork made is
  judged against the mode it inherited. A query on raw `fact_system_events`
  or `fact_entry_events` must apply the cut itself; for entry events use
  `COALESCE(timestamp, derived_timestamp)`, since several entry types state
  no time.
- A copy between root sessions has no cut: its facets, heuristics and plan
  revisions still include the replayed part, and cross-session token sums
  still double-count continuations.
- The measurements are `docs/JSONL_CONTRACT.md` claim 12.

## Typed result columns and missing payloads

`fact_tool_calls.result_payload_json` is the structured payload Claude Code
writes beside a tool result (`toolUseResult`), and the typed columns are read
from it: `read_num_lines`, `read_total_lines`, `glob_num_files`,
`grep_num_files`, `bash_stdout_bytes`, the `edit_*` columns, `write_type`,
and the `agent_*` rollups. **Where the transcript stated no payload, all of
them are NULL and nothing falls back.** A parent session always states one.
An agent session often does not, depending on the Claude Code version and on
how the agent was spawned. The measurement, the version stretches and the
canary are `docs/JSONL_CONTRACT.md` claim 11.

- NULL in a typed column means "not stated", never zero or false. A `SUM` or
  a `WHERE` on one silently leaves those calls out, and `ccutils audit` does
  not flag it, because parent rows fill the column.
- The rows are `result_entry_id IS NOT NULL AND result_payload_json IS NULL`:
  a result came back and its entry stated no payload.
- What comes from the result block itself is on every result: `is_error`,
  `result_content_text`, and `derived_exit_code` / `derived_failure_kind`,
  which are parsed from them. Prefer those for agent work.
- A failed call always has a payload (the error string), so error counts are
  not affected.

Report coverage beside any figure built on a typed column:

```sql
-- Share of tool results that carry a payload, parent against agent sessions.
SELECT ds.is_agent, ftc.tool_name,
       COUNT(*) AS results,
       COUNT(ftc.result_payload_json) AS with_payload
FROM fact_tool_calls ftc
JOIN dim_session ds ON ds.session_id = ftc.session_id
WHERE ftc.is_deleted = FALSE AND ftc.result_entry_id IS NOT NULL
GROUP BY ALL ORDER BY results DESC;
```

## Token semantics

`input_tokens` (messages, token_usage) is the API-reported figure **after the
last cache breakpoint** — it is not the total context size. Cost comparisons
use `total_uncached_equivalent_tokens` = input + cache_creation(5m+1h) + read.
Cache-creation pricing differs: 5m tokens bill at 1.25x, 1h at 2x.

## Agent (subagent) sessions

- Identity is the FILE stem: `dim_session.session_id = 'agent-<id>'`. Raw JSON
  payloads (`$.sessionId` anywhere) carry the PARENT's id — never derive
  identity from them.
- Delegations are the view `semantic_agent_delegations` (one row per
  Agent/Task call in `fact_tool_calls`); there is no delegations table. Its
  `agent_*` columns are stated by the parent and its `derived_*` columns come
  from the agent's transcript; `docs/STAR_SCHEMA.md` has the column table.
- **Count delegations from the view, not from `Agent` rows of
  `fact_tool_calls`.** A fork's transcript replays its spawner's history,
  including the call that created the fork, and each copy is a row of the
  fact (see "Replayed tool calls"). The view reads only calls a session made
  itself, so a fork does not appear to have spawned itself.
  `semantic_session_summary.total_delegations` follows the same rule.
- A background launch made inside an agent states no `agent_id` (see "Typed
  result columns and missing payloads"). The view then finds the child from
  the spawned agent's sidecar, and `agent_link_source` says which side named
  it. Use `agent_session_key` from the view.
- `agent_id` stays what the parent STATED, so it is NULL on a delegation
  whose child was found through the sidecar (`agent_link_source =
  'child_sidecar'`). Test for a linked agent with `agent_session_key IS NOT
  NULL`, not `agent_id IS NOT NULL`.
- `completion_state IS NULL` has two causes, and `agent_link_source` tells
  them apart: NULL means no side links an agent; a value means the agent is
  named and its transcript is not in the warehouse.
- `agent_is_async` is three-valued: TRUE is a background launch, FALSE a
  stated synchronous result, NULL a result that carried no payload and so
  states neither. Do not write `NOT agent_is_async` to mean synchronous: it
  drops the NULL rows silently. `agent_output_text` is NULL unless the row
  is stated synchronous or is a refused spawn.
- Pre-2026 short agent ids can collide across parents (accepted limitation).
- An agent whose parent was pruned keeps `depth_level = 0`.
- Exclude sidechains from "human conversation" metrics:
  `fact_messages.is_sidechain = FALSE`, or `dim_session.is_agent = FALSE`.

## Facet EAV shape

`fact_session_facets` populates exactly one of `value_text` / `value_json` /
`value_numeric` / `value_bool` per row, routed by `dim_facet_type.output_type`
(`OUTPUT_TYPE_TO_COL` in `src/ccutils/etl/facets/extractor.py`). Tier 2 rows can
be fallbacks — filter `is_fallback = FALSE` for real extractions. `prompt_version`
is NULL for Tier 1; the same facet_id can have rows under multiple prompt
versions, so pin `prompt_version` when comparing across time.

## DuckDB JSON idioms (project conventions)

- Lists: `json_extract(j, '$.p[*].f')::JSON[]`; block + index: `LATERAL unnest(...)`.
- `json_type()` to gate string-vs-array `$.content`.
- Raw JSON out: `CAST(json_extract(...) AS VARCHAR)` — `json_extract_string`
  mangles arrays.
- `input_json` / `*_json` columns are JSON-encoded VARCHAR, not JSON type —
  wrap in `json_extract*` to traverse.

## Operational

- Open read-only (`duckdb -readonly`) — a held write lock blocks concurrent ETL.
- The warehouse is incremental; `semantic_etl_runs` tells you what the last run
  actually ingested if numbers look stale.
- Full column lists: grep the table name in `docs/STAR_SCHEMA.md`.
