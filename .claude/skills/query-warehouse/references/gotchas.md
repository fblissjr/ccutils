# Query gotchas — read before joining raw facts

## Empty-by-design tables (DDL stubs)

`run_v15_etl` does NOT populate these; zero rows there means "not implemented",
never "no activity":

- `fact_content_blocks`, `fact_code_blocks`, `fact_entity_mentions`,
  `fact_tool_input_params`, `fact_facet_embeddings`
- `fact_turn_durations`, `fact_stop_events` — subsumed by `fact_system_events`
  (`subtype = 'turn_duration'` / stop-hook subtypes)
- `fact_tool_calls` — subsumed by `fact_tool_uses` + `fact_tool_results`
  (`semantic_tool_calls` view provides the legacy shape)

Conditionally populated: `fact_session_embeddings` has rows only when the
export ran with `--embed`; empty otherwise (that's "flag not used", not a stub).

## Unpopulated columns on populated tables

`fact_messages.estimated_tokens`, `.response_time_seconds`,
`.conversation_depth` are always NULL (DDL kept for compat). Use
`input_tokens`/`output_tokens` and `fact_system_events` turn durations instead.

## Soft deletes

Facts are never hard-deleted (`is_deleted`/`deleted_at`). Views that already
filter: `semantic_sessions`, `semantic_tool_calls`, `semantic_decisions`,
`semantic_token_usage`, `semantic_cost_analysis`. **Every raw-fact query needs
`WHERE is_deleted = FALSE`**; for other views, check the view SQL in
`src/ccutils/schemas/star/schema.py` before trusting counts.

## Join traps

- `fact_token_usage` has **no `message_id`** — join to `fact_messages` via
  `entry_id`, or `session_key` + `timestamp`.
- `fact_session_summary` has **no `unique_files_touched`** — count distinct
  files via `bridge_session_file` grouped by `session_key`.
- `fact_tool_results.is_error` is stored **tri-state** (TRUE/FALSE/NULL), but
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
  `SELECT rows_inserted FROM fact_etl_steps WHERE step_name = 'dim_memory'`.
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

## Token semantics

`input_tokens` (messages, token_usage) is the API-reported figure **after the
last cache breakpoint** — it is not the total context size. Cost comparisons
use `total_uncached_equivalent_tokens` = input + cache_creation(5m+1h) + read.
Cache-creation pricing differs: 5m tokens bill at 1.25x, 1h at 2x.

## Agent (subagent) sessions

- Identity is the FILE stem: `dim_session.session_id = 'agent-<id>'`. Raw JSON
  payloads (`$.sessionId` anywhere) carry the PARENT's id — never derive
  identity from them.
- `fact_agent_delegations.agent_session_key` is NULL when the agent's own
  transcript wasn't ETL'd — LEFT JOIN accordingly.
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
