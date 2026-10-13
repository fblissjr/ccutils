---
name: query-warehouse
description: Query a ccutils DuckDB star-schema warehouse to answer questions about Claude Code usage — sessions, token costs, tool calls, errors, files touched, plans, subagent delegations, facets, or ETL run health. Use when the user asks an analytical question about their Claude Code history, wants SQL against an archive.duckdb, or needs a warehouse generated first. Covers locating/building the database, routing questions to the right semantic view, and the join gotchas that produce silently-wrong answers.
---

# Querying the ccutils warehouse

## 1. Locate (or build) the database

```bash
find . -name '*.duckdb' -not -path '*/node_modules/*' 2>/dev/null
```

No database? Build one (incremental — safe to re-run against the same file):

```bash
uv run ccutils --source --format duckdb -o ./analytics    # every session on disk
uv run ccutils --format duckdb -o ./analytics        # interactive picker
```

Open **read-only** so you never hold a write lock against a concurrent ETL run:

```bash
duckdb -readonly ./analytics/archive.duckdb -c "SELECT ..."
```

## 2. Route the question

Prefer `semantic_*` views. A view exists because it encodes something a
query would get wrong (a filter, a derivation, a rollup, which rows are
replayed history), not to save a join: join the dimensions you need
yourself. Every view filters soft-deleted rows. Drop to raw facts only when
no view answers the question.

| Question is about... | Start here |
|---|---|
| What sessions exist / catch up on a project | `semantic_session_summary` joined to `dim_session` (classifications, `first_user_message` / `last_assistant_message`) and `dim_project`; the join is written out in `references/query-recipes.md` |
| Tokens, cost, cache reads | `semantic_session_summary` (per session), `semantic_token_usage` (per API response) |
| Tool usage, failures, tool sequences | `fact_tool_calls` (failures: `WHERE is_error`; sequences: `chain_id`, `prev_tool_key`, `next_tool_key`), `semantic_tool_patterns` |
| Files read/edited, hot files across sessions | `fact_file_operations` (+ `dim_file`), `semantic_project_files`, `semantic_file_evolution`, `semantic_session_files` |
| Plans (ExitPlanMode), decisions timeline | `fact_plan_revisions`, `semantic_decisions` |
| Subagents / Task delegations | `semantic_agent_delegations`, `dim_session` (`is_agent`, `parent_session_key`) |
| Resumed-session chains | `dim_session_chain` (join `dim_session` on `chain_key` for members) |
| Prompt history | `semantic_prompt_history` |
| What Claude has learned about a project (auto memory) | `semantic_memory` (current state), `dim_memory` (version history) |
| How memories reference each other / what the index catalogues | `semantic_memory_links` (filter `link_syntax`) |
| Facets (F01–F19 SQL, F20+ LLM) | `fact_session_facets` + `dim_facet_type` (EAV — see gotchas) |
| ETL observability / did my export work | `semantic_etl_runs`, `etl.batch_runs` |
| Permission modes, titles, attachments, queue operations, PR links | `fact_entry_events` (filter on `entry_type`) |
| Compactions, API errors, turn durations | `fact_system_events` |
| How a session worked: tool mix, shares by category, agentic runs, with percentile ranks | `semantic_session_behavior` (one row per session; descriptive, no archetype label) |
| How context grew request by request, where it was reset | `semantic_context_growth` (one row per API request; `is_context_reset`, `inbound_tokens`) |
| Which tool calls are a session's own and which are replayed history | `semantic_tool_call_replay` |

## 3. Rules that prevent wrong answers

- Raw fact tables use **soft delete**: always add `WHERE is_deleted = FALSE`.
- Before reading a zero-row result as "no data", ask the warehouse what was
  written: `SELECT * FROM etl.table_coverage` and
  `SELECT DISTINCT table_name FROM etl.steps` (`references/gotchas.md`).
- **A tool call can be several rows.** A resumed or forked session replays
  history into its own transcript under the same `tool_use_id`, and each
  copy is a row of `fact_tool_calls` and `fact_file_operations`. The
  `semantic_` views leave copies out. A query on the raw facts must join
  `semantic_tool_call_replay` and keep `NOT is_replay`, and must never join
  two facts on `tool_use_id` alone (`references/gotchas.md`, "Replayed tool
  calls").
- A typed result column on `fact_tool_calls` (`read_num_lines`,
  `glob_num_files`, the `agent_*` rollups, anything read from
  `result_payload_json`) is **NULL when the transcript stated no payload for
  that result**. A parent session always states one; an agent session often
  does not. NULL there means "not stated", never zero or false. Before
  summing or filtering on one, scope to parent sessions or report coverage;
  `references/gotchas.md` has the predicate and the coverage query.
- `input_tokens` is API-reported post-cache-breakpoint, **not** total context;
  for cost ranking use `total_uncached_equivalent_tokens`.
- Agent session identity is the `session_id` column (`agent-<id>`), never
  `$.sessionId` inside any raw JSON payload (it holds the parent's id).

## 4. Go deeper (read only what the task needs)

- `references/query-recipes.md` — worked SQL per question family (copy, adapt).
- `references/gotchas.md` — join traps, unpopulated columns/tables, EAV facets,
  DuckDB JSON idioms. Read before writing joins against raw facts.
- `docs/STAR_SCHEMA.md` — full column reference for every table and view.
  Grep for the table name; don't read the whole file.
- Interactive exploration for the user: `duckdb -ui <db>` (DuckDB's own
  local UI; first run fetches the `ui` extension, needs network). There is
  no `ccutils explore` command -- it was removed along with the harlequin
  dependency.
