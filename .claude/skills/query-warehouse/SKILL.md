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
uv run ccutils all --format duckdb -o ./analytics    # every session on disk
uv run ccutils --format duckdb -o ./analytics        # interactive picker
```

Open **read-only** so you never hold a write lock against a concurrent ETL run:

```bash
duckdb -readonly ./analytics/archive.duckdb -c "SELECT ..."
```

## 2. Route the question

Prefer `semantic_*` views — they pre-join dimensions and (mostly) pre-filter
soft-deleted rows. Drop to raw facts only when the view lacks a column.

| Question is about... | Start here |
|---|---|
| What sessions exist / catch up on a project | `semantic_sessions`, `semantic_project_context` |
| Tokens, cost, cache hit rate | `semantic_cost_analysis`, `semantic_token_usage`, `fact_session_summary` |
| Tool usage, failures, tool sequences | `semantic_tool_calls`, `fact_errors`, `semantic_tool_patterns`, `fact_tool_chain_steps` |
| Files read/edited, hot files across sessions | `semantic_file_operations`, `semantic_project_files`, `semantic_file_evolution`, `bridge_session_file` |
| Plans (ExitPlanMode), decisions timeline | `semantic_plan_revisions`, `semantic_decisions` |
| Subagents / Task delegations | `semantic_agent_delegations`, `dim_session` (`is_agent`, `parent_session_key`) |
| Resumed-session chains | `semantic_session_chains` |
| Prompt history | `semantic_prompt_history` |
| Facets (F01–F19 SQL, F20+ LLM) | `fact_session_facets` + `dim_facet_type` (EAV — see gotchas) |
| ETL observability / did my export work | `semantic_etl_runs`, `fact_etl_batch_runs` |
| Permission modes, compactions, API errors, turn durations | `fact_meta_events`, `fact_system_events` |

## 3. Rules that prevent wrong answers

- Raw fact tables use **soft delete**: always add `WHERE is_deleted = FALSE`.
- Several tables exist as **empty DDL stubs** — a zero-row result there is not
  "no data". Check the stub list in `references/gotchas.md` before concluding.
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
