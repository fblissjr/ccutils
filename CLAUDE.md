<!-- path-privacy: skip-file -- references universal ~/.claude data paths (not personal) -->
# Start Here

Uses uv. Run tests like this:

    uv run pytest

Run the development version of the tool like this:

    uv run ccutils --help

Always practice TDD: write a failing test, watch it fail, then make it pass.

Commit early and often. Commits should bundle the test, implementation, and documentation changes together.

## v0.15 ETL pipeline

- **Entry point for new code:** `run_v15_etl(conn, session_path, *, project_name, parquet_lake_root)` in `src/ccutils/etl/orchestrator.py`. The legacy `run_star_schema_etl` + `finalize_star_schema` pair was deleted in the cull.
- **Four tiers:** JSONL (Claude Code writes) → Parquet lake (Tier 1, `src/ccutils/parsers/parquet_writer.py`) → `stg_log_entries` staging (Tier 2) → fact tables (Tier 3).
- **Every v0.15 fact** follows the lineage convention via `lineage_upsert(conn, *, run, table, inbound_table, natural_key, payload_cols, hash_cols)` in `src/ccutils/etl/upsert.py`. Lineage block on every row: `created_at`, `last_updated_at`, `created_by_version_key`, `last_updated_by_version_key`, `etl_run_id`, `record_source`, `hash_diff`, `is_deleted`, `deleted_at`.

## Facet pipeline (v0.15+)

Three-tier facet system on top of v0.15 facts. Writes one row per (session × facet) into `fact_session_facets`.

- **Tier 1 (F01-F19, SQL-computed):** `populate_tier1_facets` in `src/ccutils/etl/fact_session_facets.py`. Zero inference, always runs.
- **Tier 2 (F20+, LLM-extracted via Haiku 4.5):** `populate_tier2_facets` in `src/ccutils/etl/facets/populator.py`. Opt-in via `--with-llm-facets` (local) or `--batch-llm-facets` (all). Requires `ANTHROPIC_API_KEY` env var or `ccutils-anthropic` keychain entry.
- **Tier 3 (clustering):** not yet built (Step 5+).

**Catalog as single source of truth:** `src/ccutils/etl/facets/catalog.py::FACET_SPECS` lists every Tier 2 facet + its `prompt_version`. `create_star_schema()` seeds one `dim_facet_type` row per spec via `INSERT ... ON CONFLICT DO NOTHING`. Bumping `prompt_version` on a `FacetSpec` adds a new row (old rows survive, fact rows referencing them remain valid).

**Import direction (one-way):** `schemas/star/schema.py` imports `FACET_SPECS` from `ccutils.etl.facets.catalog` at DDL seed time. The `facets/` package MUST NOT import from `schemas/` — would loop at CLI startup.

**Shared dim DDL:** `dim_facet_type` uses `CREATE TABLE IF NOT EXISTS` + `INSERT ... ON CONFLICT DO NOTHING` (not `CREATE OR REPLACE`). `create_star_schema()` runs on every CLI invocation; the historical-state strategy preserves prompt_version rows that fact_session_facets references. Apply the same pattern to any future dim that retains historical state.

**Credentials boundary:** `src/ccutils/cli/utils.py::build_facet_extractor_or_exit` is the single seam where Tier 2 credentials are resolved + an `AnthropicFacetExtractor` is constructed. Both CLI commands import it; future Tier 2 callers should too.

**Design doc:** `internal/plans/facet_extractor_protocol.md` (gitignored).

## lineage_upsert pitfalls

- `payload_cols` MUST NOT include `session_key` — it's added by `extra_keys`; duplicates cause "Multiple assignments to same column".
- Target fact table MUST have `date_key` + `time_key` columns; lineage_upsert always references them on UPDATE.
- Aggregate facts without a plain `timestamp` column: pass `timestamp_col="first_operation_timestamp"` (or similar) so date/time keys derive from the right field.
- **Shared fact tables (multiple populators writing the same table) MUST pass `soft_delete_scope_sql`** to `lineage_upsert` so each populator's soft-delete only touches rows it owns. Without it, Populator A's pass soft-deletes every row Populator B just wrote. See `fact_session_facets`: both Tier 1 and Tier 2 populators pass `facet_tier_scope_sql(N)` from `etl/facets/catalog.py`.

## Populator scoping

- Populators that read from STAGING (`stg_log_entries`) are bounded naturally — only the current session is loaded.
- Populators that read from PERMANENT FACTS (`fact_tool_uses`, `fact_attachments`, etc.) MUST add `AND session_id IN (SELECT DISTINCT session_id FROM stg_log_entries WHERE session_id IS NOT NULL)` to their inbound CTE — otherwise they rescan the whole warehouse on every per-session ETL call.

## DDL widening checklist

- After changing fact column names: grep `semantic_` views in `schemas/star/schema.py` for the old name and update.
- After renaming `tool_call_id` → `tool_use_id` (or similar): grep the column-list assertions in `tests/test_star_schema_ddl.py` too.

## Subagent JSONL layout

- Subagent sessions live at `.../projects/<project>/<parent-session-uuid>/subagents/agent-<id>.jsonl` with optional sibling `agent-<id>.meta.json` carrying `agentType` + `description`.
- `dim_session.agent_id` is the `<id>` suffix; `parent_session_key = md5(parent-session-uuid)`; cross-session linkage on `fact_agent_delegations.agent_session_key` resolves via `dim_session.agent_id` when both sessions are ETL'd.

## DuckDB JSON extraction idioms

- `json_extract(j, '$.path[*].field')::JSON[]` returns a list — clean for `list_contains` / `list_transform` checks.
- `LATERAL (SELECT unnest(json_extract(j, '$.path')::JSON[]) AS block)` for when you need the full block AND its index via `generate_subscripts`.
- `json_type(j, '$.path')` returns `'VARCHAR'`/`'ARRAY'`/`'NULL'` — use it to gate when `$.content` can be either a string or a list.
- `json_extract_string` unwraps strings cleanly but stringifies arrays/objects into bracket-text — use `CAST(json_extract(j, '$.path') AS VARCHAR)` to get raw JSON for Python `json.loads()`.

## Pydantic alias gotcha

`pydantic.alias_generators.to_camel` does NOT preserve all-caps abbreviations (ID, UUID, URL). Fields like `sourceToolUseID`, `parentToolUseID`, `sourceToolAssistantUUID` need explicit `Field(alias="...")`. Grep `src/ccutils/parsers/models.py` for `alias=` to see current overrides.

## JSON library

Stdlib `json` is the project convention (13 files use it). Do NOT auto-migrate to orjson — global rule was removed earlier; project-wide migration is out of scope.

## CLI / test patterns

- **CLI test monkeypatching:** `cli.add_command(local_cmd, "local")` makes `ccutils.cli.local` resolve to the click subcommand, shadowing the Python submodule for `getattr`-based attribute walks (including pytest's dotted-string monkeypatch). Use `importlib.import_module("ccutils.cli.local")` to get the actual module. See `tests/test_cli_llm_facets.py` for the pattern.
- **CLI stderr in tests:** `click.echo(err=True)` writes to stderr. Click's `CliRunner` may mix or separate stdout/stderr depending on version. Use `_combined_output(result)` (in `tests/test_cli_llm_facets.py`) + assert on `exit_code` for version-robust assertions.
- **F90+ test fixture convention:** test fixtures seeding hypothetical `dim_facet_type` rows MUST use facet ids in F90+ to avoid colliding with the real catalog as it grows. Canonical: `tests/test_fact_session_facets_v15.py::two_f90_versions`.
- **Live-API smoke (run once after `AnthropicFacetExtractor` changes; pennies):**
  ```bash
  ANTHROPIC_API_KEY=$(security find-generic-password -s ccutils-anthropic -a $USER -w) \
    uv run pytest tests/test_populate_tier2_facets.py::TestLiveApiSmoke -v
  ```

## Project Structure

```
ccutils/
├── src/ccutils/
│   ├── __init__.py           # Public API re-exports
│   ├── sanitize.py           # Path sanitization for --private mode
│   ├── cli/                   # CLI commands
│   │   ├── __init__.py       # CLI group and entry point
│   │   ├── local.py          # local command (default) -- picker + single-file conversion
│   │   ├── web.py            # web command
│   │   ├── all.py            # all command
│   │   ├── explore.py        # explore command (harlequin shim)
│   │   ├── import_cmd.py     # import command (Claude.ai exports)
│   │   ├── schema.py         # schema command (JSON structure inspector)
│   │   └── utils.py          # CLI utilities
│   ├── api/                   # API client and credentials
│   │   └── __init__.py
│   ├── parsers/              # Session file parsing utilities
│   │   ├── __init__.py       # Public API exports
│   │   ├── jsonl_reader.py   # Canonical JSONL parser (iter_session_entries, iter_all_session_entries)
│   │   ├── history.py        # ~/.claude/history.jsonl parser (HistoryEntry, iter_history_entries)
│   │   ├── session.py        # JSONL/JSON session parsing
│   │   ├── discovery.py      # Session discovery + two-phase selection UI
│   │   ├── metadata.py       # SessionMetadata dataclass + rich extraction
│   │   ├── claude_ai.py      # Claude.ai export parser
│   │   └── schema_inspector.py # JSON structure analysis
│   ├── schemas/              # Schema definitions
│   │   ├── __init__.py       # Unified exports for both schemas
│   │   ├── simple/           # Simple 4-table schema
│   │   │   ├── __init__.py
│   │   │   ├── schema.py     # DDL for simple schema
│   │   │   └── etl.py        # Simple schema ETL
│   │   └── star/             # Star schema (DDL + legacy ETL)
│   │       ├── __init__.py   # Public API exports
│   │       ├── schema.py     # DDL for star schema tables + semantic views (v0.15 reshape)
│   │       ├── etl.py        # LEGACY per-session ETL (dead on etl-rethink; pending Phase D)
│   │       ├── migrations/   # DDL migration runner + baseline migration
│   │       ├── extractors.py # Code blocks, entities, file extraction (legacy)
│   │       ├── heuristics.py # Keyword/metric-based classification (legacy)
│   │       ├── history_etl.py# History.jsonl -> dim_prompt ETL (legacy)
│   │       ├── json_export.py# JSON export for star schema
│   │       ├── embeddings.py # Optional ColBERT embedding pipeline
│   │       └── utils.py      # Key generation, tool/model classification, dim_date helper
│   ├── etl/                  # v0.15 four-tier ETL (NEW; supersedes legacy schemas/star/etl.py)
│   │   ├── orchestrator.py   # run_v15_etl() entry point
│   │   ├── lineage.py        # EtlRun + hash_diff + record_source_label
│   │   ├── upsert.py         # lineage_upsert() shared by every fact populator
│   │   ├── staging.py        # load_session_to_staging() (Tier 1 -> Tier 2)
│   │   ├── utils.py          # Shared ETL utilities (extract_text_from_content_json)
│   │   ├── fact_messages.py
│   │   ├── fact_tool_calls.py        # fact_tool_uses + fact_tool_results
│   │   ├── fact_token_usage.py
│   │   ├── fact_session_summary.py
│   │   ├── entry_type_facts.py       # attachments, progress, system, meta, file_history, queue_ops, pr_links
│   │   ├── fact_session_facets.py    # Tier 1 facet populator (F01-F19, SQL-computed)
│   │   └── facets/                    # Tier 2 LLM-extracted facets
│   │       ├── catalog.py             # FACET_SPECS registry + facet_tier_scope_sql helper
│   │       ├── extractor.py           # FacetExtractor Protocol + dataclasses + CannedFacetExtractor
│   │       ├── anthropic.py           # AnthropicFacetExtractor (Haiku via httpx)
│   │       └── populator.py           # populate_tier2_facets
│   ├── export/                # Export format handlers
│   │   ├── __init__.py
│   │   ├── html.py           # HTML generation
│   │   └── duckdb_archive.py # DuckDB batch export (rewired to run_v15_etl on etl-rethink)
│   ├── tui/                   # Terminal UI components
│   │   ├── __init__.py
│   │   ├── theme.py          # Color theme
│   │   ├── formatters.py     # Label formatters
│   │   ├── layout.py         # Table layout
│   │   ├── components.py     # Reusable UI components
│   │   └── selection.py      # Interactive selection
│   └── templates/            # Jinja2 templates for HTML export
│       ├── base.html
│       ├── macros.html       # Shared rendering macros (tools, messages, etc.)
│       ├── page.html
│       ├── master_index.html # Archive-level index
│       ├── project_index.html
│       ├── index.html        # Per-session index
│       ├── search.js         # Per-session search (Jinja2 template)
│       └── global_search.js  # Archive-wide search (Jinja2 template)
├── tests/
│   ├── conftest.py                   # Shared fixtures (sample_session_file, interrupted_session_file, etc.)
│   └── test_*.py                     # 24 test files (star schema split across ddl/etl/analytics/advanced)
├── docs/
│   └── STAR_SCHEMA.md        # Star schema documentation
└── README.md
```

## Key Components

### 1. CLI Commands
- `local` - **default command**: pass a file to convert it (`ccutils session.jsonl`), or no args for interactive two-phase picker. `--flat` for legacy single-list mode
- `web` - Import from Claude API (auto-detects credentials from macOS keychain). HTML-only (no `--format`)
- `all` - Batch convert all sessions (supports parallel processing with `-j`)
- `explore` - Open DuckDB database in harlequin (requires `ccutils[explore]`)
- `import` - Import Claude.ai account exports (Settings > Privacy > Export)
- `schema` - Inspect JSON structure without exposing content (safe to share publicly)
- `convert` - Hidden alias for `local` (backwards compatibility)

### 2. Export Formats
Three output formats with two schema types:

**Simple schema** (4 tables: `sessions`, `messages`, `tool_calls`, `thinking`):
- `--format duckdb` - DuckDB database file
- `--format json` - Single JSON file with nested tables

**Star schema** (v0.15, on etl-rethink branch):
- `--format duckdb-star` - DuckDB database file
- `--format json-star` - Directory with meta.json + dimensions/*.json + facts/*.json
- Pipeline entry: `run_v15_etl()` in `src/ccutils/etl/orchestrator.py` (see the v0.15 ETL pipeline section above).
- DDL: `create_star_schema()` in `src/ccutils/schemas/star/schema.py`.
- `--embed [MODEL]` flag available on both `local` and `all` commands (requires pylate optional dependency)
- `docs/STAR_SCHEMA.md` describes the legacy schema; rewrite for v0.15 pending.

**Schema inference**: Schema type is auto-inferred from `--format` -- `duckdb-star` and `json-star` use star schema, plain `duckdb` and `json` use simple schema.

**Defaults**: Thinking blocks and subagents/agents are included by default. Use `--no-thinking`, `--no-subagents` (local), or `--no-agents` (all) to exclude them.

### 3. Star Schema Tables (v0.15 on etl-rethink)

**Lineage / Meta (3):** dim_etl_version, fact_etl_runs, meta_schema_version

**Core Dimensions:** dim_session, dim_project, dim_tool, dim_model, dim_date, dim_time, dim_prompt, dim_file. Populated as stub rows by the v0.15 orchestrator; heuristic enrichment (intent, complexity, outcome, domain) was a legacy concern and will be rewritten in a separate pass.

**Staging (2):** stg_log_entries (one row per JSONL line; Tier 2 of the four-tier pipeline), stg_task_agent_map (legacy)

**v0.15 Facts (populated by `run_v15_etl`):**
- `fact_messages` (one row per user/assistant entry; includes `stop_reason`, `permission_mode_at_send`, `prompt_id`, `request_id`, `is_api_error_message`)
- `fact_tool_uses` (one row per tool_use block)
- `fact_tool_results` (one row per tool_use_id; combines tool_result content with the entry-level `toolUseResult` structured payload; per-tool typed columns: Bash `exit_code`/`interrupted`, Edit `structured_patch`, Read `num_lines`/`total_lines`, Agent rollup, etc.)
- `fact_token_usage` (per-API-response; R11 cache split into `cache_creation_5m_tokens` / `cache_creation_1h_tokens` + `total_uncached_equivalent_tokens`)
- `fact_session_summary` (one row per session, aggregates over all the above)
- `fact_attachments` (all 23 attachment subtypes)
- `fact_progress_events` (all 6 progress data variants — hook_progress, bash_progress, agent_progress, etc.)
- `fact_system_events` (all 7 system subtypes — turn_duration, stop_hook_summary, api_error, compact_boundary, local_command, away_summary, bridge_status)
- `fact_meta_events` (time-series for permission-mode, custom-title, agent-name, last-prompt)
- `fact_file_history_snapshots`, `fact_queue_operations`, `fact_pr_links`

**Legacy facts in DDL but not populated by v0.15** (pending Phase D): fact_file_operations, fact_errors, fact_content_blocks, fact_code_blocks, fact_entity_mentions, fact_tool_chain_steps, fact_diagnostics, fact_turn_durations, fact_stop_events, fact_agent_delegations, fact_plan_revisions, dim_session_chain, bridge_session_file, fact_session_embeddings, fact_tool_input_params, fact_tool_calls. Some are redundant with v0.15 (fact_turn_durations / fact_stop_events live in fact_system_events now); others still need ports.

**Semantic views:** semantic_sessions, semantic_messages, semantic_tool_calls (legacy-compat UNION over fact_tool_uses+fact_tool_results), semantic_token_usage, semantic_cost_analysis (R11-corrected hit-rate denominator), semantic_prompt_history, semantic_session_chains, semantic_project_context, plus other legacy views that still reference dropped columns — those need rewrites pending Phase D.

### 4. Token Tracking (v0.15)

**Actual tokens** (from API usage data on assistant messages):
- `fact_token_usage`: per-API-response breakdown. R11 fix: cache_creation split per pricing tier (`cache_creation_5m_tokens`, `cache_creation_1h_tokens`, `cache_creation_total_tokens`), plus `total_uncached_equivalent_tokens` = input + creation_total + read.
- `fact_messages`: `input_tokens` (renamed from `actual_input_tokens`; honestly named per Anthropic semantics — post-last-cache-breakpoint, NOT total uncached), `output_tokens`, `cache_creation_5m_tokens`, `cache_creation_1h_tokens`, `cache_read_tokens`, `total_uncached_equivalent_tokens`.
- `fact_session_summary`: aggregated `total_*` versions of all the above.
- `semantic_cost_analysis` view: `cache_hit_rate_pct` denominator now includes cache_creation (legacy view excluded it, over-stating hit rate).

**Estimated tokens** still available via `estimate_tokens()` in `schemas/star/extractors.py` (text x1.3, code x1.5) for sessions predating API-side usage data; not surfaced by default in v0.15.

### 5. Simple Schema ETL Architecture

`schemas/simple/etl.py` uses a shared extraction core:
- `_extract_session_core()` → returns `SimpleExtractionResult` dataclass with all parsed data
- `export_session_to_duckdb()` → thin wrapper that INSERTs the result into DuckDB
- `_extract_session_data()` → thin wrapper that converts the result to dicts for JSON export

### 6. Heuristic Classification

Runs during ETL with zero external dependencies:
```python
from ccutils import classify_intent, classify_complexity, classify_outcome, classify_domain, classify_error_type
```

## Testing

Run all tests:

    uv run pytest

Run star schema tests specifically:

    uv run pytest tests/test_star_schema_ddl.py tests/test_star_schema_etl.py tests/test_star_schema_analytics.py tests/test_star_schema_advanced.py -v

Run with coverage:

    uv run pytest --cov=ccutils

## Common Workflows

### Adding a new dimension
1. Write failing tests in `test_star_schema_ddl.py` (table exists, columns correct) and `test_star_schema_etl.py` (rows populated)
2. Add CREATE TABLE in `schemas/star/schema.py`
3. Add ETL logic in `schemas/star/etl.py`
4. Run tests green, then update docs/STAR_SCHEMA.md

### Adding a new v0.15 fact table
1. Write failing tests (DDL existence + lineage columns + populator behavior + idempotency) in `tests/test_<fact>_v15.py`.
2. Add CREATE TABLE in `schemas/star/schema.py` with the standard lineage block (created_at, last_updated_at, version keys, etl_run_id, record_source, hash_diff, is_deleted, deleted_at + degenerate dims).
3. Add `populate_fact_<name>(conn, *, run)` in `src/ccutils/etl/<fact>.py` -- build inbound temp table from staging, delegate to `lineage_upsert()`.
4. Wire into `run_v15_etl()` in `src/ccutils/etl/orchestrator.py` in dependency order (anything fact_session_summary aggregates over must run before it).
5. Tests green; CHANGELOG entry; (eventually) update docs/STAR_SCHEMA.md when the v0.15 rewrite of that doc happens.

### Removing a feature
1. Grep for all imports, call sites, `__all__` exports, CLI registrations, and test references
2. Delete source files, remove from `cli/__init__.py`, `schemas/*/__init__.py`, `__init__.py`
3. Remove tests, update CHANGELOG, CLAUDE.md project tree, README
4. Check CLI help text and docstrings for stale table/view counts or feature references

### v0.15 ETL pipeline (per session)
```
create_star_schema(conn)                                   # DDL
run_v15_etl(conn, session_path,                           # Four-tier per-session ETL
            project_name="...",
            parquet_lake_root="/path/to/parquet_lake")
# Internally orchestrates: write Parquet -> load staging -> stub dims ->
# populate every v0.15 fact in dependency order (fact_session_summary last).
# Optional: EmbeddingPipeline(conn).embed_sessions(conn)
```

## Versioning

- Version lives in `pyproject.toml` -- keep it in sync with CHANGELOG.md
- Tag releases: `git tag v0.X.0 <commit> -m "v0.X.0: summary"`
- Use `/release` skill to bump version, verify CHANGELOG, and tag in one step
- Current on main: v0.14.0. The etl-rethink branch is the in-progress v0.15.0.

## Automations (.claude/)

`.claude/` is gitignored -- agents, skills, hooks are local-only.

**Agents:** `schema-reviewer` (star schema consistency), `doc-drift-checker` (stale counts in docs/CLI/docstrings)
**Skills:** `new-dimension`, `new-fact`, `test-schema`, `release` (version bump + tag)
**Hooks:** PreToolUse blocks Edit/Write to `tests/__snapshots__/` (use `--snapshot-update`)

Run doc-drift-checker after any schema change or feature removal.
Use `/release` to cut versions instead of manual pyproject.toml edits.

## HTML Export Gotchas

- CSS classes used in `templates/macros.html` MUST be defined in `static/transcript.css` -- Jinja2 won't warn
- `--snapshot-update` needed after ANY change to `transcript.css`, `macros.html`, or `base.html` (CSS is inlined in every page)
- `global_search.js` and `search.js` are Jinja2 templates (not static files) -- rendered via `_jinja_env.get_template()`
- Template variables render empty (not error) if not passed -- always test rendered HTML for expected content
- Never use real usernames/paths in docstrings or test fixtures -- use `/Users/dev/workspace/project`

### HTML Security

- `render_markdown_text()` sanitizes output via `nh3.clean()` to strip dangerous HTML (`<script>`, event handlers, `<iframe>`, etc.)
- `nh3.clean(raw, attributes={"code": {"class"}})` preserves `class` on `<code>` elements for fenced code block syntax highlighting
- Jinja2 environment uses `autoescape=True` -- the `|safe` filter in macros is safe because all content is either pre-sanitized (nh3) or pre-escaped (`html.escape()`)
- `base.html` includes a Content-Security-Policy meta tag that blocks external scripts, iframes, and restricts fetch to same-origin
- Do NOT remove the CSP or nh3 sanitization without understanding the XSS implications
