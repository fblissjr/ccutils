<!-- path-privacy: skip-file -- references universal ~/.claude data paths (not personal) -->
# Start Here

Uses uv. Run tests like this:

    uv run pytest tests/ --confcutdir=tests

Run the development version of the tool like this:

    uv run ccutils --help

Always practice TDD: write a failing test, watch it fail, then make it pass.

Commit early and often. Commits should bundle the test, implementation, and documentation changes together.

## v0.15 ETL pipeline

- **Entry point for new code:** `run_v15_etl(conn, session_path, *, project_name, parquet_lake_root)` in `src/ccutils/etl/orchestrator.py`. The legacy `run_star_schema_etl` + `finalize_star_schema` pair was deleted in the cull.
- **Four tiers:** JSONL (Claude Code writes) → Parquet lake (Tier 1, `src/ccutils/parsers/parquet_writer.py`) → `stg_log_entries` staging (Tier 2) → fact tables (Tier 3).
- **Every v0.15 fact** follows the lineage convention via `lineage_upsert(conn, *, run, table, inbound_table, natural_key, payload_cols, hash_cols)` in `src/ccutils/etl/upsert.py`. Lineage block on every row: `created_at`, `last_updated_at`, `created_by_version_key`, `last_updated_by_version_key`, `etl_run_id`, `record_source`, `hash_diff`, `is_deleted`, `deleted_at`.
- **Closures wrapping `run_v15_etl` MUST list every swallowed kwarg explicitly, not `**kwargs`.** A `**_legacy_kwargs` shim silently drops args (the `--private` regression hid here for one commit). Name what you discard so signature drift fails loud.
- **`_PROGRESS_TABLES` in `src/ccutils/export/duckdb_archive.py` MUST list every fact `run_v15_etl` populates.** Stale entries undercount the progress display by multiples (3-5× on real corpora). Update when adding a new populator.

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
- **Adding a column to a table that already shipped (0.17.0+): append it to `_COLUMN_MIGRATIONS` in `schemas/star/schema.py`.** The warehouse is persistent and `CREATE TABLE IF NOT EXISTS` never widens an existing table -- without the migration entry, old warehouses break on the populator's INSERT (or on a view that references the new column). Migrations run after the CREATE TABLEs and BEFORE the views. Canonical test: `TestFactPlanRevisionsMigration`.

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
- **CLI honesty guards for v0.15:** `--private` is HTML-only on v0.15 (no PathSanitizer wiring in `src/ccutils/etl/`); `local_cmd`/`all_cmd` raise `click.UsageError` rather than silently no-op (silent no-op shipped a privacy regression once). `--no-thinking` IS wired -- it flows through `run_v15_etl(include_thinking=False)` → `extract_text_from_content_json(..., include_thinking=False)` so thinking is excluded from the FACTS (`dim_session.last_assistant_message`, Tier 2 inputs). Raw `message_json` never survives in staging regardless of the flag: `staging_scope` in `run_v15_etl` clears `stg_log_entries` unconditionally at exit (this replaced the old `--no-thinking`-only `DELETE`). `fact_messages.content_text` excludes thinking by SQL projection regardless of the flag. The Parquet lake intentionally retains everything (re-derivable cache); delete it post-run if you don't want thinking in any cache.
- **Exit-code-only flag tests are insufficient:** `assert result.exit_code == 0` documents "CLI accepted the flag", not "the flag did what its help text claims." Test the flag's actual effect (paths sanitized in stored content, thinking blocks absent, etc.).
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
│   ├── schemas/              # Schema definitions (v0.15 star only)
│   │   ├── __init__.py       # Re-exports the star schema public surface
│   │   └── star/
│   │       ├── __init__.py
│   │       ├── schema.py     # DDL for every star table + semantic views (single source of truth)
│   │       ├── json_export.py# JSON export of the star schema
│   │       ├── embeddings.py # Optional ColBERT embedding pipeline (requires pylate)
│   │       └── utils.py      # Key generation, tool/model classification helpers
│   ├── etl/                  # v0.15 four-tier ETL -- every fact populator lives here
│   │   ├── orchestrator.py   # run_v15_etl() entry point + populator order
│   │   ├── lineage.py        # EtlRun + hash_diff + record_source_label
│   │   ├── upsert.py         # lineage_upsert() shared by every fact populator
│   │   ├── staging.py        # load_session_to_staging() (Tier 1 -> Tier 2)
│   │   ├── utils.py          # Shared ETL utilities (extract_text_from_content_json)
│   │   ├── heuristics.py     # dim_session intent/complexity/outcome/domain classifiers
│   │   ├── dim_session_heuristics.py # populator that runs the classifiers
│   │   ├── dim_session_chain.py
│   │   ├── dim_prompt.py             # history.jsonl import
│   │   ├── subagent_enrichment.py    # dim_session.is_agent / parent_session_key / agent_type
│   │   ├── fact_messages.py
│   │   ├── fact_tool_calls.py        # fact_tool_uses + fact_tool_results
│   │   ├── fact_tool_chain_steps.py
│   │   ├── fact_token_usage.py
│   │   ├── fact_session_summary.py   # MUST run last; aggregates over every other fact
│   │   ├── fact_session_facets.py    # Tier 1 facet populator (F01-F19, SQL-computed)
│   │   ├── fact_file_operations.py   # + dim_file
│   │   ├── bridge_session_file.py
│   │   ├── fact_errors.py
│   │   ├── fact_diagnostics.py
│   │   ├── fact_plan_revisions.py
│   │   ├── fact_agent_delegations.py
│   │   ├── entry_type_facts.py       # attachments, progress, system, meta, file_history, queue_ops, pr_links
│   │   └── facets/                    # Tier 2 LLM-extracted facets
│   │       ├── catalog.py             # FACET_SPECS registry + facet_tier_scope_sql helper
│   │       ├── extractor.py           # FacetExtractor Protocol + dataclasses + CannedFacetExtractor
│   │       ├── anthropic.py           # AnthropicFacetExtractor (Haiku via httpx)
│   │       └── populator.py           # populate_tier2_facets
│   ├── export/                # Export format handlers
│   │   ├── __init__.py
│   │   ├── html.py           # HTML generation
│   │   ├── markdown.py       # Markdown generation (render-only, one .md per session)
│   │   └── duckdb_archive.py # DuckDB / JSON batch export (drives run_v15_etl)
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
│   ├── conftest.py           # Shared fixtures (sample_session_file, ...)
│   └── test_*.py             # ~56 test files; v0.15 facts split per-populator into test_<fact>_v15.py
├── docs/
│   ├── STAR_SCHEMA.md            # Star schema reference (DDL + populator-by-populator notes)
│   └── FACET_CLUSTER_PIPELINE.md # Facet pipeline design + status
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

Four output formats; only one schema:

- `--format html` - Browsable transcript pages (interactive use).
- `--format markdown` - One `.md` per session (render-only, like html: no ETL, no warehouse). Messages as headings, tool uses as fenced blocks in `<details>`, thinking as blockquotes. Honors `--no-thinking` and `--private`.
- `--format duckdb` - DuckDB database file. Writes the v0.15 star schema.
- `--format json` - Directory with `meta.json` + `dimensions/*.json` + `facts/*.json`. Same star schema as `duckdb`, serialized to JSON.
- `ccutils all` also accepts `--format both` (HTML + DuckDB).

Pipeline entry: `run_v15_etl()` in `src/ccutils/etl/orchestrator.py`. DDL: `create_star_schema()` in `src/ccutils/schemas/star/schema.py`. `--embed [MODEL]` (ColBERT, requires pylate) and `--with-llm-facets` / `--batch-llm-facets` (Tier 2 LLM facets) are available on the DuckDB / JSON paths.

**Legacy simple 4-table schema removed.** The `--format duckdb-star` / `--format json-star` aliases are gone -- the `-star` suffix is no longer accepted. `ccutils import` is HTML-only now (the Claude.ai export shape doesn't match v0.15's Claude Code grain).

**Defaults**: Thinking blocks and subagents/agents are included by default. Use `--no-thinking`, `--no-subagents` (local), or `--no-agents` (all) to exclude them.

### 3. Star Schema Tables

**Authoritative list:** every fact populated by `run_v15_etl` is also listed in `_PROGRESS_TABLES` in `src/ccutils/export/duckdb_archive.py`. Update both together.

**Lineage / Meta:** `dim_etl_version`, `fact_etl_runs`, `meta_schema_version`.

**Dimensions:** `dim_session` (enriched with intent/complexity/outcome/domain + subagent linkage via `populate_dim_session_heuristics` + `populate_subagent_dim_session`), `dim_project`, `dim_tool`, `dim_model`, `dim_file`, `dim_session_chain`, `dim_prompt`, `dim_facet_type` (facet registry, seeded). `dim_time` (seeded at DDL time, 1440 rows), `dim_date` (rows inserted during ETL for every date seen in staging; date_key / time_key integer surrogates on facts are derived inline and join these dims).

**Staging:** `stg_log_entries` (one row per JSONL line, Tier 2 of the four-tier pipeline).

**Facts populated by `run_v15_etl` (per-session, in dependency order):**
- `fact_messages` (stop_reason, permission_mode_at_send, prompt_id, request_id, is_api_error_message)
- `fact_tool_uses` + `fact_tool_results` (R1 structured `toolUseResult` capture; per-tool typed cols on results: Bash `exit_code`/`interrupted`, Edit `structured_patch`, Read `num_lines`, Agent rollup)
- `fact_token_usage` (R11 cache split: `cache_creation_5m_tokens` + `cache_creation_1h_tokens` + `total_uncached_equivalent_tokens`)
- `fact_attachments` (23 attachment subtypes), `fact_progress_events` (6 variants), `fact_system_events` (7 subtypes), `fact_meta_events` (permission-mode time series + custom-title + agent-name + last-prompt)
- `fact_file_history_snapshots`, `fact_queue_operations`, `fact_pr_links`
- `fact_file_operations` (+ `bridge_session_file` aggregate), `fact_diagnostics`, `fact_plan_revisions` (structural outcome via `fact_tool_results.is_error`), `fact_agent_delegations` (Task spawn + agent rollup; cross-session linkage via `dim_session.agent_id`)
- `fact_errors`, `fact_tool_chain_steps`
- `fact_session_facets` Tier 1 (F01-F19, SQL-computed; runs before fact_session_summary)
- `fact_session_summary` (MUST run last; aggregates over every other fact)

**Optionally populated:** `fact_session_facets` Tier 2 rows (F20+, via `--with-llm-facets` / `--batch-llm-facets`); `fact_facet_embeddings` (Step 5, not built yet).

**Not yet populated (DDL only):** `fact_content_blocks`, `fact_code_blocks`, `fact_entity_mentions`, `fact_session_embeddings`, `fact_tool_input_params`. Some redundant-with-v0.15 facts also remain as DDL stubs (`fact_turn_durations` / `fact_stop_events` are subsumed by `fact_system_events`; `fact_tool_calls` is subsumed by `fact_tool_uses` + `fact_tool_results`).

**Semantic views (15):** `semantic_sessions`, `semantic_messages`, `semantic_tool_calls` (UNION over uses+results), `semantic_token_usage`, `semantic_cost_analysis` (R11-corrected hit-rate denominator), `semantic_prompt_history`, `semantic_session_chains`, `semantic_project_context`, `semantic_decisions` (decision timeline UNION over plan revisions + permission-mode changes + stop/api_error/compact system events), `semantic_agent_delegations`, `semantic_file_evolution`, `semantic_file_operations`, `semantic_plan_revisions`, `semantic_project_files`, `semantic_tool_patterns`.

### 4. Token Tracking (v0.15)

**Actual tokens** (from API usage data on assistant messages):
- `fact_token_usage`: per-API-response breakdown. R11 fix: cache_creation split per pricing tier (`cache_creation_5m_tokens`, `cache_creation_1h_tokens`, `cache_creation_total_tokens`), plus `total_uncached_equivalent_tokens` = input + creation_total + read.
- `fact_messages`: `input_tokens` (renamed from `actual_input_tokens`; honestly named per Anthropic semantics — post-last-cache-breakpoint, NOT total uncached), `output_tokens`, `cache_creation_5m_tokens`, `cache_creation_1h_tokens`, `cache_read_tokens`, `total_uncached_equivalent_tokens`.
- `fact_session_summary`: aggregated `total_*` versions of all the above.
- `semantic_cost_analysis` view: `cache_hit_rate_pct` denominator now includes cache_creation (legacy view excluded it, over-stating hit rate).

**Estimated tokens** removed -- the legacy `estimate_tokens()` lived in `schemas/star/extractors.py`, which was deleted with the simple-schema cull. Sessions predating API-side usage data carry NULLs for the actual-token columns; that's the honest signal.

### 5. Heuristic Classification

Runs during ETL with zero external dependencies. Classifiers live in `src/ccutils/etl/heuristics.py`; the populator wiring them onto `dim_session` is `src/ccutils/etl/dim_session_heuristics.py`. Outputs: `intent`, `complexity`, `outcome`, `domain`, plus a separate `classify_error_type` used by `populate_fact_errors`.

## Testing

```
uv run pytest                   # full suite (~1046 tests, incl. 1 skipped live-API)
uv run pytest tests/test_<fact>_v15.py -v  # one populator
uv run pytest --cov=ccutils    # with coverage
```

v0.15 facts are split per-populator into `tests/test_<fact>_v15.py` files; HTML / template tests live in `tests/test_generate_html.py` + `tests/test_html_css_coverage.py` and need `--snapshot-update` after CSS/macro changes.

## Common Workflows

### Adding a new v0.15 fact table
1. Write failing tests (DDL existence + lineage columns + populator behavior + idempotency) in `tests/test_<fact>_v15.py`.
2. Add `CREATE OR REPLACE TABLE` in `schemas/star/schema.py` with the standard lineage block (created_at, last_updated_at, version keys, etl_run_id, record_source, hash_diff, is_deleted, deleted_at + degenerate dims). If multiple populators will write the table, use `CREATE TABLE IF NOT EXISTS` instead (see `dim_facet_type`).
3. Add `populate_fact_<name>(conn, *, run)` in `src/ccutils/etl/<fact>.py`. Build inbound temp table from staging (or from already-populated facts with the staging-scoping guard); delegate to `lineage_upsert()`.
4. Wire into `run_v15_etl()` in `src/ccutils/etl/orchestrator.py` in dependency order — anything `fact_session_summary` aggregates over MUST run before it.
5. Add the new table name to `_PROGRESS_TABLES` in `src/ccutils/export/duckdb_archive.py`.
6. Tests green; CHANGELOG entry under `[Unreleased]`; update `docs/STAR_SCHEMA.md` if the table is non-trivial.

### Removing a feature
1. Grep for all imports, call sites, `__all__` exports, CLI registrations, and test references.
2. Delete source files; remove from `cli/__init__.py`, `schemas/__init__.py`, top-level `__init__.py`.
3. Remove tests, update CHANGELOG, CLAUDE.md project tree, README, `_PROGRESS_TABLES`.
4. CLI help text + docstrings: grep for stale table/view counts or feature references.

### v0.15 ETL pipeline (per session)
```python
from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl

conn = create_star_schema(db_path)
run_v15_etl(
    conn,
    session_path,
    project_name="...",
    parquet_lake_root="/path/to/parquet_lake",
    facet_extractor=None,  # or AnthropicFacetExtractor(...) for Tier 2
)
# Optional: ccutils.schemas.star.embeddings.EmbeddingPipeline(conn).embed_sessions(conn)
```

## Versioning

- Version lives in `pyproject.toml`; keep it in sync with `CHANGELOG.md`.
- Tag releases: `git tag v0.X.0 <commit> -m "v0.X.0: summary"`.
- `[Unreleased]` is the in-flight section; promote it on release.

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
