<!-- path-privacy: skip-file -- references universal ~/.claude data paths (not personal) -->
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **`fact_plan_revisions.plan_file_path`** -- captures `input.planFilePath` from `ExitPlanMode` (newer Claude Code writes the plan to a file and passes its path alongside the plan text). NULL for sessions predating the field. Exposed on `semantic_plan_revisions`.
- **Column-migration mechanism for the persistent warehouse.** `create_star_schema()` now applies an append-only `_COLUMN_MIGRATIONS` list (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...`) after the CREATE TABLEs and before the views: `CREATE TABLE IF NOT EXISTS` never widens an existing table, so any column added after a table shipped needs a migration entry or pre-existing warehouses break on the populator's INSERT. `plan_file_path` is the first entry.

### Fixed
- **Semantic views no longer return NULL dates.** Nine views LEFT JOIN `dim_date` / `dim_time` for `full_date` / `day_name` / `time_of_day`, but neither dim was ever populated in v0.15 -- every date/time attribute came back NULL. `dim_time` is now seeded at DDL time (fixed 1440-minute dimension, `time_of_day` buckets mirroring `get_time_of_day`), and `_upsert_minimal_dimensions` inserts a `dim_date` row for every calendar date seen in staging (set-based; `day_of_week` mirrors Python's Monday=0 like `ensure_dim_date`).
- **`--private` was a silent no-op for HTML export of JSONL sessions.** Normalized JSONL loglines carry only `type`/`timestamp`/`message` -- never `cwd` -- so `_sanitize_loglines`'s fallback scan could not find a cwd and returned the loglines untouched; the only test was exit-code-only (the exact anti-pattern CLAUDE.md warns about). `generate_html` now resolves cwd from the session file via a shared `extract_header_fields()` scan and passes it to the sanitizer. Effect-asserting regression tests render a session and assert the cwd prefix is absent (`TestPrivateModeSanitizesJsonl`).
- **Header fields survive a leading `summary` line.** `extract_rich_metadata` latched header fields (sessionId, cwd, gitBranch, version) from the FIRST line only, so sessions opening with a headerless entry lost them; `extract_session_metadata` (agent/sidechain detection) had the same first-line fragility. Both now scan past headerless lines to the first occurrence of each field (cwd keeps the session's STARTING value; mid-session `cd` doesn't overwrite). The markdown exporter's private `_scan_header_fields` workaround is replaced by the shared `extract_header_fields`.

## 0.17.0

The star schema becomes a persistent, incrementally-updatable warehouse (tables accumulate across CLI runs instead of being wiped), HTML export goes CSP-strict with externalized assets, a render-only `--format markdown` lands on `local` and `all`, and the ETL-rethink proposal's Layer 6 decision backbone ships as the `semantic_decisions` view. Post-review fixes close a Tier 2 facet data-loss path and a `local` export crash; subagent depth propagation is scoped and no longer misses parents that arrive after their children.

### Added
- **`--format markdown`** on `ccutils local` and `ccutils all` -- one `.md` file per session, render-only (no ETL, no warehouse, no templates). Messages render as headings, tool uses as fenced code blocks inside `<details>` (results mapped to their tool call and truncated at 1500 chars; fences grow past embedded backtick runs), thinking as blockquoted subsections. Honors `--no-thinking` and `--private` (same PathSanitizer treatment as HTML; covered by effect-asserting CLI tests, not exit-code-only). `ccutils all --format markdown` writes a per-project directory tree without index pages. Also hardened header extraction: sessions whose first line is a non-message entry (e.g. `summary`) get a raw-line fallback scan so `--private` can still resolve `cwd` (the shared extractor keeps its first-entry-only behavior; upstreaming is a known follow-up).
- **`semantic_decisions` view** -- one unified decision timeline UNIONing the structural decision signals the v0.15 ETL already captures: plan revisions (`fact_plan_revisions.outcome`), permission-mode changes (`fact_meta_events`), and stop / api_error / compact_boundary system events (`fact_system_events`). This is the ETL-rethink proposal's "fact_decisions backbone" (Layer 6) delivered as a pure projection: every signal it wanted already lands in a fact, so no new table or populator is needed. `source_key` + `source_table` on every row link back to the underlying fact row.
- `fetch_scalar(conn, sql, params)` helper in `etl/utils.py` -- replaces the bare `.fetchone()[0]` pattern (unsound when a query can return zero rows); raises a descriptive RuntimeError instead of an opaque NoneType subscript. Migrated the three remaining call sites.

### Fixed
- **Subagent depth propagation now runs on every ETL call, and only touches the current session's tree.** Two defects in `_propagate_depth_level`: (1) it globally reset + recomputed `depth_level` for the ENTIRE `dim_session` on every per-session ETL -- O(N^2) writes across a batch on the persistent warehouse (deferred finding #3 from the 2026-07-10 review); (2) it only ran at all when the current session WAS a subagent, so a parent session landing after its children never re-rooted them -- the children kept `depth_level = 0` forever. The recompute is now scoped: walk UP from the staged session to its root, recompute DOWN through that root's subtree, leave unrelated trees untouched. Regression tests: `test_parent_arriving_after_child_fixes_child_depth`, `test_unrelated_tree_depth_untouched_by_new_session`.

### Fixed (post-review)
- **Tier 2 facets are no longer destroyed by a failed re-extraction.** `populate_tier2_facets` now scopes its soft-delete to sessions that actually produced rows this run (`session_id IN _INBOUND`), not every session in staging. Previously, re-ETLing a session with `--with-llm-facets` while the LLM API was failing left the inbound empty, and the widened staging-based soft-delete marked the session's existing good facets `is_deleted=TRUE` -- data loss from a transient failure. Tier 1 (deterministic, always emits a row per session) keeps the default scope. Regression test: `test_failed_reextraction_preserves_prior_tier2_facets`.
- **`ccutils local --format duckdb/json` no longer aborts on one empty/unparseable session.** The single-file export looped `run_v15_etl` with no error handling, so a session with no valid entries (which now raises `ValueError` from `write_session_to_parquet`) crashed the whole command with a traceback and leaked the connection. A new `_etl_session_files` helper isolates per-file failures -- reporting and skipping them -- mirroring the batch `all` path. Regression test: `test_empty_session_file_is_skipped_not_crashed`.
- **Narrowed a `.gitignore` pattern.** The JSON-export ignore block dropped the unanchored `meta.json` entry (which would silently ignore any `meta.json` anywhere in the repo); the data-bearing `dimensions/` + `facts/` dirs stay ignored, and the export's `meta.json` holds only schema metadata + row counts (no transcript data).

### Removed (post-review)
- **`stg_task_agent_map` dropped.** The staging table was created, cleared by `staging_scope`, and listed in the JSON export, but no populator ever wrote to or read from it -- vestigial infrastructure. Removed the DDL, the `staging_scope` DELETE, the `json_export` entry, and the three e2e assertions that vacuously checked it stayed empty (they passed regardless of ETL behavior). The redundant `--no-thinking`-only `DELETE FROM stg_log_entries` in `run_v15_etl` is also gone: `staging_scope` clears staging unconditionally, so raw thinking never survives the run regardless of the flag.

### Changed
- **The star schema is now a persistent, incrementally-updatable warehouse.** Every table in `create_star_schema()` switched from `CREATE OR REPLACE TABLE` to `CREATE TABLE IF NOT EXISTS`, so re-running the CLI accumulates sessions instead of wiping the warehouse each run. Aggregate/rollup populators are now scoped to the current session so a persistent warehouse doesn't get rescanned or clobbered: every rollup CTE in `fact_session_summary` and the soft-delete in `lineage_upsert` gate on `session_id IN (SELECT session_id FROM stg_log_entries ...)`, and `run_v15_etl` wraps its body in a `staging_scope` context manager that clears `stg_log_entries` + `stg_task_agent_map` on exit (staging is always cleared now, not only under `--no-thinking`).
- **Subagent depth is computed with a single recursive CTE.** `_propagate_depth_level` replaced its 100-iteration Python cursor loop with one `WITH RECURSIVE` DuckDB update over the parent/child chain.

### Fixed
- **Session complexity now reflects real agent depth.** `populate_dim_session_heuristics` joins `dim_session` for `depth_level` and passes it to `classify_complexity` (previously hardcoded `0`, so deep agent trees never earned the depth bonus).
- **`.js` / `.ts` files classify as the `web` domain** (previously fell through to `unknown`).
- **HTML export is CSP-strict.** `base.html` drops `'unsafe-inline'` from `style-src` / `script-src` (both now `'self'`); CSS/JS are written as external `transcript.css` / `transcript.js` / `search.js` / `global_search.js` files linked via a per-page-depth `rel_path`. All remaining inline `style=` attributes were moved into CSS classes (`.header-link`, `.page-subtitle`, `.index-item-size`, `.index-actions`, `.back-button`, `.image-block img`), and the search renderers + JSON highlighter were rewritten from `innerHTML` string-building to DOM-node construction, so nothing trips the tightened policy. (The inline styles were a live regression: the CSP was tightened before the attributes were removed, so headers, muted text, and image sizing were silently blocked in the browser while every test still passed.)
- `write_session_to_parquet` raises `ValueError` on a JSONL file with no valid entries instead of writing an empty Parquet table.

### Added
- `tests/test_e2e_star.py` -- a 50-test end-to-end suite exercising all four warehouse tiers, the heuristic classifiers, and the HTML output. Run the suite with `uv run pytest tests/ --confcutdir=tests` so parent-workspace imports don't shadow the package.
- CSP regression guard in `tests/test_html_css_coverage.py` -- statically scans every template (and the rendered sample) for inline `style=` / `<style>` / `on*=` / inline `<script>`, all silently blocked by the tightened `*-src 'self'` CSP. This is the check that would have caught the inline-style regression above at commit time.

## 0.16.0

The facet & cluster pipeline lands its first five steps (Tier 1 SQL facets, the Tier 2 LLM extractor boundary, the F20 `task_description` populator, and the `--with-llm-facets` / `--batch-llm-facets` CLI flags), the legacy simple 4-table schema is removed so the v0.15 star schema is the only one, and `--no-thinking` is honored on the DuckDB / JSON paths. Breaking: `--format duckdb` / `--format json` now write the star schema (the `-star` suffix is gone), `ccutils import` is HTML-only, and several simple-schema Python re-exports were dropped.

### Behavior
- **`--no-thinking` now works on `--format duckdb` and `--format json`.** Previously the CLI rejected the flag on those paths with a `click.UsageError` based on a misread of the populator (it assumed thinking text was in `fact_messages.content_text`; it never was — the SQL projection filters to `type='text'` blocks). The flag now flows through `run_v15_etl(include_thinking=False)` → `extract_text_from_content_json(..., include_thinking=False)` (so `dim_session.last_assistant_message` and Tier 2 facet inputs exclude thinking) and finishes with a `DELETE FROM stg_log_entries` so no raw thinking JSON survives in the warehouse staging table. The Parquet lake is intentionally untouched (it's the re-derivable cache); delete it post-run if you don't want thinking in any cache. `--private` remains rejected on duckdb/json until PathSanitizer is wired through the v0.15 ETL.

### Breaking
- **Simple 4-table schema removed.** `src/ccutils/schemas/simple/` is gone; `--format duckdb` and `--format json` now write the v0.15 star schema unconditionally. Previously `duckdb`/`json` produced a 4-table snapshot (`sessions`, `messages`, `tool_calls`, `thinking`) and the star schema lived behind `--format duckdb-star` / `--format json-star`. Migration: any scripts that read the 4-table shape need to switch to the star schema (query `fact_messages` / `fact_tool_uses` / `fact_session_summary` instead of `messages` / `tool_calls` / `sessions`). The `-star` suffix is no longer accepted; pass `duckdb` and `json` instead.
- **`ccutils import` is HTML-only.** The legacy `import --format duckdb` path went through the now-removed simple schema. The Claude.ai export shape doesn't match v0.15's Claude Code session JSONL grain, so no automatic migration to star; if a Claude.ai → star ETL is wanted later it warrants its own populator.
- **Internal renames** (impact callers of the public Python API): `ccutils.schemas.resolve_schema_format` removed (no longer needed; only one schema). `ccutils.export.generate_star_json_archive` → `ccutils.export.generate_json_archive`. `ccutils.export.generate_duckdb_archive` no longer accepts a `schema_type` parameter. `ccutils.schemas.create_duckdb_schema` / `export_session_to_duckdb` / `export_sessions_to_json` / `_extract_session_data` are gone (the four were the simple-schema re-exports).

### Added
- **Facet & cluster pipeline -- step 1 (DDL + Tier 1 registry).** Three new tables land in `create_star_schema()`:
  - `dim_facet_type` -- registry of facet definitions. Seeded with the 19 Tier 1 facets (F01-F19) from `docs/FACET_CLUSTER_PIPELINE.md` §3. All `method='computed'`, no prompt fields.
  - `fact_session_facets` -- one row per (session, facet_type, prompt_version). Typed value columns (text/json/numeric/bool). Carries the full v0.15 lineage envelope.
  - `fact_facet_embeddings` -- one row per (session, facet_type, embedding_model, embedding_model_version). Stores the vector as `FLOAT[384]` so DuckDB's native `array_cosine_similarity` / `array_inner_product` work without a vector DB. Lineage envelope on every row.
- **Schema-split decision** (departure from original design doc): embeddings live in their own table rather than as an inline `BLOB` column on `fact_session_facets`. Rationale lives in `docs/FACET_CLUSTER_PIPELINE.md` §4.
- **Synthesized natural-key columns** -- `fact_session_facets.facet_row_key = md5(session_id || facet_id || prompt_version)` and `fact_facet_embeddings.embedding_row_key = md5(session_id || facet_type_key || model || model_version)`. Step-1 follow-up surfaced by step 2 (`lineage_upsert` takes a single-column natural key).
- **Facet & cluster pipeline -- step 2 (Tier 1 SQL populator).** New `populate_tier1_facets(conn, *, run)` in `src/ccutils/etl/fact_session_facets.py`. Computes all 19 Tier 1 facets per session via SQL aggregations over the v0.15 facts (`dim_session` heuristics, `fact_tool_uses`, `fact_tool_chain_steps`, `fact_errors`, `fact_file_operations`, `fact_token_usage`, `fact_agent_delegations`, `fact_pr_links`, `fact_plan_revisions`). Bool facets (F17/F18/F19) default to `FALSE` on absence; JSON facets default to `[]` so downstream consumers can rely on schema-stable shapes.
- Wired into `run_v15_etl` after the heuristic / chain populators and before `populate_fact_session_summary`, preserving the "summary runs last" invariant.
- **Facet & cluster pipeline -- step 3 (Tier 2 extractor + orchestrator hook).** Lands the LLM-extractor boundary so step 4's `populate_tier2_facets` is mechanical:
  - `src/ccutils/etl/facets/extractor.py` -- `FacetExtractor` Protocol (runtime_checkable), frozen `FacetSpec` / `SessionInputs` / `FacetOutput` dataclasses, and `CannedFacetExtractor` fake-backend for tests.
  - `src/ccutils/etl/facets/anthropic.py` -- `AnthropicFacetExtractor` calls `api.anthropic.com/v1/messages` via `httpx` (no SDK dep). Default model `claude-haiku-4-5-20251001`. Per-facet validation via dynamic Pydantic schema (built from the enabled `FacetSpec` list, `extra="ignore"`). Sentinel-wrapped user prompt, cached system prompt. Two retry budgets: HTTP (3 retries on 429 / 5xx / network) and validation (1 retry on bad JSON / Pydantic error). Soft-fail (null / missing / empty) emits `is_fallback=True` without retry; hard-fail (wrong type / bad enum) triggers the one validation retry. 401 raises `FacetExtractionError` (no retry; CLI surfaces it cleanly).
  - `src/ccutils/api/resolve_anthropic_key()` -- env-var-first (`ANTHROPIC_API_KEY`), keychain fallback (service `ccutils-anthropic`, Darwin only), fails loud with both options spelled out.
  - DDL: `fact_session_facets` gains `is_fallback BOOLEAN NOT NULL DEFAULT FALSE` (QA aid; queryable "where did extraction fail") and `extraction_metadata_json JSON NULL` (raw response + retry_count + cache_hit for debugging cluster QA).
  - DDL: `dim_facet_type` seed switches from `CREATE OR REPLACE` to `CREATE TABLE IF NOT EXISTS` + `INSERT ... ON CONFLICT DO NOTHING`. Historical `prompt_version` rows survive `create_star_schema()` re-runs (the CLI path), preserving the registry that `fact_session_facets` rows reference by `facet_type_key`.
  - `run_v15_etl(..., facet_extractor: FacetExtractor | None = None)` -- new optional parameter. None (default) disables Tier 2 entirely (existing callers unchanged). When supplied, runs the Tier 2 populator stub between Tier 1 facets and `fact_session_summary`. Body of the stub is step 4.
  - Design doc lives at `internal/plans/facet_extractor_protocol.md` (gitignored).
- **Facet & cluster pipeline -- step 4 (F20 task_description populator).** Lands the first Tier 2 facet end-to-end:
  - `src/ccutils/etl/facets/catalog.py` -- `FACET_SPECS` registry (F20 v1 currently), and `facet_tier_scope_sql(tier)` helper that produces the `lineage_upsert` soft-delete scope clause for a given tier. Centralized so cross-tier soft-delete interference can't recur.
  - `src/ccutils/etl/facets/populator.py` -- `populate_tier2_facets(conn, *, run, extractor)`. Builds a `SessionInputs` per staged session from `fact_messages` / `fact_tool_uses` / `fact_token_usage` / `dim_session`, calls `extractor.extract(inputs, FACET_SPECS)`, writes one `fact_session_facets` row per (session × spec) via `lineage_upsert` with `is_fallback` + `extraction_metadata_json` flowing through. Per-session failure isolation: a raising extractor logs + skips that session; other sessions still process. Uses `executemany` to push all inbound rows in one DuckDB call.
  - `src/ccutils/etl/facets/extractor.py` -- new `OUTPUT_TYPE_TO_COL` constant mapping `FacetSpec.output_type` to the storage column. Removes the implicit dispatch contract from `populator.py`.
  - `src/ccutils/etl/utils.py` (new) -- `extract_text_from_content_json` lifted out of `dim_session_heuristics` (it's now used by two populators). Both call sites import from utils; `dim_session_heuristics._extract_text` is gone.
  - `AnthropicFacetExtractor.extract` now stamps the full `extraction_metadata_json` schema documented in protocol §3.1: `raw_response`, `prompt_version`, `retry_count`, `cache_hit`, `input_tokens`, `output_tokens`, `latency_ms`. Wall-clock latency captured around the whole `extract()` call (includes backoff sleeps).
  - **`lineage_upsert` gains `soft_delete_scope_sql: str | None`** -- optional extra WHERE clause for the soft-delete step. Needed because `fact_session_facets` is now written by two populators (Tier 1 + Tier 2); without scoping, each populator's soft-delete would wipe the other's rows on the next run. Both tier populators pass `facet_tier_scope_sql(N)`. Default `None` preserves existing populator semantics.
  - `dim_facet_type` seeded with F20 v1 (Tier 2) row in `create_star_schema` alongside the Tier 1 seeds; same `ON CONFLICT DO NOTHING` pattern so historical prompt_versions survive re-runs.
  - Orchestrator's stub call replaced with the real `populate_tier2_facets` import; old `_populate_tier2_facets_stub` deleted.
  - Tests (`tests/test_populate_tier2_facets.py`): 8 cases covering canned-value happy path, missing-canned fallback, default-None skip, extractor-raises isolation, dim_facet_type seed, metadata storage, idempotency. Live-API smoke test (`TestLiveApiSmoke`) skips unless `ANTHROPIC_API_KEY` is set.
- **Facet & cluster pipeline -- step 4.5 (CLI flags).** Tier 2 extraction is now reachable from the CLI:
  - `ccutils local --with-llm-facets ...` -- single-session opt-in.
  - `ccutils all --batch-llm-facets ...` -- batch opt-in.
  - Both flags live in a new `Enrichment` option group (the original `Embeddings` group was misleading -- LLM facets aren't embeddings).
  - Both flags resolve credentials via `resolve_anthropic_key()` (env var first, macOS keychain second), construct `AnthropicFacetExtractor`, and thread it through to `run_v15_etl` via `facet_extractor=`. `CredentialsError` is caught at the CLI boundary and emits a helpful message + non-zero exit; users never see a stack trace. Construction happens once in the command entry point (`local_cmd` / `all_cmd`) before any session-parsing work begins.
  - Shared `build_facet_extractor_or_exit` helper in `src/ccutils/cli/utils.py` -- both CLI commands import it, so credential-resolution wording stays in lockstep.
  - `generate_duckdb_archive` and `generate_json_archive` accept the new `facet_extractor=None` parameter and pass it through the per-session ETL loop.
  - **Footgun:** pairing `--batch-llm-facets` with `--format json` runs the full LLM extraction against a temporary DuckDB that's discarded after the JSON export. The user pays the API cost but gets no queryable DuckDB to inspect F20 outputs. Documented in README; long-term fix is either keeping the DuckDB sidecar or surfacing a warning at invocation time.
  - **Live-API smoke command** (run once after any change to `AnthropicFacetExtractor` or its prompt template; costs pennies):
    ```bash
    ANTHROPIC_API_KEY=$(security find-generic-password -s ccutils-anthropic -a $USER -w) \
      uv run pytest tests/test_populate_tier2_facets.py::TestLiveApiSmoke -v
    ```

## 0.15.0

A reshape of the ETL and most of the star schema, driven by a structural-signal bug in v0.14's plan-revision outcome classification. v0.15 widens what gets captured from Claude Code JSONL (all 12 entry types, 23 attachment subtypes, 6 progress variants, 7 system subtypes), corrects cache-token arithmetic for the 5m/1h pricing tiers, splits `fact_tool_calls` into `fact_tool_uses` + `fact_tool_results` with structured per-tool `toolUseResult` payloads, and adds a lineage block to every fact so re-ETL on unchanged source is a verifiable no-op.

**Re-ETL recommended.** v0.15 reshapes most of the star schema and renames several columns. Existing v0.14 databases will not read cleanly against the new code -- delete `archive.duckdb` and re-run `ccutils all --format duckdb-star` to rebuild on the new pipeline.

### Added
- **Four-tier ETL pipeline.** New `run_v15_etl(conn, session_path, *, project_name, parquet_lake_root)` in `src/ccutils/etl/orchestrator.py` is the single per-session entry point. Tiers: raw JSONL -> Parquet lake (Tier 1, `parquet_lake/projects/<project>/<session>/log_entries.parquet`) -> DuckDB staging (`stg_log_entries`) -> fact / dim tables.
- **Pydantic discriminated-union parser.** `src/ccutils/parsers/models.py` covers all 12 Claude Code 2.x entry types. `extra="allow"` on every model so undocumented fields land in Parquet immediately.
- **Lineage convention on every fact.** `created_at`, `last_updated_at`, `created_by_version_key`, `last_updated_by_version_key`, `etl_run_id`, `record_source`, `hash_diff`, `is_deleted`, `deleted_at`. New `dim_etl_version` and `fact_etl_runs` track every batch; `meta_schema_version` tracks DDL migrations.
- **hash_diff-gated UPDATEs.** Re-running the ETL on unchanged source produces zero UPDATEs, so `last_updated_at` is a precise temporal signal rather than "last ETL touch."
- **Shared `lineage_upsert(conn, *, run, table, inbound_table, ...)`** helper in `src/ccutils/etl/upsert.py` used by every v0.15 fact populator.
- **R1 -- structured `toolUseResult` capture.** `fact_tool_results` gains per-tool typed columns: Edit `edit_structured_patch_json`, Bash `bash_exit_code` / `bash_interrupted` / `bash_stdout_bytes` / `bash_duration_ms`, Read `read_num_lines` / `read_total_lines`, Glob, Grep, WebFetch, Agent rollup (`agent_total_duration_ms`). JSON catch-all (`result_payload_json`) for unknown tools.
- **R11 -- cache-token split.** `fact_token_usage`, `fact_messages`, and `fact_session_summary` split `cache_creation_tokens` into `cache_creation_5m_tokens` (1.25x pricing) and `cache_creation_1h_tokens` (2x pricing). New `total_uncached_equivalent_tokens` = input + creation_total + read. `semantic_cost_analysis.cache_hit_rate_pct` denominator now includes cache_creation (legacy denominator over-stated the hit rate).
- **R16 -- tri-state `is_error`.** `fact_tool_results.is_error` is nullable BOOLEAN; missing-vs-false-vs-true are now distinct states.
- **All 12 entry types captured.** New fact tables for the entry types v0.14 dropped entirely: `fact_attachments` (all 23 attachment subtypes), `fact_progress_events` (hook, bash, agent, query update, search results, MCP), `fact_system_events` (turn_duration, stop_hook_summary, api_error, compact_boundary, local_command, away_summary, bridge_status), `fact_meta_events` (permission-mode time series + custom-title + agent-name + last-prompt), `fact_file_history_snapshots`, `fact_queue_operations`, `fact_pr_links`.
- **fact_messages widened.** New columns: `stop_reason`, `permission_mode_at_send`, `prompt_id`, `request_id`, `is_api_error_message`, `api_error_text`. `fact_messages` now carries `entry_id` / `message_id` / `session_id` as degenerate dimensions.
- **fact_tool_uses + fact_tool_results.** Replaces legacy `fact_tool_calls`. Both tables carry `entry_id` / `message_id` / `session_id` / `tool_use_id` degenerate dims.
- **HTML rendering of non-message JSONL entries.** Phase 1: parser preserves the 9 non-message entry types (`system`, `attachment`, `permission-mode`, `custom-title`, `agent-name`, `last-prompt`, `file-history-snapshot`, `queue-operation`, `pr-link`, `summary`), renderer dispatches with styled banners for the high-signal ones (permission-mode, queued prompts, turn duration, stop-hook summary, hook duration, diagnostics) and a collapsed `<details>` fallback for everything else so nothing captured is silently invisible. Adds `redacted_thinking`, `server_tool_use`, `web_search_tool_result`, `mcp_tool_use`, `mcp_tool_result`, `code_execution_tool_result` content blocks. Progress entries still skipped inline (too high-volume; captured in `fact_progress_events`).
- **`dim_session` minimal enrichment on insert.** `project_key` FK, `first_timestamp`, `last_timestamp` are now populated by the orchestrator so `semantic_sessions` / `semantic_project_context` / `semantic_cost_analysis` return rows immediately instead of being empty.
- **`dim_file` + `fact_file_operations`**: per-tool-call file operations with operation_type classification. Derived from `fact_tool_uses` + `fact_tool_results`. Lineage block on the fact; `dim_file` stays catalog-shaped.
- **`bridge_session_file`**: M:N aggregate of `fact_file_operations` per (session, file). Read / write / edit counts + timestamp window.
- **`fact_diagnostics`**: flattens `fact_attachments.attachment_type='diagnostics'`, one row per individual LSP diagnostic.
- **`fact_plan_revisions`**: ExitPlanMode outcome classification via the structural `fact_tool_results.is_error` signal (R16 tri-state), with full-content approval-signature fallback when `is_error` is NULL. This was the original v0.15 driver.
- **`fact_agent_delegations`**: Task tool spawns + agent rollup metrics from the R1 structured `toolUseResult` capture. `parent_session_key` set to the delegating session; `agent_session_key` resolves via `dim_session.agent_id` when the subagent is also ETL'd.
- **`fact_errors`**: one row per `fact_tool_results.is_error=TRUE`, with `error_type` classified via DuckDB regex CASE mirroring `heuristics.classify_error_type`.
- **`fact_tool_chain_steps`**: per (session, tool_use, step_position) with prev/next tool keys for adjacency-pattern queries. Chain = single assistant message_id.
- **`dim_session_chain`**: slug-grouped chain aggregate.
- **`dim_prompt`**: imports Claude Code's prompt history JSONL. Idempotent natural key on (display_text || iso_timestamp).
- **`dim_session` heuristic enrichment**: intent / complexity / outcome / domain + first_user_message / last_assistant_message. Reads inputs from the v0.15 facts; runs zero-dep classifiers in Python.
- **`dim_session` subagent linkage**: detects subagent JSONL files by source-path shape, sets `is_agent` / `agent_id` / `parent_session_key`, reads optional `.meta.json` sidecar for `agent_type` / `agent_description`, then walks the parent chain to set `depth_level`.

### Removed
- **Legacy per-session ETL.** Deleted `schemas/star/etl.py` (1456 lines), `schemas/star/heuristics.py` (163), `schemas/star/extractors.py` (274), `schemas/star/history_etl.py` (87). The public API no longer exports `run_star_schema_etl` or `finalize_star_schema`; both were superseded by the v0.15 orchestrator.
- **Cross-session finalize step.** `finalize_star_schema(conn, history_path=None, ...)` and its six helpers (`_calculate_session_depths`, `_build_session_chains`, `_link_agent_delegations`, `_link_plan_revisions`, `_build_session_file_bridge`, `_rollup_agent_metrics`) are gone. The cross-session work they did is now done inline by the v0.15 orchestrator and the Phase D populators it dispatches.
- **Legacy tests.** Deleted `test_star_schema_etl.py`, `test_star_schema_advanced.py`, `test_star_schema_analytics.py`, `test_history.py`, `test_heuristics.py`. Surgical edits to `test_sanitize.py`, `test_json_export.py`, `test_star_schema_ddl.py` to remove dependence on the dropped modules.

### Not yet (deferred to 0.16)
- DAG-invariant fact tables (`fact_task_decomposition`, `fact_routing_decision`, `fact_execution_step`, `fact_pruning_event`, `fact_synthesis_result`, `fact_verification`).
- Granular content extracts: `fact_content_blocks`, `fact_code_blocks`, `fact_entity_mentions`.
- Optional: `fact_session_embeddings`, `fact_tool_input_params`.

## 0.14.0

### Added
- **`ccutils explore` harlequin shim**: `ccutils explore archive.duckdb` launches harlequin for interactive SQL exploration. Requires `uv pip install ccutils[explore]`. Shows install instructions if harlequin is missing.

### Fixed
- **XSS in HTML export**: `render_markdown_text()` now sanitizes output via `nh3` to strip `<script>`, event handlers (`onerror`), `<iframe>`, and other dangerous HTML that the Python `markdown` library passes through by default. Previously, malicious content in session JSONL files could execute JavaScript when exported HTML was opened in a browser.

### Removed
- **Data Explorer SPA**: Removed the browser-based data explorer (1,700-line vanilla JS SPA, `explorer/` directory, `docs/DATA_EXPLORER.md`). Replaced by the harlequin shim above.
- **`create_semantic_model()` and `meta_semantic_model` table**: Removed the schema metadata table and its generator (~200 lines in `semantic.py`, 12 tests). Only consumer was the deleted explorer SPA. Harlequin introspects DuckDB natively.

## 0.13.0

### Added
- **Actual API token usage**: Star schema now captures real token counts from Claude API usage data instead of relying solely on word-count heuristics
  - `fact_token_usage` table: per-response token breakdown (input, output, cache creation, cache read, ephemeral tiers, service_tier, speed)
  - `actual_input_tokens`, `actual_output_tokens`, `cache_read_tokens` columns on `fact_messages` for per-message actual tokens
  - `actual_input_tokens`, `actual_output_tokens`, `cache_creation_tokens`, `cache_read_tokens` aggregated on `fact_session_summary`
  - `semantic_token_usage` view for token analysis with model/project context
  - `semantic_cost_analysis` view with `cache_hit_rate_pct` calculation
- **Turn durations**: `fact_turn_durations` table captures actual turn processing time (`durationMs`, `messageCount`) from system entries
  - `total_turn_duration_ms` and `turn_count` on `fact_session_summary`
- **LSP diagnostics**: `fact_diagnostics` table captures code diagnostics (severity, source, code, message, line range) linked to `dim_file`
  - `total_diagnostics` on `fact_session_summary`
- **Stop events**: `fact_stop_events` table captures session/turn stop reasons, hook counts, and prevented continuations
  - `stop_count` and `prevented_continuations` on `fact_session_summary`
- **Session metadata**: `dim_session` now stores `entrypoint` (cli/web), `custom_title`, `permission_mode`, `agent_type`, `agent_description`
- **Agent type from .meta.json**: Subagent sessions read `agentType` and `description` from `.meta.json` sidecar files (e.g., "Explore", "Plan", "code-reviewer")
- **Prompt history**: `dim_prompt` table + `semantic_prompt_history` view ingests `~/.claude/history.jsonl` (5700+ prompts across 71 projects). Links to sessions via `sessionId`. Loaded via `load_history()` or `finalize_star_schema(history_path=...)`
- **Hook run counting**: `total_hook_runs` on `fact_session_summary`
- **New parser types**: `SessionSystemEntry`, `SessionAttachment`, `SessionMetaEntry` dataclasses + `iter_all_session_entries()` function that yields all JSONL entry types (backward-compatible -- existing `iter_session_entries()` unchanged)
- **History parser**: `parsers/history.py` with `HistoryEntry` dataclass and `iter_history_entries()`
- **`entrypoint` field**: `SessionMetaHeader` and `SessionEntry` now capture the `entrypoint` field from session entries

### Fixed
- **`agent_type` data mapping**: The `agent-name` JSONL entry contains the session title (same as `custom-title`), not the agent type. Now correctly maps to `custom_title` as fallback. `agent_type` is populated exclusively from `.meta.json` sidecar files

### Changed
- Star schema expanded from 22 tables + 10 views to 27 tables + 13 views
- Estimated token counts and actual token counts coexist -- old sessions without usage data get NULL for actual columns

## 0.12.0

### Changed
- **CLI simplification**: Opinionated defaults, removed dead weight, grouped options
  - `convert` command absorbed into `local` -- pass a file as positional arg to convert directly: `ccutils session.jsonl`
  - No file arg = interactive picker (previous `local` behavior). File arg = convert it (previous `convert` behavior)
  - URL input support removed (`curl url > file.jsonl && ccutils file.jsonl` instead)
  - `convert` still works as a hidden alias for backwards compatibility
  - Thinking blocks and subagents/agents now **included by default** -- use `--no-thinking`, `--no-subagents` (local), `--no-agents` (all) to opt out
  - Removed `--schema` flag (auto-inferred from `--format`)
  - Removed `--json`, `--output-auto`, `--repo` (from local), `--limit` flags
  - Merged `--embed` + `--embed-model` into single `--embed [MODEL]` flag
  - Options grouped into sections (Output, Selection, Content, Processing, Embeddings) via `click-option-group`

### Added
- `click-option-group` dependency for grouped CLI help output

## 0.11.0

### Added
- **`convert` command**: Renamed from `json`, now supports all output formats via `--format` (html, duckdb, duckdb-star, json, json-star) and `--schema` (simple, star) -- single entry point for converting JSON/JSONL files or URLs
- **Token estimation breakdown**: `total_thinking_tokens` and `total_tool_io_tokens` columns on `fact_session_summary` -- thinking blocks and tool I/O were previously uncounted
- **`estimated_tokens` column** on simple schema `sessions` table
- **Inclusive agent metric rollup**: `fact_session_summary` now carries `_incl_agents` columns (`total_estimated_tokens_incl_agents`, `total_tool_calls_incl_agents`, `total_errors_incl_agents`, `total_duration_incl_agents`) that aggregate metrics from all descendant subagent sessions. Bottom-up rollup runs during `finalize_star_schema()` using `dim_session.depth_level`. `fact_agent_delegations` also carries denormalized `agent_estimated_tokens`
- **Semantic view updates**: `semantic_sessions` and `semantic_project_context` expose `_incl_agents` columns; `semantic_agent_delegations` exposes `agent_estimated_tokens`
- **CLI test coverage**: New test files for 4 previously untested commands -- `test_convert_cmd.py`, `test_schema_cmd.py`, `test_import_cmd.py`, `test_web_cmd.py`

### Fixed
- **Orphan tool use preservation**: Tool calls interrupted before receiving a result (session killed mid-tool) are now stored in both simple and star schema DuckDB exports with NULL `output_text` and `result_message_id` -- previously silently dropped, creating asymmetry with JSON export which already included them
- **Token estimation accuracy**: Star schema ETL now counts thinking blocks and tool input/output in token estimates (previously only counted text blocks). Per-message `estimated_tokens` in `fact_messages` now includes all content types (thinking, tool I/O, text) for that message -- previously only counted text, making `SUM(estimated_tokens)` miss ~75% of tokens
- **URL project_name**: `convert` command now uses URL filename stem as `project_name` instead of temp directory name
- **CSS brace bug in import command**: Multi-session index used `.format()` which conflicted with CSS `{}` braces -- switched to f-string with doubled braces
- **Simple ETL duplication**: Extracted `_extract_session_core()` and `SimpleExtractionResult` dataclass to share ~200 lines of logic between DuckDB and JSON export paths

### Removed
- Dead `get_terminal_width` wrapper from `parsers/discovery.py`
- `json` CLI command (replaced by `convert`)

## 0.10.2

### Added
- **Project context views**: Two new semantic views for catching up on project state
  - `semantic_project_context`: sessions with first/last messages, intent, metrics -- ordered by recency
  - `semantic_project_files`: file activity aggregated by project with session count, read/write/edit totals
- **Session message columns**: `first_user_message` and `last_assistant_message` persisted on `dim_session` (truncated to 500 chars) -- previously extracted during ETL but discarded after heuristic classification
- **Date/time on all semantic views**: Every view now exposes a DATE field and time_of_day for filtering and sorting
  - `semantic_sessions`: `session_datetime`, `time_of_day`, `hour` from dim_time
  - `semantic_file_operations`: `full_date`, `time_of_day` from dim_date/dim_time
  - `semantic_session_chains`: `chain_start_date` derived from first_timestamp
  - `semantic_agent_delegations`: `delegation_date`, `time_of_day` from dim_date/dim_time
  - `semantic_file_evolution`: `first_seen_date`, `last_seen_date` derived from timestamps
  - `semantic_project_context`: `session_date`, `time_of_day` from dim_time
  - `semantic_project_files`: `last_touched_date` derived from timestamp
- **time_key on fact_session_summary**: Enables dim_time joins for session-level views

### Changed
- View count updated from 8 to 10 across all docs and docstrings
- Embedding pipeline docs updated to honestly describe current status (infrastructure for future semantic search, no built-in query consumer yet)

## 0.10.1

### Fixed
- **Master index HTML rendering**: `_generate_master_index()` now passes `total_projects`, `total_sessions`, `recent_date`, and `global_search_js` to the template -- previously rendered empty
- **Project index HTML rendering**: `_generate_project_index()` now passes `session_count` to the template
- **CSS class mismatch**: `.index-item-number` in 3 templates renamed to `.index-item-num` to match stylesheet
- **33 missing CSS definitions**: Added styles for `.file-tool-*`, `.edit-*`, `.tool-header`, `.tool-icon`, `.todo-header`, `.todo-items`, `.index-commit-*`, `.search-result-*`, `.search-modal`, `.disabled`, `.continuation`, `.commit-card-hash`, `.image-block`, `.date`
- **Docstring privacy**: Removed hardcoded username from `metadata.py` docstring
- **Star schema post-ETL wiring**: `local` command now runs post-ETL steps (session chains, agent delegations, file bridge, depth calculation) that were previously only called by the `all` command. New `finalize_star_schema()` public function.

### Changed
- **Score-based intent classification**: `classify_intent()` now counts keyword matches per intent and returns the one with the most hits (ties broken by priority order). Fixes compound messages like "implement new error handling" being misclassified as `bug_fix` instead of `feature`

### Removed
- Dead templates: `star_schema_dashboard.html`, `data_explorer.html` (never loaded by any Python code)
- Dead code: `entity_type_key` generation in `extractors.py` (unused since degenerate dimension switch)

### Internal
- Split `test_star_schema.py` (3382 lines, 38 classes) into 4 focused files: `_ddl`, `_etl`, `_analytics`, `_advanced`
- Shared fixtures extracted to `conftest.py`
- README.md rewritten for accuracy (22 tables, heuristic classification, all CLI options)
- Source docstrings updated from "25+ tables" to "22 tables + 8 views"
- STAR_SCHEMA.md intent section updated to document score-based matching

## 0.10.0

### Breaking Changes
- **Star schema rebuilt from 37 tables to 22 tables + 8 views**
  - 15 tables removed: stg_raw_messages, dim_message_type, dim_content_block_type, dim_error_type, dim_entity_type, dim_programming_language, dim_intent, dim_topic, dim_sentiment, dim_goal, dim_task, dim_attempt, fact_message_enrichment, fact_message_topics, fact_session_insights
  - LLM enrichment pipeline (`enrichment.py`) deleted -- required user-provided callbacks that nobody used
  - Removed exports: `run_llm_enrichment`, `run_session_insights_enrichment`, `run_goal_task_enrichment`
  - `_populate_reference_data()` removed -- pre-populated dimension rows were misleading
  - JSON export meta.json version changed from "1.0" to "2.0"

### Added
- **Heuristic classification** runs during ETL with zero external dependencies (no LLM, no API key)
  - `classify_intent()`: bug_fix, feature, refactor, debug, test, docs, review, explore (from first user message keywords)
  - `classify_complexity()`: trivial, simple, moderate, complex (from session metrics)
  - `classify_outcome()`: success, failure, unknown (from last assistant message + error rate)
  - `classify_domain()`: web, backend, data, devops, docs, mixed, unknown (from file extensions)
  - `classify_error_type()`: permission_denied, file_not_found, syntax_error, timeout, import_error, tool_error (from error text)
  - Results stored on `dim_session` (intent, complexity, outcome, domain) and `fact_errors` (error_type)
  - 39 tests for heuristic classifiers
- **Tool call duration tracking**: `duration_seconds` on `fact_tool_calls` (time between invoke and result)
- **Tool chain enhancements**: `next_tool_key` and `is_error` on `fact_tool_chain_steps`
- **Enhanced session summary**: `total_errors`, `unique_tools_used`, `unique_files_touched`, `max_conversation_depth`, `total_estimated_tokens` on `fact_session_summary`
- **Agent delegation metrics**: `agent_tool_calls`, `agent_errors`, `agent_duration_seconds` denormalized on `fact_agent_delegations`
- **File language detection**: `language` column on `dim_file` inferred from extension
- **Week of year**: `week_of_year` column on `dim_date`
- **New view**: `semantic_tool_patterns` -- common tool sequences with frequency and error rates

### Changed
- 6 low-cardinality dimension tables replaced with degenerate VARCHAR columns on fact tables (Kimball best practice)
- Embedding pipeline default changed from "summary" to "first_user_message" (summary depended on removed LLM enrichment)
- `dim_session` no longer has `goal_key`, `task_key`, `attempt_key` columns
- Star schema docs (`docs/STAR_SCHEMA.md`) fully rewritten

## 0.9.5

### Added
- **`--private` flag** for privacy-preserving exports: sanitizes absolute file paths in HTML, DuckDB, and JSON output
  - `PathSanitizer` class converts cwd-relative paths to relative, home-relative to `~/...`, leaves system paths unchanged
  - Applied at ETL time so all downstream consumers get clean data automatically
  - Available on all commands: `local`, `all`, `json`, `web`, `import`
  - 49 new tests (31 unit + 18 integration)

### Changed
- `_export_to_html` cleanup: eliminated temp-file round-trip by passing loglines directly to `generate_html(loglines=)`, reused `_group_loglines_by_session` helper, simplified `auto_open` logic, removed unused `metadata` binding and `json` import

## 0.9.4

### Removed
- **Gist upload feature**: Removed `--gist` option from `local`, `web`, and `json` commands; deleted `create_gist()`, `GistError`, `inject_gist_preview_js()`, gist preview JS, and ~310 lines of gist tests
- Backward-compat re-exports (`build_project_choices`, `build_session_choices`, etc.) from `parsers/__init__.py`

### Changed
- **Codebase cleanup round 2**: Eliminated ~845 additional lines across 5 phases
  - Static file loading uses `importlib.resources.files()` instead of `Path(__file__)` for wheel/zip compatibility
  - Star ETL `_extract_star_data()` decomposed: `BlockContext` dataclass + extracted `_handle_tool_use_block()` and `_handle_tool_result_block()` handlers
  - Import command DuckDB export now reuses `simple/etl.py:export_session_to_duckdb()` via new `iter_loglines()` adapter, replacing ~200 lines of duplicated insert logic

## 0.9.3

### Changed
- **Codebase cleanup**: Eliminated ~790 lines of duplicated/dead code across 7 phases
  - Deleted 170-line `generate_html_from_session_data` clone; `generate_html()` now accepts optional `loglines` param
  - Unified `_extract_text()` duplicate in `parsers/metadata.py` with `extract_text_from_content()` from `parsers/session.py`
  - Extracted ~232 lines of inline CSS/JS from `export/html.py` to `src/ccutils/static/` files
  - New shared JSONL parser (`parsers/jsonl_reader.py`) with `iter_session_entries()` generator replaces triple-parsed sessions in simple and star schema ETL
  - Decomposed `star/etl.py` with `StarExtractionResult` dataclass; `_load_dimensions`/`_load_facts` take structured result instead of 20+ positional args
  - Removed 6 deprecated wrapper functions from `parsers/discovery.py`; imports now go through `tui/` package
  - Extracted `handle_gist_upload()` and `maybe_open_browser()` helpers to `cli/utils.py`

## 0.9.2

### Added
- **Styled TUI package** (`src/ccutils/tui/`): New modular package for terminal UI with semantic coloring
  - `theme.py`: Color constants for prompt_toolkit (questionary) and Rich, with model-family sub-styles (opus=bold, sonnet=normal, haiku=italic magenta)
  - `formatters.py`: Pure formatting functions for relative dates, durations, project names, summaries, branch names, file sizes, message counts
  - `layout.py`: Terminal width detection and proportional column width calculation with `ColumnSpec` dataclass
  - `components.py`: Rich table renderers using ratio-based columns that expand to fill terminal width; summary column gets remaining space
  - `selection.py`: Questionary choice builders using `FormattedText` (list of `(style, text)` tuples) for per-segment coloring in checkboxes/selects
- Styled questionary chrome: blue pointer/highlight, green selected markers, dim instructions via `questionary_style()`
- Color-coded session labels: dates in yellow, project names in blue, models in magenta, counts in green, summaries in default

### Changed
- `local` command now uses styled choices and styled questionary chrome for both flat and two-phase selection modes
- `web` command session picker now uses styled choices with color-coded repo/date/title
- `import` command interactive picker now uses styled choices with color-coded date/count/name
- Project table title now shows total counts: "Projects (N found, M sessions)"
- Session table columns use `expand=True` with `ratio` so summaries fill remaining terminal width
- `discovery.py` refactored: display/selection functions replaced with thin wrappers delegating to `tui/` (~400 lines removed)
- All backward-compatible re-exports preserved in `parsers/__init__.py`

## 0.9.1

### Changed
- **Deterministic agent delegation linking**: `_link_agent_delegations` now uses `progress` records from JSONL for zero-ambiguity matching (confidence 1.0) instead of relying solely on timestamp proximity heuristics
  - New `stg_task_agent_map` staging table captures `tool_use_id` -> `agent_id` links from progress records during ETL
  - Falls back to timestamp-based heuristic matching (confidence 0.5-0.8) for older data without progress records
  - Multiple simultaneous Task delegations are now matched correctly

## 0.9

### Added
- **Session chains**: `dim_session_chain` groups sessions sharing the same `slug` into chains
  - `chain_key` added to `dim_session` for chain membership
  - `semantic_session_chains` view for chain-level analytics
  - Chains auto-built during batch export from shared slug values
- **Agent delegation tracking**: `fact_agent_delegations` links agent sessions to their parent's Task tool_use calls
  - Heuristic matching by timestamp proximity with confidence scoring
  - Captures task description, prompt, subagent_type from Task tool inputs
  - `semantic_agent_delegations` view joining parent/agent sessions with metrics
- **Session hierarchy**: Goal > Task > Attempt dimensional tables
  - `dim_goal`, `dim_task`, `dim_attempt` tables (populated via LLM enrichment)
  - `goal_key`, `task_key`, `attempt_key` soft FKs on `dim_session`
  - `run_goal_task_enrichment(conn, classify_func)` enrichment API
- **ColBERT embedding pipeline**: Semantic matching via PyLate (optional dependency)
  - `EmbeddingPipeline` class with lazy model loading (`mxbai-edge-colbert-v0-32m`)
  - `embed_sessions()`: Embed session summaries into `fact_session_embeddings`
  - `match_delegations()`: Re-score agent delegation matches using semantic similarity
  - `cluster_sessions()`: K-means clustering with auto task assignment
  - `--embed` and `--embed-model` CLI flags for batch export
- **Cross-session file bridge**: `bridge_session_file` aggregates file operations across sessions
  - Per-file operation counts (read/write/edit) by session
  - `semantic_file_evolution` view for files touched by multiple sessions
- **Session slug storage**: `slug` column in `dim_session` preserves chain resume identifiers
- **Agent depth calculation**: `depth_level` correctly calculated for nested agent hierarchies
  - Iterative batch calculation handles arbitrarily deep nesting
  - Single-session ETL attempts parent lookup during insert

### Dependencies
- Added `pylate` as optional dependency: `pip install ccutils[colbert]`

## 0.8

### Added
- **Two-phase session selection UI**: Redesigned `local` command with project-first navigation
  - Phase 1: Pick project(s) from a rich summary table showing session counts, models, branches, last active date
  - Phase 2: Pick session(s) within selected projects with detailed metadata (model, branch, duration, message count)
  - Automatic skip of phase 1 when only one project matches (or when using `-p` filter)
  - `--flat` flag preserves old single-list behavior
  - `--expand-chains` flag shows individual sessions in resumed chains
- **Rich metadata extraction**: New `SessionMetadata` dataclass and extraction pipeline
  - `extract_rich_metadata()`: Single-pass extraction of cwd, model, branch, slug, duration, message counts
  - `get_meaningful_summary()`: Smarter summary extraction that skips interrupted/error/XML messages
  - `shorten_model_name()`: Human-friendly model names (`claude-opus-4-6` -> `opus-4.6`)
  - `format_duration()`: Human-readable duration (`45m`, `1h 5m`)
  - `derive_project_name()`: Derives project name from `cwd` field (actual directory name, not encoded path)
- **Rich terminal tables**: `rich` library for colorized project and session tables
  - `print_project_table()`: Summarizes projects with session counts, models, branches
  - `print_session_table()`: Shows session details with relative dates, model, branch, duration
- **New discovery functions**: `find_local_sessions_rich()`, `group_by_project()`, `build_project_choices()`, `build_session_choices_for_projects()`

### Changed
- Default `local` command now uses two-phase selection (projects then sessions)
- Project names derived from `cwd` metadata field when available (more accurate than folder name parsing)
- Session summaries no longer show `[Request interrupted...]` or XML system prompts

### Dependencies
- Added `rich` for terminal formatting

## 0.7

### Added
- **Repo display and filtering in `web` command**: Shows which GitHub repo each session belongs to (adapted from upstream simonw/claude-code-transcripts v0.6)
  - `extract_repo_from_session()`: Extracts repo from API session metadata (outcomes or sources URL)
  - `enrich_sessions_with_repos()`: Adds `repo` key to session list data
  - `filter_sessions_by_repo()`: Client-side filtering by repo name
  - Session picker now shows `{repo}  {date}  {title}` instead of `{session_id}  {date}  {title}`
  - `--repo` flag now filters session list in addition to setting default for commit links
- **Un-nested tool parameters in star schema**: Extract common tool parameters from JSON blobs for easier querying
  - New columns in `fact_tool_calls`: `file_path`, `command`, `pattern`, `query_text`
  - New `fact_tool_input_params` table: Key-value pairs for all tool input parameters
  - Updated `semantic_tool_calls` view to include extracted columns
  - Supports queries like `SELECT * FROM fact_tool_calls WHERE file_path IS NOT NULL`
- **Star schema support in `all` command**: Full star schema support for batch exports
  - New format options: `--format duckdb-star`, `--format json-star`
  - Uses 25+ dimensional tables for richer analytics
  - Progress reporting shows row counts, DB size, and processing rate
- **Performance options for batch processing**:
  - `-j/--jobs N`: Parallel workers for processing (default: 1)
  - `--batch-size N`: Sessions per transaction batch (default: 10)
  - Progress callback now includes stats (rows_inserted, db_size_mb, rate)
- **Enhanced progress reporting**: Shows rows processed, storage size, and sessions/sec rate
- **Claude.ai account export import**: New `import` command to convert Claude.ai account exports (from Settings > Privacy)
  - Supports all existing output formats: HTML, DuckDB
  - Lists conversations: `ccutils import ./export --list`
  - Interactive selection: `ccutils import ./export --interactive`
  - Filter by conversation UUID: `ccutils import ./export -c <uuid>`
  - Preserves thinking blocks and tool calls
  - New parser functions: `parse_claude_ai_export()`, `convert_conversation_to_loglines()`
- **JSON export format**: Export sessions to JSON in addition to HTML and DuckDB
  - `--format json`: Simple schema (sessions, messages, tool_calls, thinking) in single JSON file
  - `--format json-star`: Star schema exported as directory structure (meta.json + dimensions/*.json + facts/*.json)
  - New `--schema` option to explicitly set schema type (`simple` or `star`)
  - Backwards-compatible: compound format names (`duckdb-star`, `json-star`) still work
  - New functions: `resolve_schema_format()`, `export_sessions_to_json()`, `export_star_schema_to_json()`
- **Multi-select for local command**: Select multiple sessions using SPACE, confirm with ENTER
- **DuckDB export from local command**: New `--format` option supports `html`, `duckdb`, `duckdb-star`, `json`, or `json-star`
- **Subagent support**: New `--include-subagents` flag auto-includes related agent sessions (recursive)
- **Agent metadata in DuckDB**: Sessions and messages now track agent relationships
  - Sessions table: `is_agent`, `agent_id`, `parent_session_id`, `depth_level` columns
  - Messages table: `is_sidechain` column
  - Star schema: Same columns in `dim_session` and `fact_messages`
- New functions: `extract_session_metadata()`, `find_agent_sessions()` for agent discovery

### Changed
- Expanded README documentation for star schema analytics with comparison table, quick start code, and overview of dimensions/facts
- `local` command now uses `questionary.checkbox()` for multi-select (was single-select)

### Added (continued)
- Full-text search across the entire HTML archive generated by the `all` command
  - Search index (`search-index.js`) is generated alongside HTML files
  - In-browser JavaScript search with snippet highlighting
  - Search works offline and on `file://` protocol (unlike existing per-session search)
  - Results show project, type, timestamp, and link directly to the matching content
  - Mobile-friendly responsive design
- New CLI option `--no-search-index` to skip search index generation for faster/smaller output
- New functions: `extract_searchable_content()`, `extract_snippet()` for search indexing
- DuckDB export for structured analytics on transcript data
  - Export sessions, messages, tool calls, and thinking blocks to a single DuckDB database
  - Query your transcripts with SQL for analytics and insights
  - New CLI option `--format` to choose output format: `html` (default), `duckdb`, or `both`
  - New CLI option `--include-thinking` to include thinking blocks in DuckDB export (opt-in, can be large)
  - New functions: `create_duckdb_schema()`, `export_session_to_duckdb()`, `generate_duckdb_archive()`
