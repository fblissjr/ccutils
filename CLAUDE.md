# Start Here

Uses uv. Run tests: `uv run pytest tests/ --confcutdir=tests`. Run the dev CLI: `uv run ccutils --help`.

Always practice TDD: write a failing test, watch it fail, then make it pass. Commit early and often; bundle test + implementation + docs per commit.

Reference docs (read these instead of expanding this file):

- `README.md` -- CLI commands, export formats, defaults.
- `docs/STAR_SCHEMA.md` -- every table/view, lineage convention, run-metadata grains, populator order.
- `docs/FACET_CLUSTER_PIPELINE.md` -- facet pipeline design + status.
- `.claude/skills/` -- task-scoped skills (`etl-dev`, `query-warehouse`, `render-exports`, `new-fact`, `new-dimension`, `test-schema`, `release`) with on-demand reference files; they load the detail above progressively, so prefer triggering the matching skill over re-reading the docs wholesale.

## Architecture in one paragraph

Four tiers: Claude Code JSONL -> Parquet lake (`parsers/parquet_writer.py`, re-derivable cache) -> `stg_log_entries` staging (`etl/staging.py`, cleared after every run) -> star-schema facts (`etl/*.py`, one populator per fact). Entry point: `run_v15_etl(conn, session_path, ...)` in `etl/orchestrator.py`; DDL single source of truth: `create_star_schema()` in `schemas/star/schema.py`. Batch drivers: `export/duckdb_archive.py` (duckdb + json) and the render-only html/markdown exporters. Run metadata is three grains -- `fact_etl_batch_runs` (CLI invocation, `BatchRun` context manager) / `fact_etl_runs` (session) / `fact_etl_steps` (DAG node, recorded by `lineage_upsert`) -- see STAR_SCHEMA.md "Run metadata".

## Rules that exist because something shipped broken

- **Every fact populator goes through `lineage_upsert` (`etl/upsert.py`).** Pitfalls: `payload_cols` must NOT include `session_key`; target table needs `date_key`+`time_key`; aggregate facts pass `timestamp_col=`; shared tables (two populators, e.g. `fact_session_facets`) MUST pass `soft_delete_scope_sql` or one populator soft-deletes the other's rows.
- **A populator's projection MUST emit one row per its declared `natural_key`; `lineage_upsert` RAISES otherwise.** A duplicate key means the projection does not produce the grain it declares -- fix the projection (see the `QUALIFY` in `_PROJECT_USES_SQL` / `_PROJECT_RESULTS_SQL`), never collapse inside the shared helper: only the populator knows whether collapsing is safe and which row should survive. A generic collapse there once applied one fact's judgment to all 13 and was invisible in the step counts. `NATURAL_KEYS` (`schemas/star/schema.py`) is the single source of truth for table -> key, guarded by a drift test; `create_star_schema` repairs pre-existing violations on open, because the assertion alone bricks warehouses built before it.
- **Populators reading PERMANENT facts (not staging) must scope inbound to staged sessions** (`AND session_id IN (SELECT DISTINCT session_id FROM stg_log_entries ...)`) or they rescan the whole persistent warehouse every call.
- **Adding a column to a table that shipped in a release: append to `_COLUMN_MIGRATIONS` in `schemas/star/schema.py`.** `CREATE TABLE IF NOT EXISTS` never widens. Migrations run after CREATEs, before views. After renames, grep the `semantic_` views and the column-list assertions in `tests/test_star_schema_ddl.py`.
- **Closures wrapping `run_v15_etl` list every swallowed kwarg explicitly, never `**kwargs`** -- a `**_legacy_kwargs` shim once silently dropped `--private`.
- **`_PROGRESS_TABLES` (`export/duckdb_archive.py`) lists every DATA fact `run_v15_etl` populates, excluding the audit tables (`fact_etl_runs/batch_runs/steps`).** Stale entries undercount the progress display; audit rows would inflate it.
- **The project-boundary rule lives in exactly two mirrored places**: `project_dir_sql` (`etl/utils.py`, SQL) and the walk-up loop in `parsers/discovery.py::find_all_sessions`. `tests/test_all.py::TestProjectRuleEquivalence` fails on drift. Never add a third copy -- eight drifted copies caused the "subagents" mis-attribution.
- **Subagent JSONL contract: agent transcript entries carry the PARENT's `sessionId` on every line.** Identity comes from the FILE (`agent-<id>` stem), stamped at Tier 1 (`parquet_writer`) and re-enforced at staging load. Never key anything on an agent file's embedded sessionId; synthetic fixtures must use `entry_session_id` (see `tests/helpers_ccutils.py::write_minimal_session`) to model the real contract. Layout: `<project>/<parent-uuid>/subagents/agent-<id>.jsonl` + optional `.meta.json` sidecar (agentType/description). Pre-2026 short agent ids can collide across parents (accepted limitation); agents whose parent transcript was pruned keep `depth_level = 0` (honest -- no root to walk to).
- **Facet catalog is the single source of truth** (`etl/facets/catalog.py::FACET_SPECS`); `schemas/star` imports FROM it at DDL seed time, never the reverse (import cycle at CLI startup). `dim_facet_type` uses `CREATE TABLE IF NOT EXISTS` + `INSERT ... ON CONFLICT DO NOTHING` so historical prompt_version rows survive -- same pattern for any future history-retaining dim. Tier 2 credentials resolve only via `cli/utils.py::build_facet_extractor_or_exit`.
- **`--private` is best-effort on render formats (html/markdown), NOT a sharing guarantee, and NOT wired into the duckdb/json ETL** (loud `UsageError` there). It fails loud when cwd is unresolvable instead of no-opping -- the silent-privacy-no-op class shipped three times; never reintroduce it. Comprehensive channel-walking: `internal/plans/private_hardening.md`. **`--no-thinking` is enforced separately on EVERY surface** -- ETL (`etl/orchestrator.py`, staging cleared), markdown (`export/markdown.py`, per-block skip), HTML (`export/html.py::_strip_thinking_blocks`, filters loglines before any consumer sees them), Claude.ai import (`parsers/claude_ai.py`, filtered at parse). It shipped **ignored entirely by HTML** -- flag accepted, exit 0, output byte-identical -- while a test file named after the flag stayed green because it covered the ETL tier. Adding a render surface means wiring it again and asserting the effect. Raw `message_json` never survives staging regardless; the Parquet lake intentionally retains everything (delete post-run if unwanted).
- **Exit-code-only CLI flag tests are insufficient** -- assert the flag's actual effect (sanitized paths, absent thinking), not just acceptance.
- **HTML export security is load-bearing -- do NOT remove without understanding the XSS implications.** `render_markdown_text` sanitizes via `nh3.clean(raw, attributes={"code": {"class"}})` (the carve-out keeps fenced-code highlighting); Jinja2 runs `autoescape=True` and every `|safe` in the macros is safe ONLY because content is pre-sanitized (nh3) or pre-escaped (`html.escape`); every document ships a strict CSP built by `export/html.py::_build_csp` -- `default-src 'none'`, inline `<style>`/`<script>` blocks pinned by sha256, never `unsafe-inline`/`unsafe-hashes` (a hash covers a BLOCK; `style=`/`on*=` ATTRIBUTES stay forbidden everywhere -- `TestNoInlineConstructsForCsp` guards this). The hash must be computed over the exact emitted bytes: Jinja autoescape once corrupted an inline script AND its hash silently. Transcript content is untrusted input.
- `filter.js` is a Jinja2 TEMPLATE rendered via `_jinja_env`, not a static file; `static/transcript.js` is read as text and inlined. Both are hashed into the CSP.
- **Never derive identity from `$.sessionId` in `raw_json`** -- raw payloads stay byte-faithful, so agent files' raw entries carry the PARENT's sessionId. Identity is the `session_id` COLUMN (stamped at Tier 1).

## Testing

- Full suite green + 1 skipped live-API test is the healthy state.
- v0.15 facts: one `tests/test_<fact>_v15.py` per populator. Shared fixtures in `tests/conftest.py`; `write_minimal_session` / `make_minimal_session_lines` in `tests/helpers_ccutils.py`.
- Batch/CLI monkeypatching gotchas and stderr-robust assertion helpers: see `tests/test_cli_llm_facets.py` (`_combined_output`, `importlib.import_module` for shadowed submodules).
- Facet-fixture ids for hypothetical facet types use F90+ (`tests/test_fact_session_facets_v15.py::two_f90_versions`).
- HTML snapshot tests (syrupy) were removed with C1/C2 -- coverage is explicit assertions and invariants in `test_generate_html.py` / `test_html_css_coverage.py`. CSS classes used in templates MUST exist in `transcript.css` (Jinja2 won't warn); template variables render empty, not error -- assert on rendered output.
- Never use real usernames or personal paths in docstrings or fixtures -- always the generic invented placeholder style the existing fixtures use.
- Live-API smoke after `AnthropicFacetExtractor` changes (pennies):
  `ANTHROPIC_API_KEY=$(security find-generic-password -s ccutils-anthropic -a $USER -w) uv run pytest tests/test_populate_tier2_facets.py::TestLiveApiSmoke -v`

## DuckDB / parsing idioms

- `json_extract(j, '$.p[*].f')::JSON[]` for lists; `LATERAL unnest(...)` when you need block + index; `json_type()` to gate string-vs-array `$.content`; `CAST(json_extract(...) AS VARCHAR)` for raw JSON (`json_extract_string` mangles arrays).
- DuckDB DML via `conn.execute(...)` returns the affected-row count as a one-row result -- that is how step counts stay real.
- `pydantic.alias_generators.to_camel` breaks on all-caps abbreviations (`sourceToolUseID` etc.) -- explicit `Field(alias=...)`; grep `alias=` in `parsers/models.py`.
- Stdlib `json` is the project convention; do not migrate to orjson.

## Workflow recipes

- **New fact table**: failing tests (DDL + lineage cols + behavior + idempotency) -> DDL with the standard lineage block -> `populate_fact_<x>` delegating to `lineage_upsert` (its projection MUST emit one row per declared `natural_key`; add the table to `NATURAL_KEYS`) -> wire into `run_v15_etl` in dependency order (`fact_session_summary` stays LAST) -> add to `_PROGRESS_TABLES` -> CHANGELOG + STAR_SCHEMA.md.
- **Removing a feature**: grep imports/`__all__`/CLI registrations/tests; delete; update CHANGELOG, README, `_PROGRESS_TABLES`; grep help text for stale counts.
- **Versioning**: version in `pyproject.toml`, sync with `CHANGELOG.md` (`[Unreleased]` promotes on release) AND `PARSER_VERSION` in `src/ccutils/_version.py` (it stamps every lineage row -- a stale value makes rows from different contracts indistinguishable); tag `v0.X.0`. Semver, no major bumps without permission.
