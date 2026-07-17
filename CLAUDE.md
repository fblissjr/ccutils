<!-- path-privacy: skip-file -- references universal Claude data layout paths (not personal) -->
# Start Here

Uses uv. Run tests: `uv run pytest tests/ --confcutdir=tests`. Run the dev CLI: `uv run ccutils --help`.

Always practice TDD: write a failing test, watch it fail, then make it pass. Commit early and often; bundle test + implementation + docs per commit.

Reference docs (read these instead of expanding this file):

- `README.md` -- CLI commands, export formats, defaults.
- `docs/STAR_SCHEMA.md` -- every table/view, lineage convention, run-metadata grains, populator order.
- `docs/FACET_CLUSTER_PIPELINE.md` -- facet pipeline design + status.

## Architecture in one paragraph

Four tiers: Claude Code JSONL -> Parquet lake (`parsers/parquet_writer.py`, re-derivable cache) -> `stg_log_entries` staging (`etl/staging.py`, cleared after every run) -> star-schema facts (`etl/*.py`, one populator per fact). Entry point: `run_v15_etl(conn, session_path, ...)` in `etl/orchestrator.py`; DDL single source of truth: `create_star_schema()` in `schemas/star/schema.py`. Batch drivers: `export/duckdb_archive.py` (duckdb + json) and the render-only html/markdown exporters. Run metadata is three grains -- `fact_etl_batch_runs` (CLI invocation, `BatchRun` context manager) / `fact_etl_runs` (session) / `fact_etl_steps` (DAG node, recorded by `lineage_upsert`) -- see STAR_SCHEMA.md "Run metadata".

## Rules that exist because something shipped broken

- **Every fact populator goes through `lineage_upsert` (`etl/upsert.py`).** Pitfalls: `payload_cols` must NOT include `session_key`; target table needs `date_key`+`time_key`; aggregate facts pass `timestamp_col=`; shared tables (two populators, e.g. `fact_session_facets`) MUST pass `soft_delete_scope_sql` or one populator soft-deletes the other's rows.
- **Populators reading PERMANENT facts (not staging) must scope inbound to staged sessions** (`AND session_id IN (SELECT DISTINCT session_id FROM stg_log_entries ...)`) or they rescan the whole persistent warehouse every call.
- **Adding a column to a table that shipped in a release: append to `_COLUMN_MIGRATIONS` in `schemas/star/schema.py`.** `CREATE TABLE IF NOT EXISTS` never widens. Migrations run after CREATEs, before views. After renames, grep the `semantic_` views and the column-list assertions in `tests/test_star_schema_ddl.py`.
- **Closures wrapping `run_v15_etl` list every swallowed kwarg explicitly, never `**kwargs`** -- a `**_legacy_kwargs` shim once silently dropped `--private`.
- **`_PROGRESS_TABLES` (`export/duckdb_archive.py`) lists every DATA fact `run_v15_etl` populates, excluding the audit tables (`fact_etl_runs/batch_runs/steps`).** Stale entries undercount the progress display; audit rows would inflate it.
- **The project-boundary rule lives in exactly two mirrored places**: `project_dir_sql` (`etl/utils.py`, SQL) and the walk-up loop in `parsers/discovery.py::find_all_sessions`. `tests/test_all.py::TestProjectRuleEquivalence` fails on drift. Never add a third copy -- eight drifted copies caused the "subagents" mis-attribution.
- **Subagent JSONL contract: agent transcript entries carry the PARENT's `sessionId` on every line.** Identity comes from the FILE (`agent-<id>` stem), stamped at Tier 1 (`parquet_writer`) and re-enforced at staging load. Never key anything on an agent file's embedded sessionId; synthetic fixtures must use `entry_session_id` (see `tests/conftest.py::write_minimal_session`) to model the real contract. Layout: `<project>/<parent-uuid>/subagents/agent-<id>.jsonl` + optional `.meta.json` sidecar (agentType/description). Pre-2026 short agent ids can collide across parents (accepted limitation); agents whose parent transcript was pruned keep `depth_level = 0` (honest -- no root to walk to).
- **Facet catalog is the single source of truth** (`etl/facets/catalog.py::FACET_SPECS`); `schemas/star` imports FROM it at DDL seed time, never the reverse (import cycle at CLI startup). `dim_facet_type` uses `CREATE TABLE IF NOT EXISTS` + `INSERT ... ON CONFLICT DO NOTHING` so historical prompt_version rows survive -- same pattern for any future history-retaining dim. Tier 2 credentials resolve only via `cli/utils.py::build_facet_extractor_or_exit`.
- **`--private` is best-effort on render formats (html/markdown), NOT a sharing guarantee, and NOT wired into the duckdb/json ETL** (loud `UsageError` there). It fails loud when cwd is unresolvable instead of no-opping -- the silent-privacy-no-op class shipped three times; never reintroduce it. Comprehensive channel-walking: `internal/plans/private_hardening.md`. `--no-thinking` IS wired through the facts; raw `message_json` never survives staging regardless; the Parquet lake intentionally retains everything (delete post-run if unwanted).
- **Exit-code-only CLI flag tests are insufficient** -- assert the flag's actual effect (sanitized paths, absent thinking), not just acceptance.

## Testing

- Full suite green + 1 skipped live-API test is the healthy state.
- v0.15 facts: one `tests/test_<fact>_v15.py` per populator. Shared fixtures + `write_minimal_session` in `tests/conftest.py`.
- Batch/CLI monkeypatching gotchas and stderr-robust assertion helpers: see `tests/test_cli_llm_facets.py` (`_combined_output`, `importlib.import_module` for shadowed submodules).
- Facet-fixture ids for hypothetical facet types use F90+ (`tests/test_fact_session_facets_v15.py::two_f90_versions`).
- HTML snapshot tests need `--snapshot-update` after ANY change to `transcript.css` / `macros.html` / `base.html`; CSS classes used in templates MUST exist in `transcript.css` (Jinja2 won't warn); template variables render empty, not error -- assert on rendered output.
- Never use real usernames or personal paths in docstrings or fixtures -- always the generic invented placeholder style the existing fixtures use.
- Live-API smoke after `AnthropicFacetExtractor` changes (pennies):
  `ANTHROPIC_API_KEY=$(security find-generic-password -s ccutils-anthropic -a $USER -w) uv run pytest tests/test_populate_tier2_facets.py::TestLiveApiSmoke -v`

## DuckDB / parsing idioms

- `json_extract(j, '$.p[*].f')::JSON[]` for lists; `LATERAL unnest(...)` when you need block + index; `json_type()` to gate string-vs-array `$.content`; `CAST(json_extract(...) AS VARCHAR)` for raw JSON (`json_extract_string` mangles arrays).
- DuckDB DML via `conn.execute(...)` returns the affected-row count as a one-row result -- that is how step counts stay real.
- `pydantic.alias_generators.to_camel` breaks on all-caps abbreviations (`sourceToolUseID` etc.) -- explicit `Field(alias=...)`; grep `alias=` in `parsers/models.py`.
- Stdlib `json` is the project convention; do not migrate to orjson.

## Workflow recipes

- **New fact table**: failing tests (DDL + lineage cols + behavior + idempotency) -> DDL with the standard lineage block -> `populate_fact_<x>` delegating to `lineage_upsert` -> wire into `run_v15_etl` in dependency order (`fact_session_summary` stays LAST) -> add to `_PROGRESS_TABLES` -> CHANGELOG + STAR_SCHEMA.md.
- **Removing a feature**: grep imports/`__all__`/CLI registrations/tests; delete; update CHANGELOG, README, `_PROGRESS_TABLES`; grep help text for stale counts.
- **Versioning**: version in `pyproject.toml`, sync with `CHANGELOG.md` (`[Unreleased]` promotes on release); tag `v0.X.0`. Semver, no major bumps without permission.
