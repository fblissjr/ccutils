---
name: etl-dev
description: Extend or modify the ccutils ETL pipeline and star schema — adding a fact table or populator, adding/renaming columns on shipped tables, adding or versioning facets (Tier 1 SQL or Tier 2 LLM), touching the parser/staging/orchestrator, or cutting a release. Use for any change under src/ccutils/etl/, src/ccutils/schemas/, or src/ccutils/parsers/. Routes to the exact workflow checklist so shipped-broken classes of bugs (lineage drift, silent migrations, subagent misattribution) don't recur.
---

# ccutils ETL development

Always TDD: failing test → watch it fail → make it pass. Full suite:
`uv run pytest tests/ --confcutdir=tests` (green + 1 skipped live-API test =
healthy). Commit test + implementation + docs together.

## Route by task

| Task | Read |
|---|---|
| New fact table / populator | `references/new-fact-table.md` (full checklist) |
| Add/rename column on a shipped table; version bump/release | `references/migrations-and-versioning.md` |
| Add/change a facet (F01+), bump a prompt version | `references/facets.md` |
| Query-shape questions while developing | the `query-warehouse` skill's references |
| Table/view column ground truth | `docs/STAR_SCHEMA.md` (grep the table name) |
| Facet pipeline design/status | `docs/FACET_CLUSTER_PIPELINE.md` |

## Contracts that hold everywhere (violations have shipped broken before)

- **Every fact populator goes through `lineage_upsert`** (`etl/upsert.py`) —
  never hand-roll INSERT/UPDATE/soft-delete. Its parameter pitfalls are in
  `references/new-fact-table.md`.
- **Populators reading permanent facts (not staging) must scope inbound to
  staged sessions**: `AND session_id IN (SELECT DISTINCT session_id FROM
  stg_log_entries ...)` — otherwise every run rescans the whole warehouse.
- **Subagent identity comes from the FILE** (`agent-<id>` stem), stamped at
  Tier 1 (`parsers/parquet_writer.py`) and re-enforced at staging load. Never
  key anything on `$.sessionId` inside agent JSONL — every line carries the
  PARENT's id. Synthetic fixtures model this with `entry_session_id`
  (`tests/helpers_ccutils.py::write_minimal_session`).
- **The project-boundary rule lives in exactly two mirrored places**:
  `project_dir_sql` (`etl/utils.py`) and the walk-up loop in
  `parsers/discovery.py::find_all_sessions`.
  `tests/test_all.py::TestProjectRuleEquivalence` fails on drift. Never add a
  third copy.
- **Closures wrapping `run_v15_etl` list every swallowed kwarg explicitly**,
  never `**kwargs` — a shim once silently dropped `--private`.
- Orchestration lives in `etl/orchestrator.py::run_v15_etl`; DDL single source
  of truth is `schemas/star/schema.py::create_star_schema()`;
  `fact_session_summary` always populates LAST.
- Parser models (`parsers/models.py`): `pydantic.alias_generators.to_camel`
  breaks on all-caps abbreviations — use explicit `Field(alias=...)`.
  Stdlib `json` is the convention; do not introduce orjson.
- Run metadata is three grains (batch / run / step) with counts derived from
  children at `complete()` — never tally counts by hand in a caller. Details:
  `docs/STAR_SCHEMA.md` "Run metadata".

## Testing conventions

- One `tests/test_<fact>_v15.py` per populator; shared fixtures +
  `write_minimal_session` in `tests/conftest.py`.
- Exit-code-only CLI flag tests are insufficient — assert the flag's actual
  effect (e.g. absent thinking), not just acceptance.
- Batch/CLI monkeypatching gotchas and stderr-robust helpers:
  `tests/test_cli_llm_facets.py` (`_combined_output`, `importlib.import_module`
  for shadowed submodules).
- Never use real usernames or personal paths in fixtures/docstrings — use the
  generic placeholder style the existing fixtures use.
