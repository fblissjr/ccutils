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
| Ingest a source that is NOT per-session (a global file, a per-repo directory) | `etl/dim_memory.py` is the worked example -- wrap it as a recorded run, see the contract below |
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
- **Global (non-per-session) sources still go through the run-metadata
  system.** Living outside `run_v15_etl` is a statement about GRAIN, not a
  licence to skip lineage. `import_memories` first shipped as a bare call
  inside `except Exception: pass` — no run row, no step row, no `etl_run_id`,
  and a failure that left a warehouse indistinguishable from one where the
  source did not exist. Wrap it like `run_memory_import` /
  `run_post_session_reconciliation`: `EtlRun.start(run_kind=...)`, a
  `run.step()` whose counts come from the work done, `run.fail()` on error.
  Re-raise only when the output is load-bearing (reconciliation does; memory
  is additive and records instead). `dim_prompt` is still un-wrapped.
- **A Type 2 dimension does not inherit the fact lineage block wholesale, and
  must not use `lineage_upsert`** (Type 1 upsert-with-soft-delete fights Type
  2). Take `created_at`/`created_by_version_key`/`etl_run_id`/`record_source`;
  omit `is_deleted`/`deleted_at` (the `is_current`/`valid_to` pair IS the
  deletion mechanism — two of them disagree the first time a row is retired)
  and `hash_diff` (the content hash is the change detector). Version identity
  is `(entity_id, version_num)`, never `(entity_id, content_hash)`: reverted
  content repeats its hash and the revert would silently vanish.
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
