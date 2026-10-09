---
name: etl-dev
description: Extend or modify the ccutils ETL pipeline and star schema — adding a fact table, dimension or populator, adding/renaming columns on shipped tables, adding or versioning facets (Tier 1 SQL or Tier 2 LLM), touching the parser/staging/orchestrator, or cutting a release. Use for any change under src/ccutils/etl/, src/ccutils/schemas/, or src/ccutils/parsers/. Routes to the exact workflow checklist so shipped-broken classes of bugs (lineage drift, silent migrations, subagent misattribution) don't recur.
---

# ccutils ETL development

Always TDD: failing test → watch it fail → make it pass. Full suite:
`uv run pytest tests/ --confcutdir=tests`. Commit test + implementation +
docs together.

## Route by task

| Task | Read |
|---|---|
| New fact table / populator | `references/new-fact-table.md` (full checklist) |
| New dimension table | `references/new-dimension.md` |
| Add/rename column on a shipped table; version bump/release | `references/migrations-and-versioning.md` |
| Add/change a facet (F01+), bump a prompt version | `references/facets.md` |
| Ingest a source that is NOT per-session (a global file, a per-repo directory) | `etl/dim_memory.py::run_memory_import` is the worked example: an `EtlRun` with `run_kind="global_source"`, a step whose counts come from the work, `run.fail()` on error. Call it from `etl/global_sources.py::run_global_sources` so every entry point runs it |
| Query-shape questions while developing | the `query-warehouse` skill's references |
| Table/view column ground truth | `docs/STAR_SCHEMA.md` (grep the table name) |
| Where a new table, view or populator belongs | `docs/ETL_ARCHITECTURE.md` (the three rules) |
| What is decided, open, or next | `docs/ROADMAP.md`. Read the decided items before designing, not after |
| Facet pipeline design/status | `docs/FACET_CLUSTER_PIPELINE.md` |

## The rules are in `CLAUDE.md`

The contracts that hold everywhere (every populator through `lineage_upsert`,
one row per declared natural key, agent identity from the file, no
migrations, global sources recorded as runs, and the rest) live in
`CLAUDE.md`, which is loaded in every session. They are not repeated here:
a second copy went stale the first time one of them changed.
