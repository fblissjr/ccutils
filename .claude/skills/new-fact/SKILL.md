---
name: new-fact
description: Guided workflow for adding a new fact table to the star schema
---

Follow this workflow to add a new fact table. Practice TDD throughout. Every
v0.15 fact -- including cross-session aggregates like `bridge_session_file`
and `fact_agent_delegations` -- is a per-session populator wired into
`run_v15_etl`; there is no separate post-ETL pass (no `finalize_star_schema`,
no `_extract_star_data`/`StarExtractionResult`/`_load_facts` -- that's
pre-v0.15 architecture and no longer exists in this codebase).

## Step 1: Design the fact

Ask the user:
- What event or measurement does this fact capture?
- What is the grain (one row per ___)?
- Which dimensions does it reference? (session, project, tool, model, date, time, file)
- What measures does it carry? (counts, durations, sizes)

## Step 2: Write failing tests

New file `tests/test_<fact>_v15.py` (one per populator -- see the 24 existing
`test_*_v15.py` files for the pattern). Cover: DDL (table + lineage block +
date_key/time_key), behavior (rows from a synthetic session via
`tests/conftest.py::write_minimal_session`), idempotency (rerun on unchanged
source = no-op), soft delete. Run tests, confirm they fail.

## Step 3: Add DDL

Add `CREATE TABLE IF NOT EXISTS fact_<name>` with the standard lineage block
in `src/ccutils/schemas/star/schema.py::create_star_schema()`.

## Step 4: Add the populator

Full detail, pitfalls, and the `lineage_upsert` contract:
`.claude/skills/etl-dev/references/new-fact-table.md`. Summary: build a temp
inbound table from `stg_log_entries` (or scoped permanent facts), delegate to
`lineage_upsert(conn, run=run, table=..., inbound_table=..., natural_key=...,
payload_cols=[...], hash_cols=[...])`. Wire into `run_v15_etl` in dependency
order (`fact_session_summary` stays LAST), then add to `_PROGRESS_TABLES` in
`export/duckdb_archive.py`.

## Step 5: Consider a semantic view

If this fact will be queried directly by users, add a `CREATE OR REPLACE VIEW semantic_<name>` in `schema.py` that joins relevant dimensions. Include date/time_of_day for filtering.

## Step 6: Run tests green

```bash
uv run pytest tests/test_star_schema_ddl.py tests/test_<fact>_v15.py -v --tb=short
```

## Step 7: Update documentation

- `docs/STAR_SCHEMA.md` -- table description + populator-order list
- `CHANGELOG.md` -- entry under `[Unreleased]`
- `README.md` -- "Tables populated by run_v15_etl" list if user-facing

## Step 8: Full suite

```bash
uv run pytest tests/ --confcutdir=tests -v --tb=short
```
