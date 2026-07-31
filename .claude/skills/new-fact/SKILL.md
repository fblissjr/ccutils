---
name: new-fact
description: Guided workflow for adding a new fact table to the star schema
---

Follow this workflow to add a new fact table. Practice TDD throughout.

## Step 1: Design the fact

Ask the user:
- What event or measurement does this fact capture?
- What is the grain (one row per ___)?
- Which dimensions does it reference? (session, project, tool, model, date, time, file)
- What measures does it carry? (counts, durations, sizes)
- Is it populated during per-session ETL or during `finalize_star_schema()` post-ETL?

Post-ETL facts (like `bridge_session_file`, `fact_agent_delegations`) require cross-session data and are populated in `src/ccutils/export/duckdb_archive.py`.

## Step 2: Write failing tests

In `tests/test_star_schema_ddl.py`:
- Table exists with expected columns and types

In `tests/test_star_schema_etl.py` (per-session facts) or `tests/test_star_schema_advanced.py` (post-ETL facts):
- Rows populated from fixture data
- Foreign keys reference valid dimension entries
- Measures have expected values

Run tests, confirm they fail.

## Step 3: Add DDL

Add `CREATE TABLE fact_<name>` in `src/ccutils/schemas/star/schema.py`.

## Step 4: Add ETL

For per-session facts:
- Add data collection in `_extract_star_data()` in `etl.py`
- Add a list field to `StarExtractionResult` dataclass
- Add INSERT loop in `_load_facts()`

For post-ETL facts:
- Add a new function in `duckdb_archive.py` (pattern: DELETE then INSERT)
- Call it from `finalize_star_schema()`
- Make it idempotent (DELETE before INSERT)

## Step 5: Consider a semantic view

If this fact will be queried directly by users, add a `CREATE OR REPLACE VIEW semantic_<name>` in `schema.py` that joins relevant dimensions. Include date/time_of_day for filtering.

## Step 6: Run tests green

```bash
uv run pytest tests/test_star_schema_ddl.py tests/test_star_schema_etl.py tests/test_star_schema_advanced.py -v --tb=short
```

## Step 7: Update documentation

- `docs/STAR_SCHEMA.md` -- table description with columns
- `CLAUDE.md` -- update table/view counts
- `CHANGELOG.md` -- entry under current version

## Step 8: Full suite

```bash
uv run pytest tests/ -v --tb=short
```
