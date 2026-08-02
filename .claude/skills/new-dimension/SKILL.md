---
name: new-dimension
description: Guided workflow for adding a new dimension table to the star schema
---

Follow this workflow to add a new dimension table. Practice TDD throughout.
There is no separate `etl.py`/`_load_dimensions()`/`_extract_star_data()`
module in this codebase (pre-v0.15 architecture) -- dimensions are populated
either inline in `etl/orchestrator.py::_upsert_minimal_dimensions` (the
pattern for stub dims: `dim_session`, `dim_project`, `dim_tool`,
`dim_model` -- INSERT ... SELECT ... WHERE NOT EXISTS, keyed by `md5(...)`
computed directly in SQL) or in a dedicated `etl/dim_<name>.py` populator
for richer dims (see `dim_prompt.py`, `dim_session_chain.py` for the
pattern). Model the new dimension on whichever is the closer fit.

## Step 1: Design the dimension

Ask the user:
- What entity does this dimension describe?
- What attributes should it have?
- Is it a regular dimension or a degenerate dimension (low-cardinality categorical stored inline on facts)?

For degenerate dimensions, no table is needed -- just add a VARCHAR column directly to the relevant fact table.

## Step 2: Write failing tests

New file `tests/test_dim_<name>_v15.py`. Cover: table exists after
`create_star_schema()` with expected columns/types, dimension populates
from a synthetic session (`tests/helpers_ccutils.py::write_minimal_session`),
re-running ETL doesn't duplicate rows (`WHERE NOT EXISTS` / natural-key
uniqueness). Run tests, confirm they fail.

## Step 3: Add DDL

Add the `CREATE TABLE IF NOT EXISTS dim_<name>` statement in
`src/ccutils/schemas/star/schema.py::create_star_schema()`. Get the column
set right now -- a column added after the table ships in a release needs a
`_COLUMN_MIGRATIONS` entry (`CREATE TABLE IF NOT EXISTS` never widens an
existing table). See
`.claude/skills/etl-dev/references/migrations-and-versioning.md`.

## Step 4: Add ETL population

Either extend `_upsert_minimal_dimensions` (stub-dim pattern, inline SQL
keyed by `md5(natural_key)`) or add a new `etl/dim_<name>.py` populator
wired into `run_v15_etl` at the right point in dependency order. Surrogate
keys are `md5(natural_key)` computed in SQL, not the Python
`generate_dimension_key()` helper (that's used for entry-level identity and
embedding keys, not the core dimension upserts).

## Step 5: Run tests green

```bash
uv run pytest tests/test_star_schema_ddl.py tests/test_dim_<name>_v15.py -v --tb=short
```

## Step 6: Update documentation

- `docs/STAR_SCHEMA.md` -- add table to the appropriate section with column descriptions
- `CHANGELOG.md` -- entry under `[Unreleased]`

## Step 7: Run full suite

```bash
uv run pytest tests/ --confcutdir=tests -v --tb=short
```

Semantic views auto-validate on every `create_star_schema()` call (a bad
column reference fails fast), so add one only if the dimension needs a
joined shape consumers will query directly.
