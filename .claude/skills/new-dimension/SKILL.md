---
name: new-dimension
description: Guided workflow for adding a new dimension table to the star schema
---

Follow this workflow to add a new dimension table. Practice TDD throughout.

## Step 1: Design the dimension

Ask the user:
- What entity does this dimension describe?
- What attributes should it have?
- Is it a regular dimension or a degenerate dimension (low-cardinality categorical stored inline on facts)?

For degenerate dimensions, no table is needed -- just add a VARCHAR column directly to the relevant fact table.

## Step 2: Write failing tests

Create tests in `tests/test_star_schema_ddl.py`:
- Table exists after `create_star_schema()`
- All expected columns present with correct types
- Can insert a row

Create tests in `tests/test_star_schema_etl.py`:
- Dimension is populated after `run_star_schema_etl()`
- Expected rows present from sample fixture data

Run tests, confirm they fail.

## Step 3: Add DDL

Add the `CREATE TABLE dim_<name>` statement in `src/ccutils/schemas/star/schema.py` in the appropriate section.

## Step 4: Add ETL population

In `src/ccutils/schemas/star/etl.py`:
- Add extraction logic to `_extract_star_data()` or a new handler
- Add INSERT in `_load_dimensions()` or `_load_facts()` as appropriate
- Use `generate_dimension_key()` from `utils.py` for surrogate keys

## Step 5: Run tests green

```bash
uv run pytest tests/test_star_schema_ddl.py tests/test_star_schema_etl.py -v --tb=short
```

## Step 6: Update documentation

- `docs/STAR_SCHEMA.md` -- add table to the appropriate section with column descriptions
- `CLAUDE.md` -- update table counts and any relevant sections
- `CHANGELOG.md` -- add entry under current version

## Step 7: Run full suite

```bash
uv run pytest tests/ -v --tb=short
```

The semantic model (`semantic.py`) auto-introspects schema changes, so it usually doesn't need updates.
