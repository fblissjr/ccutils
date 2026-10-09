# Adding a new dimension table

Work through in order, test first. There are three shapes; pick before
writing anything, because they share almost no code.

| Shape | When | Model on |
|---|---|---|
| Degenerate | A low-cardinality categorical (a type, a status, a language) | No table: a VARCHAR column on the fact that carries it |
| Stub dimension | An entity named by every session (project, tool, model) | `etl/orchestrator.py::_upsert_minimal_dimensions`: `INSERT ... SELECT ... WHERE NOT EXISTS`, keyed by `md5(natural_key)` computed in SQL |
| Dedicated populator | An entity with its own attributes or its own source | `etl/dim_session_chain.py` (per session), `etl/dim_prompt.py` (a global source), `etl/dim_memory.py` (Type 2, versioned) |

A Type 2 dimension and a dimension fed by a global source each have a rule
in `CLAUDE.md` that exists because the first attempt shipped broken. Read
those two before modelling on `dim_memory` or `dim_prompt`.

## 1. Failing tests first (`tests/test_dim_<name>_v15.py`)

- The table exists after `create_star_schema()` with the expected columns
  and types.
- It populates from a synthetic session built with
  `tests/helpers_ccutils.py::write_minimal_session`.
- Re-running the ETL does not duplicate rows.
- The key covers the entity's whole identity, not the part that happens to
  be unique in the corpus. Write a test with two entities that differ only
  in the field you were tempted to leave out.

Run them and watch them fail.

## 2. DDL in `schemas/star/schema.py::_create_objects()`

`CREATE TABLE IF NOT EXISTS dim_<name>`. There are no migrations: a column
added after the table ships is one edit to the CREATE, and every existing
warehouse is refused on open and rebuilt (`migrations-and-versioning.md`).

Declare the table in `TABLE_COVERAGE` beside the DDL, with what writes it
and why it exists. A drift test fails until you do.

## 3. Population

Extend `_upsert_minimal_dimensions` for a stub dimension, or add
`etl/dim_<name>.py` and wire it into `run_v15_etl` in dependency order.
Surrogate keys are `md5(natural_key)` computed in SQL. The Python
`generate_dimension_key()` helper is for entry-level identity at Tier 1
(`parsers/parquet_writer.py`), not for dimension keys.

Every step that writes the table records it: `run.step(..., table="dim_<name>")`,
so `etl.steps.table_name` and `ccutils audit` can see it was written.

## 4. A view, only if it earns one

`docs/ETL_ARCHITECTURE.md` rule 3: an object exists because it encodes
something a consumer would get wrong, not because it saves a JOIN. Views
are validated on every `create_star_schema()` call, so a bad column
reference fails fast.

## 5. Docs and the full suite

- `docs/STAR_SCHEMA.md`: the table, with column descriptions.
- `CHANGELOG.md`: an entry under `[Unreleased]`.
- `uv run pytest tests/ --confcutdir=tests`
