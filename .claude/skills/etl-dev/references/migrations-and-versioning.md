# Schema migrations and release versioning

## Adding a column to a table that shipped in a release

`CREATE TABLE IF NOT EXISTS` never widens an existing table — a new column in
the CREATE statement silently does nothing on existing databases. Append the
column to `_COLUMN_MIGRATIONS` in `schemas/star/schema.py`. Mechanics:

- Migrations run **after** the CREATEs and **before** view creation, so views
  can reference migrated columns immediately.
- Schema-level DDL history is tracked in `meta_schema_version` (distinct from
  `dim_etl_version`, which tracks the business-rules/parser version).
- Test: assert the column exists on a database created from an *old* DDL
  snapshot, not just a fresh one.

## Renaming columns

After any rename, grep two places or the suite/views break at a distance:

1. The `semantic_` views in `schemas/star/schema.py` (view creation validates
   column references on every `create_star_schema()` call — a stale reference
   fails there).
2. The column-list assertions in `tests/test_star_schema_ddl.py`.

## History-retaining dimensions

`dim_facet_type` is the template: `CREATE TABLE IF NOT EXISTS` +
`INSERT ... ON CONFLICT DO NOTHING`, so historical rows (old prompt_versions)
survive re-seeding. Use the same pattern for any future dim that must keep
history across DDL runs.

## Release / version bump

Three places must move together — a stale one makes lineage rows from
different contracts indistinguishable:

1. `version` in `pyproject.toml`.
2. `CHANGELOG.md` — promote `[Unreleased]` to the new version.
3. `PARSER_VERSION` in `src/ccutils/_version.py` — it stamps every lineage row
   (`dim_etl_version` / `record_source` chain), so bump it whenever parsing or
   populator semantics changed, not just on release day.

Then tag `v0.X.0`. Semver; **no major bumps without permission**.

## Removing a feature

Grep imports, `__all__`, CLI registrations, and tests; delete; update
`CHANGELOG.md`, `README.md`, `_PROGRESS_TABLES`
(`export/duckdb_archive.py`); grep CLI help text for stale counts.
