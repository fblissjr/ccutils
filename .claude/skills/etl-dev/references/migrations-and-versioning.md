# Schema changes and release versioning

## There are no schema migrations

Since 1.0.0 the warehouse has no upgrade path. `create_star_schema`
fingerprints every base table's shape (name, columns, types, order) on
open and raises `SchemaMismatchError` when the file differs from what the
current DDL creates; the CLI turns that into "rebuild". `meta_schema_version`
is a one-row stamp (fingerprint + ccutils version), not a ledger.

So adding, renaming or dropping a column on a shipped table is one edit:
the `CREATE TABLE` in `schemas/star/schema.py`. Every existing warehouse is
then refused and rebuilt. Do not add an `ALTER`, a backfill on open, or a
repair: the whole documented upgrade-path bug class (ALTER carries no
DEFAULT, NULL-blind predicates over rows written before a column existed, a
content hash blind to a widened projection) existed only because one did.

Test for a shape change: `tests/test_schema_refuses_foreign_warehouse.py`
pins the contract; the column-list assertions in
`tests/test_star_schema_ddl.py` pin each table.

## Renaming columns

After any rename, grep two places or the suite/views break at a distance:

1. The `semantic_` views in `schemas/star/schema.py` (view creation validates
   column references on every `create_star_schema()` call -- a stale reference
   fails there).
2. The column-list assertions in `tests/test_star_schema_ddl.py`.

## History-retaining dimensions

`dim_facet_type` is the template: `CREATE TABLE IF NOT EXISTS` +
`INSERT ... ON CONFLICT DO NOTHING`, so historical rows (old prompt_versions)
survive re-seeding within one warehouse's life. Use the same pattern for any
future dim that must keep history across ETL runs.

## Release / version bump

Three places must move together -- a stale one makes lineage rows from
different contracts indistinguishable:

1. `version` in `pyproject.toml`.
2. `CHANGELOG.md` -- promote `[Unreleased]` to the new version.
3. `PARSER_VERSION` in `src/ccutils/_version.py` -- it stamps every lineage row
   (`dim_etl_version` / `record_source` chain) and the `meta_schema_version`
   stamp, so bump it whenever parsing or populator semantics changed, not just
   on release day.

Then tag `vX.Y.Z`. Semver binds from 1.0.0; **no major bumps without
permission**.

## Removing a feature

Grep imports, `__all__`, CLI registrations, and tests; delete; update
`CHANGELOG.md`, `README.md`, `_PROGRESS_TABLES`
(`export/duckdb_archive.py`); grep CLI help text for stale counts.
