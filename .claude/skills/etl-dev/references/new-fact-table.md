# Adding a new fact table / populator

Work through in order; each step has a test-first counterpart. Model the whole
thing on an existing thin populator (e.g. `etl/fact_diagnostics.py` for
derived facts, `etl/entry_type_facts.py` for per-entry-type facts).

## 1. Failing tests first (`tests/test_<fact>_v15.py`)

Cover four things before writing any implementation:

1. **DDL**: table exists after `create_star_schema()` with the standard
   lineage block (`created_at`, `last_updated_at`, `created_by_version_key`,
   `last_updated_by_version_key`, `etl_run_id`, `record_source`, `hash_diff`,
   `is_deleted`, `deleted_at`) plus `date_key`, `time_key`, and `session_id`
   as a degenerate dimension.
2. **Behavior**: rows land with correct business values from a synthetic
   session built with `tests/helpers_ccutils.py::write_minimal_session` (use
   `entry_session_id` when modeling agent files — the JSONL contract puts the
   PARENT's sessionId on every line).
3. **Idempotency**: run ETL twice on unchanged source; second run must be a
   no-op (hash_diff gate — assert zero updated rows via `fact_etl_steps` or
   unchanged `last_updated_at`).
4. **Soft delete**: an entry present in run 1 and absent in run 2 gets
   `is_deleted = TRUE`, not a hard DELETE.

## 2. DDL in `schemas/star/schema.py::create_star_schema()`

`CREATE TABLE IF NOT EXISTS` with the standard lineage block. `date_key` +
`time_key` are REQUIRED — `lineage_upsert` derives them and the INSERT fails
without them. If the table ships in a release and you later add a column, that
goes through `_COLUMN_MIGRATIONS` (see `migrations-and-versioning.md`), so get
the column set right now.

## 3. Populator `etl/fact_<x>.py`

Shape: build a temp inbound table (one row per natural key) from
`stg_log_entries` (or from permanent facts), then delegate to
`lineage_upsert(conn, run=run, table=..., inbound_table=..., natural_key=...,
payload_cols=[...], hash_cols=[...])`.

`lineage_upsert` pitfalls (each has shipped broken):

- `payload_cols` must NOT include `session_key`, the natural key, or
  `session_id` — those are handled separately; including them breaks the SQL.
- `hash_cols` = the mutable business columns. Omit one and changes to it never
  propagate; include a volatile per-run column and idempotency dies.
- Aggregate facts whose inbound table has no `timestamp` column pass
  `timestamp_col=` (e.g. `fact_session_summary` uses `first_timestamp`).
- **Shared tables** (two populators writing one table, e.g.
  `fact_session_facets`) MUST pass `soft_delete_scope_sql` or each populator
  soft-deletes the other's rows.
- Inbound built from **permanent facts** (not staging) must scope to staged
  sessions: `AND session_id IN (SELECT DISTINCT session_id FROM stg_log_entries ...)`.
- The step row (`upsert:<table>`) is self-recorded with real affected-row
  counts — don't add manual step bookkeeping around it.

## 4. Wire into `etl/orchestrator.py::run_v15_etl`

Insert at the right point in dependency order (see the populator-order list in
`docs/STAR_SCHEMA.md`). `fact_session_summary` stays LAST. If your populator
reads another fact, it goes after that fact's populator.

## 5. Progress display

Add the table to `_PROGRESS_TABLES` in `export/duckdb_archive.py`. That list
must contain every DATA fact `run_v15_etl` populates and exclude the audit
tables (`fact_etl_runs` / `fact_etl_batch_runs` / `fact_etl_steps`) — stale
entries undercount the display; audit rows would inflate it.

## 6. Optional semantic view

If consumers need a joined shape, add a `semantic_*` view in
`create_star_schema()`. View creation validates column references on every
call, so a bad reference fails fast. Filter `is_deleted = FALSE` in the view.

## 7. Docs + changelog

- `docs/STAR_SCHEMA.md`: table section + populator-order list.
- `CHANGELOG.md` under `[Unreleased]`.
- `README.md` "Tables populated by run_v15_etl" list if user-facing.

## 8. Verify

`uv run pytest tests/ --confcutdir=tests` fully green (+1 skipped live-API),
then an end-to-end smoke: `uv run ccutils all --format duckdb -o /tmp/etl-smoke`
against a fixture or real source, and query the new table.
