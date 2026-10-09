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
   no-op (hash_diff gate — assert zero updated rows via `etl.steps` or
   unchanged `last_updated_at`).
4. **Soft delete**: an entry present in run 1 and absent in run 2 gets
   `is_deleted = TRUE`, not a hard DELETE.

## 2. DDL in `schemas/star/schema.py::create_star_schema()`

`CREATE TABLE IF NOT EXISTS` with the standard lineage block. `date_key` +
`time_key` are REQUIRED — `lineage_upsert` derives them and the INSERT fails
without them. There are no migrations: a column added later is one edit to
the CREATE, and every existing warehouse is refused and rebuilt (see
`migrations-and-versioning.md`).

## 3. Populator `etl/fact_<x>.py`

Shape: build a temp inbound table (one row per natural key) from
`etl.log_entries` (or from permanent facts), then delegate to
`lineage_upsert(conn, run=run, table=..., inbound_table=..., natural_key=...,
payload_cols=[...], hash_cols=[...])`.

`lineage_upsert` pitfalls (each has shipped broken):

- `payload_cols` must NOT include `session_key`, the natural key, or
  `session_id` — those are handled separately; including them breaks the SQL.
- `hash_cols` = the mutable business columns. Omit one and changes to it never
  propagate; include a volatile per-run column and idempotency dies.
- Aggregate facts whose inbound table has no `timestamp` column pass
  `timestamp_col=` (e.g. `semantic_session_summary` uses `first_timestamp`).
- **Shared tables** (two populators writing one table, e.g.
  `fact_session_facets`) MUST pass `soft_delete_scope_sql` or each populator
  soft-deletes the other's rows.
- Inbound built from **permanent facts** (not staging) must scope to staged
  sessions: `AND session_id IN (SELECT DISTINCT session_id FROM etl.log_entries ...)`.
- The step row (`upsert:<table>`) is self-recorded with real affected-row
  counts — don't add manual step bookkeeping around it.

## 4. Wire into `etl/orchestrator.py::run_v15_etl`

Insert at the right point in dependency order (see the populator-order list in
`docs/STAR_SCHEMA.md`). nothing has to run last any more: the summary is a view. If your populator
reads another fact, it goes after that fact's populator.

## 5. Progress display

Add the table to `_PROGRESS_TABLES` in `export/duckdb_archive.py`. That list
must contain every DATA fact `run_v15_etl` populates and exclude the audit
tables (`etl.runs` / `etl.batch_runs` / `etl.steps`) — stale
entries undercount the display; audit rows would inflate it.

## 6. Declare it

Two dicts beside the DDL in `schemas/star/schema.py`, each pinned by a drift
test that fails until the table is in it:

- `NATURAL_KEYS`: table -> the key the projection emits one row per.
  `lineage_upsert` raises on a duplicate, and `ccutils audit` walks this dict.
- `TABLE_COVERAGE`: status, what writes it, and why it exists. The reader's
  guide and the audit's coverage check are generated from it.

## 6b. A view, only if it earns one

`docs/ETL_ARCHITECTURE.md` rule 3: an object exists because it encodes
something a consumer would get wrong, not because it saves a JOIN. The
plain-join views are being deleted (`docs/ROADMAP.md`), so do not add
another. A view that does earn its place filters `is_deleted = FALSE`, and
is validated on every `create_star_schema()` call, so a bad column
reference fails fast.

If the table is only a rollup of other facts, it should BE the view:
`semantic_session_summary` and `semantic_agent_delegations` were both tables
first, and both went stale or needed a repair pass until they were not.

## 7. Docs + changelog

- `docs/STAR_SCHEMA.md`: table section + populator-order list.
- `CHANGELOG.md` under `[Unreleased]`.
- `README.md` "Tables populated by run_v15_etl" list if user-facing.

## 8. Verify

`uv run pytest tests/ --confcutdir=tests` fully green (+1 skipped live-API),
then against the real corpus, because fixtures agree with whoever wrote
them: a one-project build, `uv run ccutils --source --format duckdb -o <dir>
-p <project>`, then `uv run ccutils audit -o <dir>`, and query the new table.
Keep `<dir>` out of any git worktree: a warehouse holds unredacted
transcripts.
