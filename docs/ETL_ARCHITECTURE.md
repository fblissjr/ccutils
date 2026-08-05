# ETL Architecture — target state and the rules behind it

Last updated: 2026-08-05

Companion to `STAR_SCHEMA.md`, which describes the warehouse **as built**. This
document describes the layering the ETL is **moving to**, and — more
importantly — the three rules that decide where any new table, view or
populator belongs.

> **Status: DECIDED, NOT YET IMPLEMENTED.** The rules below govern new work
> starting now, so anything added today should be born conforming. The
> existing populators do not conform yet; the migration is tracked in
> `internal/plans/etl_layer_rewrite.md`. Where this document and the current
> code disagree, the code is the present tense and this document is the
> target — do not read it as a description of what runs today.

---

## The layering

```
Tier 0   raw JSONL on disk                        immutable
   |
Tier 1   parquet lake                             re-derivable cache, full history
   |
Tier 2   staging: N SHAPED tables                 transient, one incremental slice per session
           stg_log_entries    raw lines (exists today)
           stg_tool_uses      \
           stg_tool_results    >  extracted ONCE, here
           stg_messages       /
           stg_attachments   ...
   |
Tier 3   dims + facts                             read staging ONLY; compute their own keys
   |
Tier 4   views                                    anything fully derived
```

The change from today is Tier 2. There is currently **one** staging table
holding raw lines with the payloads still as JSON columns, so extraction
happens inside the facts — and any second consumer of an extracted shape has
to either duplicate the extraction or read the fact that did it. That is the
root cause of every fact-to-fact dependency in the codebase.

---

## Rule 1 — dims and facts read staging, never each other

A populator's only input is staging. Not another fact, not another dim, not
even a dim consulted purely to resolve a surrogate key (see Rule 2).

**Why.** Reading another fact creates a dependency that is invisible in the
schema and enforced only by populator ordering. The costs are all present in
this repo's history:

- Four populators filtered `is_deleted` on the wrong side of a
  `fact_tool_results` join. Two of them died on duplicate natural keys; the
  third silently double-counted `tool_count` and `error_count` with no
  assertion to catch it.
- The natural-key fan-out that killed three sessions' ETL after every session
  had already been processed.
- The scaffolding that exists only to manage the ordering: the
  `fact_session_summary` runs LAST rule, the cross-session reconciliation
  pass, and the "populators reading permanent facts must scope inbound to
  staged sessions" rule.

When extraction moves into staging, all of that scaffolding becomes
unnecessary rather than merely better-documented.

**Two edges, both decided:**

- `dim_session_heuristics` does not write a table — it `UPDATE`s `dim_session`
  with `intent` / `complexity` / `outcome` / `domain`, computed from three
  facts. A dimension cannot become a view, so instead the classifiers run over
  **staging** during `dim_session`'s own populate. The columns stay where
  consumers expect them; the fact dependency disappears.
- `fact_session_facets` mixes Tier 1 (SQL-derivable) and Tier 2 (billed Haiku
  calls, not recomputable). It splits: Tier 2 stays a stored table, Tier 1
  becomes a view, and a union view presents them as one for consumers that
  just want every facet for a session.

---

## Rule 2 — facts compute their own surrogate keys

A fact derives a key from the natural key inline (`md5(...)`) rather than
looking it up in a dimension. `session_key = md5(session_id)` is already this
pattern.

**The prerequisite this creates.** If five facts each inline `md5(file_path)`
and the dim populator does something subtly different — trailing slash, case,
resolved versus raw path — two facts disagree about the same file's key and
nothing catches it. That is the eight-drifted-copies failure the project rule
about `project_dir_sql` already exists to prevent.

So each key formula gets **one definition**, exposed in both a Python and a
SQL form, used by the dimension populator and by every fact, with a drift test
asserting the two forms agree. Today the formulas are already split across
`generate_dimension_key()` (Python) and `project_key_sql()` / `md5(col)` (SQL)
with nothing asserting they match.

**Accepted consequence:** a fact may reference a key whose dimension row does
not exist yet (late-arriving dimension). This is already the documented
pattern for `agent_session_key`; consumers `LEFT JOIN`.

---

## Rule 3 — an object earns its existence by encoding something a consumer would get wrong

Not by saving someone a `JOIN`.

This is the test for whether a table or view should exist at all. Applied to
the 20 semantic views, it sorts them cleanly:

| Encodes logic — keep | What it encodes |
|---|---|
| `semantic_sessions`, `semantic_tool_calls`, `semantic_decisions`, `semantic_session_behavior`, `semantic_token_usage`, `semantic_context_growth`, `semantic_cost_analysis`, `semantic_etl_runs` | soft-delete filters |
| `semantic_cost_analysis`, `semantic_tool_patterns` | tri-state `is_error`, corrected denominators |
| `semantic_memory`, `semantic_memory_links`, `semantic_prompt_history` | `is_current` on a Type 2 SCD |
| `semantic_context_growth` | window functions / dedupe |
| `semantic_file_evolution`, `semantic_project_files` | aggregation |

Every one of those categories has a corresponding defect in `CHANGELOG.md`:
soft-deletes filtered on the wrong join side; `is_error = FALSE` silently
dropping 38,035 succeeded-by-omission rows; the R11 cache-hit denominator.
Those views are not convenience — they are the accumulated corrections, and
deleting them relocates the bug into every query that would have used them.

| Pure joins — delete |
|---|
| `semantic_messages`, `semantic_file_operations`, `semantic_session_chains`, `semantic_agent_delegations`, `semantic_plan_revisions`, `semantic_project_context` |

These add a name and nothing else. (`semantic_project_context` fails
independently too: 78% column overlap with `semantic_sessions`.)

**Check consumers before deleting.** Absence of guards is measurable, but a
consumer may depend on a view's exact column set, and the `query-warehouse`
skill references several by name.

### Table or view?

Rule 3 decides whether an object should exist. This decides how it is stored:

- **Fully derived → view.** Measured on a 2,447-session / 1.5 GB warehouse, a
  three-table session rollup runs in **9–10 ms**; the real nine-table version
  is ~30–50 ms. There is no performance case for materialising it. Lineage
  columns on a fully-derived object answer a question about the cache, not
  about the data.
  - `fact_session_summary` (49 columns, all derived) → view
  - `bridge_session_file` → view. It earns *existence* (real aggregation:
    operation/read/write/edit counts, chars written, first/last timestamps per
    session-file) but not *storage*.
- **Referenced by surrogate key → table.** `dim_session_chain` stays a table
  because `dim_session.chain_key` is a foreign key into it. It is being joined
  *to*, not joining things together.

> **Trap, must ship with any such demotion:** `json_export.py` selects
> `table_type = 'BASE TABLE'`. Converting a table to a view silently drops it
> from the JSON export with no error. One-line fix, plus a test asserting the
> object still appears.

---

## Run metadata

Three grains, joined `batch → run → step`:

| Table | Grain |
|---|---|
| `fact_etl_batch_runs` | one CLI invocation |
| `fact_etl_runs` | one session ETL — carries `batch_run_id`, CDC window, `run_kind` |
| `fact_etl_steps` | one task within a run — `(etl_run_id, step_id)` |

Already correct and not to be rebuilt: insert / update / soft-delete counts
are separated at every grain, and every rollup is derived from children rather
than tallied by the caller.

**The gap:** table identity is encoded *inside* `step_name` as the string
`upsert:fact_messages`, so "counts by table" is a string parse rather than a
`GROUP BY`. Adding `table_name` (NULL for stage steps, which `step_kind`
already discriminates) plus `data_start_ts` / `data_end_ts` at step grain
gives each table its own CDC window instead of inheriting the session's.

---

## How this gets verified

Two gates, both cheap, both born from defects this codebase actually shipped:

1. **Upgrade-path fixture.** Defects in one cycle were reachable only on an
   *upgraded* warehouse and invisible on a fresh build — including one that
   shipped in a tagged release, and one where the *fix* for it was itself
   broken in the same way (edges relinked but never resolved, because
   resolution ran only on the path where new versions were written). A rebuild
   recomputes everything and structurally cannot show that class. A `conftest`
   fixture materialising a previous-release-shaped warehouse turns "remember
   to check" into "the suite fails if you didn't".

   **This is overdue, not preparatory.** It is listed here as a gate for the
   rewrite, but every one of the defects above predates the rewrite. Each was
   caught by a human review round that the fixture would have made unnecessary.

   Corollary, learned the same way: **assert the property that makes the
   feature work, not the one that is easiest to query.** A test asserting an
   edge row EXISTS passes while the edge is unresolved and unusable. Reach for
   the consumer-facing surface — the semantic view — rather than the table the
   populator just wrote.
2. **Full-corpus regression baseline.** Re-run the ETL against a copy of a
   real multi-thousand-session warehouse and diff per-table row counts. A
   silent count change is the signature of most of what goes wrong here.
