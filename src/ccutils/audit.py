"""Warehouse plausibility checks: the eyeballing that found every shipped
structural bug, as a command that can gate.

Four families:

- structural invariants (natural keys unique, API-response grain, foreign
  keys resolve, views execute, no run left 'running' or 'failed');
- coverage reconciliation (what `etl.table_coverage` declares against what
  `etl.steps.table_name` says was written);
- lake reconciliation (the declared grains of `fact_messages` and
  `fact_tool_calls`, per session, against the Parquet lake the build wrote);
- column plausibility (a column 100% NULL, or single-valued, on a table
  with enough rows to mean it).

The first three are ERRORS: they decide the exit code. Column plausibility
is a WARNING unless the caller asks for strict. It fires on any build
scoped to one project, where one git branch or one memory scope is the
scope and not a defect, and on 2026-11-13 it made a warehouse whose every
run had failed and its correct rebuild exit alike. It still earns its keep
on a full-corpus build, which is what `--strict` is for.

Every finding is (check, object, detail, measured). `etl.audit_exceptions`
is the allowlist: an accepted finding is reviewable data with a reason, not
a deleted check. A finding that matches an exception is reported as
allowed, not failed.

The sparse-column checks skip below a row threshold and SAY so: one
session cannot tell a NULL column from an unwritten one, and a green that
reflects "did not look" must be distinguishable from one that looked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ccutils.schemas.star.json_export import _KEY_TARGETS
from ccutils.schemas.star.schema import NATURAL_KEYS, TABLE_COVERAGE

# Below this many live rows, "100% NULL" and "single-valued" are not
# evidence of anything. Calibrated so a one-session fixture skips and a
# real corpus does not.
MIN_ROWS_FOR_COLUMN_CHECKS = 50

# Columns whose NULL or single value is structural, not a finding.
_STRUCTURAL_COLUMNS = {
    "deleted_at", "valid_to", "error_message", "is_deleted",
    "record_source", "created_by_version_key", "last_updated_by_version_key",
    "etl_run_id", "hash_diff", "created_at", "last_updated_at",
    # Type 2 bookkeeping: identical on every row of a single-import build.
    "valid_from", "is_current", "version_num",
}


@dataclass(frozen=True)
class Finding:
    check: str
    object: str
    detail: str | None
    measured: str

    def __str__(self) -> str:
        where = f"{self.object}.{self.detail}" if self.detail else self.object
        return f"{self.check}: {where} -- {self.measured}"


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)   # errors
    warnings: list[Finding] = field(default_factory=list)
    allowed: list[Finding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def strict_clean(self) -> bool:
        return not self.findings and not self.warnings


def _scalar(conn, sql, params=None):
    return conn.execute(sql, params or []).fetchone()[0]


def _live_count(conn, table):
    """Live rows, or -1 when the table is missing (a missing declared table
    is reported by the coverage checks, not by crashing the row count)."""
    cols = _columns(conn, table)
    if not cols:
        return -1
    where = " WHERE is_deleted = FALSE" if "is_deleted" in cols else ""
    return _scalar(conn, f"SELECT COUNT(*) FROM {table}{where}")


def _columns(conn, table):
    schema, _, name = table.rpartition(".")
    schema = schema or "main"
    return [
        r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = ? AND table_name = ? ORDER BY ordinal_position",
            [schema, name],
        ).fetchall()
    ]


def _exceptions(conn):
    return {
        (c, o, d): reason
        for c, o, d, reason in conn.execute(
            "SELECT check_name, object_name, detail, reason FROM etl.audit_exceptions"
        ).fetchall()
    }


# --------------------------------------------------------------------------
# checks: each yields Finding objects
# --------------------------------------------------------------------------

def check_natural_key_unique(conn):
    for table, key in NATURAL_KEYS.items():
        total, distinct = conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT {key}) FROM {table} "
            f"WHERE is_deleted = FALSE AND {key} IS NOT NULL"
        ).fetchone()
        if total != distinct:
            yield Finding("natural_key_unique", table, key,
                          f"{total} live rows, {distinct} distinct keys")


def check_api_response_grain(conn):
    """One row per API response WITHIN a session. Across sessions the same
    api_message_id legitimately recurs: a continued session replays its
    history (706 ids across 1,403 rows on the corpus, every one inside a
    session chain), which is a fact about the source, not a grain bug."""
    total, distinct = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT (session_id, api_message_id)) FROM fact_token_usage "
        "WHERE is_deleted = FALSE AND api_message_id IS NOT NULL"
    ).fetchone()
    if total != distinct:
        yield Finding("api_response_grain", "fact_token_usage", "api_message_id",
                      f"{total} rows for {distinct} (session, API response) pairs")


def check_stuck_runs(conn):
    for table in ("etl.runs", "etl.batch_runs"):
        n = _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE status = 'running'")
        if n:
            yield Finding("stuck_runs", table, None, f"{n} rows still 'running'")


def check_failed_runs(conn):
    """A session run that failed loaded nothing past the step that raised,
    and the dimensions written before it stay: `dim_session` still lists
    the session. The build exits 1 when this happens; a warehouse opened
    later has only this row to say so."""
    n, example = conn.execute(
        "SELECT COUNT(*), ANY_VALUE(error_message) FROM etl.runs "
        "WHERE status = 'failed'"
    ).fetchone()
    if n:
        first = (example or "").splitlines()[0][:120] if example else "no error recorded"
        yield Finding("failed_runs", "etl.runs", None,
                      f"{n} runs failed, e.g. {first}")


# The lake every build writes beside its warehouse: one log_entries file per
# session (parsers/parquet_writer.py).
LAKE_GLOB = "projects/*/sessions/*/log_entries.parquet"


def check_lake_reconciliation(conn, lake_root: Path):
    """Each declared grain, per session, against the lake.

    `fact_messages` is one row per user/assistant entry and
    `fact_tool_calls` one row per tool call per transcript (TABLE_COVERAGE),
    so per session the lake and the live rows must agree. Only lake files a
    SUCCESSFUL run of this warehouse wrote are counted: a lake directory can
    hold another build's files, or a failed run's (`failed_runs` reports
    those). Session identity is the lake's `session_id` column, stamped from
    the file name at Tier 1, the same column the facts carry.
    """
    files = [str(p) for p in sorted(lake_root.glob(LAKE_GLOB))]
    if not files:
        return
    conn.execute(
        "CREATE OR REPLACE TEMP TABLE _audit_lake AS "
        "SELECT session_id, type, message_json FROM read_parquet(?) "
        "WHERE etl_run_id IN (SELECT etl_run_id FROM etl.runs WHERE status = 'success')",
        [files],
    )
    try:
        grains = {
            "fact_messages": (
                "SELECT session_id, COUNT(*) AS n FROM _audit_lake "
                "WHERE type IN ('user', 'assistant') GROUP BY 1",
                "SELECT session_id, COUNT(*) AS n FROM fact_messages "
                "WHERE is_deleted = FALSE GROUP BY 1",
            ),
            "fact_tool_calls": (
                "SELECT session_id, COUNT(DISTINCT json_extract_string(b, '$.id')) AS n "
                "FROM (SELECT session_id, unnest(json_extract(message_json, '$.content[*]')) AS b "
                "      FROM _audit_lake WHERE type = 'assistant') "
                "WHERE json_extract_string(b, '$.type') = 'tool_use' GROUP BY 1",
                "SELECT session_id, COUNT(*) AS n FROM fact_tool_calls "
                "WHERE is_deleted = FALSE GROUP BY 1",
            ),
        }
        for table, (lake_sql, fact_sql) in grains.items():
            rows = conn.execute(
                f"""
                WITH lake AS ({lake_sql}), fact AS ({fact_sql}),
                     scoped AS (SELECT DISTINCT session_id FROM _audit_lake)
                SELECT s.session_id, COALESCE(l.n, 0), COALESCE(f.n, 0)
                FROM scoped s
                LEFT JOIN lake l USING (session_id)
                LEFT JOIN fact f USING (session_id)
                WHERE COALESCE(l.n, 0) <> COALESCE(f.n, 0)
                ORDER BY s.session_id
                """
            ).fetchall()
            if rows:
                sid, lake_n, fact_n = rows[0]
                yield Finding(
                    "lake_reconciliation", table, None,
                    f"{len(rows)} sessions disagree with the lake, "
                    f"e.g. {sid}: lake {lake_n}, table {fact_n}",
                )
    finally:
        conn.execute("DROP TABLE IF EXISTS _audit_lake")


def check_fk_unresolved(conn, tables):
    for table in tables:
        cols = _columns(conn, table)
        for col in cols:
            target = _KEY_TARGETS.get(col)
            if target is None:
                continue
            to_table, to_col = target
            if table == to_table and col == to_col:
                continue
            where = " AND t.is_deleted = FALSE" if "is_deleted" in cols else ""
            n = _scalar(
                conn,
                f"SELECT COUNT(*) FROM {table} t WHERE t.{col} IS NOT NULL{where} "
                f"AND NOT EXISTS (SELECT 1 FROM {to_table} d WHERE d.{to_col} = t.{col})",
            )
            if n:
                yield Finding("fk_unresolved", table, col,
                              f"{n} values with no row in {to_table}")


def check_declared_but_unwritten(conn):
    written = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT table_name FROM etl.steps WHERE status = 'success'"
        ).fetchall()
    }
    kinds_run = {
        r[0] for r in conn.execute("SELECT DISTINCT run_kind FROM etl.runs").fetchall()
    }
    for name, (kind, status, by, _) in TABLE_COVERAGE.items():
        if kind != "table" or status != "populated":
            continue
        if by == "session" and "session" in kinds_run and name not in written:
            yield Finding("declared_but_unwritten", name, None,
                          "declared populated per session; no successful step names it")
        if by == "global_source" and "global_source" in kinds_run and name not in written:
            yield Finding("declared_but_unwritten", name, None,
                          "declared populated by a global source; no successful step names it")


def check_rows_without_step(conn):
    written = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT table_name FROM etl.steps WHERE status = 'success'"
        ).fetchall()
    }
    for name, (kind, status, by, _) in TABLE_COVERAGE.items():
        if kind != "table" or by == "ddl_seed" or name in written:
            continue
        n = _live_count(conn, name)
        if n > 0:
            yield Finding("rows_without_step", name, None,
                          f"{n} rows but no successful step names the table")


def check_view_executes(conn):
    """A view that raises is dead for every consumer. Zero rows is not a
    finding: a delegation view over a corpus with no delegations is empty
    and correct."""
    for name, (kind, status, _, _) in TABLE_COVERAGE.items():
        if kind != "view" or status != "keep":
            continue
        try:
            _scalar(conn, f"SELECT COUNT(*) FROM {name}")
        except Exception as exc:  # noqa: BLE001 -- the failure IS the finding
            yield Finding("view_executes", name, None, f"raises: {str(exc).splitlines()[0][:120]}")


def check_null_columns(conn, tables):
    for table in tables:
        cols = [c for c in _columns(conn, table) if c not in _STRUCTURAL_COLUMNS]
        if not cols:
            continue
        where = " WHERE is_deleted = FALSE" if "is_deleted" in _columns(conn, table) else ""
        counts = conn.execute(
            "SELECT COUNT(*), " + ", ".join(f"COUNT({c})" for c in cols) + f" FROM {table}{where}"
        ).fetchone()
        total = counts[0]
        for col, non_null in zip(cols, counts[1:]):
            if non_null == 0:
                yield Finding("null_column", table, col, f"NULL on all {total} live rows")


def check_single_valued_columns(conn, tables):
    """A DENSE column (non-NULL on every live row) with one distinct value
    discriminates nothing. Sparse columns with one value are normal: a
    rare-event column is mostly NULL and one value when set. Foreign-key
    columns are skipped: one project_key on a build scoped to one project
    is scope, not a defect."""
    for table in tables:
        cols = [
            c for c in _columns(conn, table)
            if c not in _STRUCTURAL_COLUMNS and c not in _KEY_TARGETS
            and c not in ("project_path", "project_name")
        ]
        if not cols:
            continue
        where = " WHERE is_deleted = FALSE" if "is_deleted" in _columns(conn, table) else ""
        row = conn.execute(
            "SELECT COUNT(*), "
            + ", ".join(f"COUNT(DISTINCT {c}), COUNT({c})" for c in cols)
            + f" FROM {table}{where}"
        ).fetchone()
        total = row[0]
        for i, col in enumerate(cols):
            distinct, non_null = row[1 + 2 * i], row[2 + 2 * i]
            if distinct == 1 and non_null == total:
                value = conn.execute(f"SELECT MIN({col}) FROM {table}").fetchone()[0]
                yield Finding("single_valued_column", table, col,
                              f"one distinct value ({value!r}) on every one of {total} live rows")


def check_delegation_ground_truth(conn):
    """Where a delegation states a tool count AND its agent's transcript
    records a completion, the derived count can be scored against the stated
    one. Both are columns of the same view row, under different names."""
    row = conn.execute(
        """
        SELECT COUNT(*),
               COUNT(*) FILTER (
                   WHERE agent_total_tool_use_count <> derived_tool_use_count)
        FROM semantic_agent_delegations
        WHERE completion_state = 'completed'
          AND agent_total_tool_use_count IS NOT NULL
          AND derived_tool_use_count IS NOT NULL
        """
    ).fetchone()
    scored, wrong = row
    if scored and wrong:
        yield Finding("delegation_ground_truth", "semantic_agent_delegations",
                      "derived_tool_use_count",
                      f"derived value disagrees with the stated one on {wrong} of {scored} scorable rows")


# --------------------------------------------------------------------------

def run_audit(conn, lake_root: Path | None = None) -> AuditReport:
    """Run every check. ``lake_root`` is the build's ``parquet_lake``
    directory; without one, ``lake_reconciliation`` is a stated skip."""
    report = AuditReport()
    exceptions = _exceptions(conn)

    populated_tables = [
        n for n, (kind, status, _, _) in TABLE_COVERAGE.items()
        if kind == "table" and status == "populated"
    ]
    # Seeded constants (dim_time) are not evidence of anything.
    big_tables = [
        t for t in populated_tables
        if TABLE_COVERAGE[t][2] != "ddl_seed"
        and _live_count(conn, t) >= MIN_ROWS_FOR_COLUMN_CHECKS
    ]

    checks = [
        ("natural_key_unique", lambda: check_natural_key_unique(conn)),
        ("api_response_grain", lambda: check_api_response_grain(conn)),
        ("stuck_runs", lambda: check_stuck_runs(conn)),
        ("failed_runs", lambda: check_failed_runs(conn)),
        ("fk_unresolved", lambda: check_fk_unresolved(conn, populated_tables)),
        ("declared_but_unwritten", lambda: check_declared_but_unwritten(conn)),
        ("rows_without_step", lambda: check_rows_without_step(conn)),
        ("view_executes", lambda: check_view_executes(conn)),
        ("delegation_ground_truth", lambda: check_delegation_ground_truth(conn)),
    ]
    if lake_root is not None and any(Path(lake_root).glob(LAKE_GLOB)):
        checks.append(("lake_reconciliation",
                       lambda: check_lake_reconciliation(conn, Path(lake_root))))
    else:
        report.skipped["lake_reconciliation"] = (
            f"no lake at {lake_root}" if lake_root is not None else "no lake given"
        )
    warning_checks = {"null_column", "single_valued_column"}
    if big_tables:
        checks.append(("null_column", lambda: check_null_columns(conn, big_tables)))
        checks.append(("single_valued_column", lambda: check_single_valued_columns(conn, big_tables)))
    else:
        note = f"no populated table has {MIN_ROWS_FOR_COLUMN_CHECKS}+ live rows"
        report.skipped["null_column"] = note
        report.skipped["single_valued_column"] = note

    for name, fn in checks:
        report.checks_run.append(name)
        try:
            produced = list(fn())
        except Exception as exc:  # noqa: BLE001 -- a crashing check must not hide the rest
            produced = [Finding("check_error", name, None,
                                f"check raised: {str(exc).splitlines()[0][:120]}")]
        for finding in produced:
            key_exact = (finding.check, finding.object, finding.detail)
            key_any = (finding.check, finding.object, None)
            if key_exact in exceptions or key_any in exceptions:
                report.allowed.append(finding)
            elif name in warning_checks:
                report.warnings.append(finding)
            else:
                report.findings.append(finding)
    return report


def format_report(report: AuditReport, path) -> str:
    lines = [f"ccutils audit: {path}", ""]
    for f in report.findings:
        lines.append(f"  FAIL  {f}")
    for f in report.warnings:
        lines.append(f"  warn  {f}")
    for f in report.allowed:
        lines.append(f"  allow {f}")
    for check, why in report.skipped.items():
        lines.append(f"  skip  {check}: {why}")
    lines.append("")
    lines.append(
        f"{len(report.findings)} findings, {len(report.warnings)} warnings, "
        f"{len(report.allowed)} allowed, "
        f"{len(report.checks_run)} checks run, {len(report.skipped)} skipped"
    )
    return "\n".join(lines)
