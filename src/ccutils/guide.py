"""The reader's guide: generated from the warehouse, written beside it.

Everything here is measured from the file it describes -- coverage rows,
step history, row counts, accepted audit findings, the join convention --
so it says what THIS warehouse holds. The primary reader is an agent
opening the file cold.
"""

from __future__ import annotations

from datetime import datetime

from ccutils._version import PARSER_VERSION
from ccutils.schemas.star.json_export import _KEY_TARGETS

GUIDE_NAME = "README.md"


def _rows(conn, sql, params=None):
    return conn.execute(sql, params or []).fetchall()


def _scalar(conn, sql, params=None):
    return conn.execute(sql, params or []).fetchone()[0]


def _live_count(conn, table):
    cols = {r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()}
    where = " WHERE is_deleted = FALSE" if "is_deleted" in cols else ""
    return _scalar(conn, f"SELECT COUNT(*) FROM {table}{where}")


def render_guide(conn) -> str:
    coverage = _rows(
        conn,
        "SELECT object_name, object_type, status, populated_by, reason "
        "FROM etl.table_coverage ORDER BY object_type DESC, object_name",
    )
    written = {
        r[0] for r in _rows(conn, "SELECT DISTINCT table_name FROM etl.steps WHERE status = 'success'")
    }
    stamp = conn.execute(
        "SELECT ccutils_version, created_at FROM etl.schema_version LIMIT 1"
    ).fetchone()
    projects = _rows(conn, "SELECT project_name FROM dim_project ORDER BY 1")
    sessions, agents = conn.execute(
        "SELECT COUNT(*), COUNT(*) FILTER (WHERE is_agent) FROM dim_session"
    ).fetchone()
    first, last = conn.execute(
        "SELECT MIN(first_timestamp), MAX(last_timestamp) FROM dim_session"
    ).fetchone()
    runs = _rows(
        conn,
        "SELECT run_kind, COUNT(*), COUNT(*) FILTER (WHERE status = 'success') "
        "FROM etl.runs GROUP BY 1 ORDER BY 1",
    )
    exceptions = _rows(
        conn,
        "SELECT check_name, object_name, detail, reason FROM etl.audit_exceptions ORDER BY 1, 2, 3",
    )

    out = []
    w = out.append
    w("# This warehouse")
    w("")
    w(f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')} by ccutils "
      f"{PARSER_VERSION} from the file beside this guide (`archive.duckdb`). "
      "Everything below is measured from that file, not from documentation. "
      "Regenerate with `ccutils guide -o <this directory>`; check the file with "
      "`ccutils audit -o <this directory>`.")
    w("")
    if stamp:
        w(f"Schema written by ccutils {stamp[0]} on {stamp[1]}. A warehouse written by "
          "a different schema is refused on open and must be rebuilt; there are no migrations.")
        w("")
    w("## What it holds")
    w("")
    w(f"- Projects ({len(projects)}): " + (", ".join(p[0] for p in projects) if projects else "none"))
    w(f"- Sessions: {sessions}, of which {agents} are subagent sessions "
      "(one row each in `dim_session`; `parent_session_key`, `parent_agent_id` and "
      "`spawn_depth` carry the delegation tree)")
    w(f"- Time span: {first} to {last}")
    for kind, n, ok in runs:
        w(f"- ETL runs of kind `{kind}`: {n} ({ok} succeeded)")
    w("")
    w("## Where to start")
    w("")
    w("- `main` holds only what a consumer should query: dimensions (`dim_*`), facts "
      "(`fact_*`, `bridge_*`) and views (`semantic_*`). Machinery lives in the `etl` "
      "schema: `etl.runs`, `etl.batch_runs`, `etl.steps` (with `table_name` on every "
      "step that wrote a table), `etl.versions`, `etl.schema_version`, "
      "`etl.table_coverage`, `etl.audit_exceptions`, and the transient staging table "
      "`etl.log_entries`.")
    w("- Every fact carries `session_id` as a degenerate dimension and `session_key` "
      "as the FK, so `WHERE session_id = ...` works without a join.")
    w("- Every fact row carries a lineage block: `created_at`, `last_updated_at`, "
      "`created_by_version_key` / `last_updated_by_version_key` (`<ccutils "
      "version>/<business rules version>`, readable as-is), `etl_run_id`, "
      "`record_source`, `hash_diff`, `is_deleted`, `deleted_at`. Filter "
      "`is_deleted = FALSE`; nothing is ever hard-deleted.")
    w("- Subagent identity comes from the file name (`agent-<id>`), never from the "
      "`sessionId` inside an agent transcript, which carries the parent's.")
    w("")
    w("## Tables")
    w("")
    w("Status is declared beside the DDL and seeded into `etl.table_coverage`; "
      "`written` says whether a successful step in this file named the table; rows "
      "are live rows now.")
    w("")
    w("| table | status | rows | written | populated by | what it is |")
    w("|---|---|---|---|---|---|")
    for name, kind, status, by, reason in coverage:
        if kind != "table":
            continue
        n = _live_count(conn, name)
        w(f"| {name} | {status} | {n} | {'yes' if name in written else 'no'} | {by} | {reason} |")
    w("")
    w("## Views")
    w("")
    w("A view exists only when it encodes something a consumer would get wrong: a "
      "derivation, an aggregation, a resolution. Views marked `delete` are plain "
      "joins scheduled for removal.")
    w("")
    w("| view | status | why |")
    w("|---|---|---|")
    for name, kind, status, by, reason in coverage:
        if kind == "view":
            w(f"| {name} | {status} | {reason} |")
    w("")
    w("## Join paths")
    w("")
    w("Foreign keys are by column-name convention; these are the ones present in this file.")
    w("")
    tables = [name for name, kind, *_ in coverage if kind == "table"]
    dims = {name for name in tables if name.startswith("dim_")}
    for table in tables:
        cols = [r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()]
        for col in cols:
            target = _KEY_TARGETS.get(col)
            if target is None or target[0] not in dims:
                continue
            if table == target[0] and col == target[1]:
                continue
            w(f"- `{table}.{col}` -> `{target[0]}.{target[1]}`")
    w("")
    w("## Accepted audit findings")
    w("")
    if exceptions:
        w("These are known and accepted, each with a reason. `ccutils audit` reports "
          "them as allowed, not failed.")
        w("")
        for check, obj, detail, reason in exceptions:
            where = f"{obj}.{detail}" if detail else obj
            w(f"- `{check}` on `{where}`: {reason}")
    else:
        w("None.")
    w("")
    w("## What is not in here")
    w("")
    w("- Reasoning text: thinking blocks are persisted by Claude Code with an empty "
      "`thinking` field, so `has_thinking` counts blocks whose content the source "
      "never wrote.")
    w("- Bash-driven file operations: `fact_file_operations` records the file tools "
      "only; writes through `python3 -` heredocs or `cat >` are not derived yet.")
    w("- Large tool outputs: when Claude Code spills a result to `tool-results/`, the "
      "transcript keeps a preview and a path; the body is not ingested.")
    w("- Anything from projects not listed above: a warehouse is scoped, not scrubbed.")
    w("")
    return "\n".join(out)


def write_guide(conn, directory) -> str:
    """Render the guide and write it as README.md beside the archive."""
    from pathlib import Path

    path = Path(directory) / GUIDE_NAME
    path.write_text(render_guide(conn))
    return str(path)
