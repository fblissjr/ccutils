"""Import Claude Code's prompt history (history.jsonl) into dim_prompt.

Not part of run_v15_etl -- history.jsonl is a global file, not per-session,
so this populator is called explicitly (e.g. by `ccutils --source` after the
per-session loop). Idempotent: prompt_key = md5(display_text || iso_timestamp)
is stable across reloads, so re-importing the same history.jsonl produces
no duplicates.

session_key is set when the history entry carries a sessionId and that
session_id is present in dim_session; otherwise NULL.
"""

from __future__ import annotations

from pathlib import Path

from ccutils.etl.utils import insert_missing_dim_dates
from ccutils.parsers.history import iter_history_entries


def _project_dir_name(project_path: str) -> str:
    """Claude Code's encoding of a real path: `/home/user/repo` -> `-home-user-repo`."""
    return project_path.replace("/", "-")


def _covers_project(conn):
    """Predicate: does this warehouse cover the project a prompt came from?

    Matched two ways, both EXACT. `dim_session.cwd` is a real path, which is
    what a history entry carries. `dim_project.project_name` is Claude Code's
    dashed encoding of one, so the entry's path is encoded FORWARD and
    compared -- never decoded backward, because the encoding is lossy:
    `-home-user-fb-claude-skills` decodes to `/home/user/fb/claude/skills`, which
    is a different, nonexistent path. Encoding forward is exact for names
    containing dashes; decoding is not.

    No prefix or substring matching, per the resolver rule this project
    already follows: a near-match invents a relationship the data never
    stated, and here it would leak a neighbouring project's prompts into a
    scoped artifact.
    """
    cwds = {
        row[0] for row in conn.execute(
            "SELECT DISTINCT cwd FROM dim_session WHERE cwd IS NOT NULL"
        ).fetchall()
    }
    encoded = {
        row[0] for row in conn.execute(
            "SELECT DISTINCT project_name FROM dim_project "
            "WHERE project_name IS NOT NULL"
        ).fetchall()
    }

    def covers(project_path: str | None) -> bool:
        if not project_path:
            return False
        return project_path in cwds or _project_dir_name(project_path) in encoded

    return covers


def import_history(conn, history_path: str | Path, *,
                   only_projects: bool = False) -> int:
    """Load history.jsonl into dim_prompt. Returns the row count inserted.

    Missing or unreadable files are a no-op (returns 0) so callers can
    treat history ingestion as best-effort.

    ``only_projects``: keep only prompts belonging to projects this warehouse
    already covers. history.jsonl is MACHINE-WIDE, so importing it unscoped
    put every prompt the user had ever typed into every warehouse -- measured
    2026-08-28, a one-session warehouse held 11,606 prompts from 103
    projects. That contradicts the rule that a shared artifact is scoped
    rather than scrubbed.

    Default False so a full-corpus build (which covers everything anyway)
    keeps taking the lot and cross-project analysis is unaffected; the
    scoping only bites where the user asked for a subset. An empty covered
    set with ``only_projects=True`` imports NOTHING, never everything --
    reading an empty scope as "unfiltered" would invert the whole point.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        # Still reconcile dim_date for any dim_prompt rows a prior import
        # left behind (an existing warehouse whose history.jsonl has since
        # been rotated/deleted must not keep NULL prompt dates forever).
        _backfill_prompt_dates(conn)
        return 0

    covers = _covers_project(conn) if only_projects else None

    rows = []
    for entry in iter_history_entries(history_path):
        if covers is not None and not covers(entry.project_path):
            continue
        # Stable natural key: prompt + exact timestamp. Re-ingesting the
        # same line produces the same key.
        ts_iso = entry.timestamp.isoformat() if entry.timestamp else ""
        prompt_key_input = f"{entry.display}|{ts_iso}"
        rows.append(
            (
                prompt_key_input,
                entry.session_id,
                entry.project_path,
                entry.project_name,
                entry.display,
                entry.timestamp,
                entry.has_pasted_content,
            )
        )

    if not rows:
        _backfill_prompt_dates(conn)
        return 0

    conn.execute("DROP TABLE IF EXISTS _inbound_prompts")
    conn.execute(
        """
        CREATE TEMP TABLE _inbound_prompts (
            prompt_key_input VARCHAR,
            session_id VARCHAR,
            project_path VARCHAR,
            project_name VARCHAR,
            display_text VARCHAR,
            timestamp TIMESTAMP,
            has_pasted_content BOOLEAN
        )
        """
    )
    conn.executemany(
        "INSERT INTO _inbound_prompts VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    inserted = conn.execute(
        """
        INSERT INTO dim_prompt (
            prompt_key, session_key, project_path, project_name,
            display_text, timestamp, date_key, time_key, has_pasted_content
        )
        SELECT
            md5(ip.prompt_key_input) AS prompt_key,
            -- Resolve session_key via dim_session lookup. Absent => NULL.
            (SELECT ds.session_key FROM dim_session ds
             WHERE ds.session_id = ip.session_id LIMIT 1) AS session_key,
            ip.project_path,
            ip.project_name,
            ip.display_text,
            ip.timestamp,
            CASE WHEN ip.timestamp IS NOT NULL
                THEN CAST(strftime(ip.timestamp, '%Y%m%d') AS INTEGER)
                ELSE NULL END AS date_key,
            CASE WHEN ip.timestamp IS NOT NULL
                THEN CAST(strftime(ip.timestamp, '%H%M') AS INTEGER)
                ELSE NULL END AS time_key,
            ip.has_pasted_content
        FROM _inbound_prompts ip
        WHERE NOT EXISTS (
            SELECT 1 FROM dim_prompt dp
            WHERE dp.prompt_key = md5(ip.prompt_key_input)
        )
        """
    ).rowcount

    conn.execute("DROP TABLE IF EXISTS _inbound_prompts")
    _backfill_prompt_dates(conn)
    return inserted or 0


def _backfill_prompt_dates(conn) -> None:
    """Ensure dim_date has a row for every dim_prompt date.

    history.jsonl carries dates no staged session covers; without their
    dim_date rows semantic_prompt_history returns NULL full_date. Scans
    the whole dim_prompt table (a small, once-per-archive dimension, not a
    per-session fact -- the populator-scoping rule targets facts), so it
    also repairs prompt rows loaded by an older ccutils.
    """
    insert_missing_dim_dates(conn, "dim_prompt", "timestamp")
