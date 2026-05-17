# path-privacy: skip-file -- references universal Claude Code data paths (not personal)
"""ETL for loading ~/.claude/history.jsonl into dim_prompt."""

from pathlib import Path

from ...parsers.history import iter_history_entries
from .utils import ensure_dim_date, generate_dimension_key, ts_to_date_key, ts_to_time_key


def load_history(conn, history_path, private=False):
    """Load history.jsonl into dim_prompt table.

    Links prompts to sessions via sessionId where available.
    Idempotent: clears dim_prompt before repopulating.

    Args:
        conn: DuckDB connection with star schema
        history_path: Path to history.jsonl file
        private: If True, sanitize project paths
    """
    history_path = Path(history_path)
    if not history_path.exists():
        return

    conn.execute("DELETE FROM dim_prompt")

    # Build session_id -> session_key lookup from existing dim_session
    session_lookup = {}
    try:
        rows = conn.execute(
            "SELECT session_id, session_key FROM dim_session"
        ).fetchall()
        session_lookup = {r[0]: r[1] for r in rows}
    except Exception:
        pass

    dates_seen = set()

    batch = []
    for entry in iter_history_entries(history_path):
        session_key = None
        if entry.session_id:
            session_key = session_lookup.get(entry.session_id)
            if session_key is None:
                session_key = generate_dimension_key(entry.session_id)

        date_key = None
        time_key = None
        if entry.timestamp:
            date_key = ts_to_date_key(entry.timestamp)
            time_key = ts_to_time_key(entry.timestamp)
            dates_seen.add(date_key)

        project_path = entry.project_path
        project_name = entry.project_name
        if private and project_path:
            project_path = project_name

        prompt_key = generate_dimension_key(
            entry.session_id or "",
            str(entry.timestamp) if entry.timestamp else "",
            entry.display[:50],
        )

        batch.append((
            prompt_key,
            session_key,
            project_path,
            project_name,
            entry.display,
            entry.timestamp,
            date_key,
            time_key,
            entry.has_pasted_content,
        ))

    if batch:
        conn.executemany(
            """INSERT INTO dim_prompt
               (prompt_key, session_key, project_path, project_name,
                display_text, timestamp, date_key, time_key, has_pasted_content)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            batch,
        )

    for date_key in dates_seen:
        ensure_dim_date(conn, date_key)
