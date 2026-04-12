"""ETL for loading ~/.claude/history.jsonl into dim_prompt."""

from pathlib import Path

from ...parsers.history import iter_history_entries
from .utils import generate_dimension_key, get_time_of_day


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

    # Collect date/time keys to insert into dim tables
    dates_seen = set()

    batch = []
    for entry in iter_history_entries(history_path):
        session_key = None
        if entry.session_id:
            session_key = session_lookup.get(entry.session_id)
            # If not in lookup, generate the key anyway -- it may link later
            if session_key is None and entry.session_id:
                session_key = generate_dimension_key(entry.session_id)

        date_key = None
        time_key = None
        if entry.timestamp:
            date_key = int(entry.timestamp.strftime("%Y%m%d"))
            time_key = int(entry.timestamp.strftime("%H%M"))
            dates_seen.add(date_key)

        project_path = entry.project_path
        project_name = entry.project_name
        if private and project_path:
            # Just use project name, hide full path
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

    # Bulk insert
    if batch:
        conn.executemany(
            """INSERT INTO dim_prompt
               (prompt_key, session_key, project_path, project_name,
                display_text, timestamp, date_key, time_key, has_pasted_content)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            batch,
        )

    # Ensure date/time dimensions exist for history entries
    from datetime import datetime

    day_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    for date_key in dates_seen:
        if not conn.execute(
            "SELECT 1 FROM dim_date WHERE date_key = ?", [date_key]
        ).fetchone():
            year = date_key // 10000
            month = (date_key // 100) % 100
            day = date_key % 100
            try:
                full_date = datetime(year, month, day)
                day_of_week = full_date.weekday()
                quarter = (month - 1) // 3 + 1
                is_weekend = day_of_week >= 5
                week_of_year = full_date.isocalendar()[1]
                conn.execute(
                    "INSERT INTO dim_date VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        date_key, full_date.date(), year, month, day,
                        day_of_week, day_names[day_of_week], month_names[month - 1],
                        quarter, is_weekend, week_of_year,
                    ],
                )
            except ValueError:
                pass
