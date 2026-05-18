"""Import Claude Code's prompt history (history.jsonl) into dim_prompt.

Not part of run_v15_etl -- history.jsonl is a global file, not per-session,
so this populator is called explicitly (e.g. by `ccutils all` after the
per-session loop). Idempotent: prompt_key = md5(display_text || iso_timestamp)
is stable across reloads, so re-importing the same history.jsonl produces
no duplicates.

session_key is set when the history entry carries a sessionId and that
session_id is present in dim_session; otherwise NULL.
"""

from __future__ import annotations

from pathlib import Path

from ccutils.parsers.history import iter_history_entries


def import_history(conn, history_path: str | Path) -> int:
    """Load history.jsonl into dim_prompt. Returns the row count inserted.

    Missing or unreadable files are a no-op (returns 0) so callers can
    treat history ingestion as best-effort.
    """
    history_path = Path(history_path)
    if not history_path.exists():
        return 0

    rows = []
    for entry in iter_history_entries(history_path):
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
    return inserted or 0
