"""Parser for ~/.claude/history.jsonl -- global prompt history.  # path-privacy: ignore

Each line contains a user prompt with project context and optional session link.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass
class HistoryEntry:
    """A single entry from history.jsonl."""

    display: str
    project_path: str | None = None
    project_name: str | None = None
    session_id: str | None = None
    timestamp: datetime | None = None
    has_pasted_content: bool = False


def iter_history_entries(path: str | Path) -> Iterator[HistoryEntry]:
    """Iterate over entries in a history.jsonl file.

    Args:
        path: Path to the history.jsonl file

    Yields:
        HistoryEntry for each valid line
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            display = obj.get("display", "")
            if not display:
                continue

            project_path = obj.get("project") or None
            project_name = None
            if project_path:
                project_name = Path(project_path).name or None

            ts_ms = obj.get("timestamp")
            timestamp = None
            if ts_ms and isinstance(ts_ms, (int, float)):
                try:
                    timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                except (ValueError, OSError):
                    pass

            pasted = obj.get("pastedContents", {})
            has_pasted = bool(pasted) if isinstance(pasted, dict) else False

            yield HistoryEntry(
                display=display,
                project_path=project_path,
                project_name=project_name,
                session_id=obj.get("sessionId"),
                timestamp=timestamp,
                has_pasted_content=has_pasted,
            )
