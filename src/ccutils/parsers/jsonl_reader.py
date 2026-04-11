"""Shared JSONL session file reader.

Provides a single canonical parser for Claude Code session JSONL files.
All consumers (simple ETL, star ETL, HTML export) should use this
instead of implementing their own file-reading loops.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


@dataclass
class SessionMetaHeader:
    """Metadata extracted from the first entry of a session file."""

    session_id: str
    cwd: str | None = None
    git_branch: str | None = None
    version: str | None = None
    slug: str | None = None
    agent_id: str | None = None
    parent_session_id: str | None = None
    is_sidechain: bool = False
    entrypoint: str | None = None

    @property
    def is_agent(self) -> bool:
        return self.agent_id is not None


@dataclass
class SessionEntry:
    """A single parsed entry from a session JSONL file.

    Represents one line from the file with normalized fields.
    Only user/assistant/progress entries are yielded.
    """

    entry_type: str  # "user", "assistant", or "progress"
    uuid: str = ""
    parent_uuid: str | None = None
    timestamp_raw: str = ""
    timestamp: datetime | None = None
    model: str | None = None
    message_data: dict = field(default_factory=dict)
    content: str | list | None = None
    is_compact_summary: bool = False
    is_meta: bool = False
    is_sidechain: bool = False

    # Progress-specific fields
    progress_parent_tool_id: str | None = None
    progress_agent_id: str | None = None

    # New fields from recent Claude Code versions
    entrypoint: str | None = None
    usage: dict | None = None

    # Raw entry for consumers that need additional fields
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class SessionSystemEntry:
    """A system entry from a session JSONL file (turn_duration, stop_hook_summary, etc.)."""

    subtype: str
    timestamp: datetime | None = None
    data: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class SessionAttachment:
    """An attachment entry from a session JSONL file (diagnostics, hook_success, etc.)."""

    attachment_type: str
    timestamp: datetime | None = None
    data: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class SessionMetaEntry:
    """Session-level metadata entry (custom-title, agent-name, permission-mode)."""

    meta_type: str
    value: str = ""
    raw: dict = field(default_factory=dict, repr=False)


def parse_session_header(path: str | Path) -> SessionMetaHeader | None:
    """Parse just the header metadata from a session file.

    Reads only the first entry to extract session-level metadata.
    Returns None if the file is empty or unreadable.
    """
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                agent_id = obj.get("agentId")
                parent_session_id = None
                if agent_id is not None:
                    parent_session_id = obj.get("sessionId")

                return SessionMetaHeader(
                    session_id=path.stem,
                    cwd=obj.get("cwd"),
                    git_branch=obj.get("gitBranch"),
                    version=obj.get("version"),
                    slug=obj.get("slug"),
                    agent_id=agent_id,
                    parent_session_id=parent_session_id,
                    is_sidechain=obj.get("isSidechain", False),
                    entrypoint=obj.get("entrypoint"),
                )
    except OSError:
        pass
    return None


def _parse_timestamp(ts_str: str) -> datetime | None:
    """Parse an ISO timestamp string, handling Z suffix."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def iter_loglines(loglines: list[dict]) -> Iterator[SessionEntry]:
    """Convert pre-parsed logline dicts into SessionEntry objects.

    Used for Claude.ai export data that has already been parsed into
    the ccutils logline format (with type, uuid, timestamp, message keys).

    Args:
        loglines: List of logline dicts in ccutils format

    Yields:
        SessionEntry for each user/assistant entry
    """
    for obj in loglines:
        entry_type = obj.get("type")
        if entry_type not in ("user", "assistant"):
            continue

        message_data = obj.get("message", {})
        content = message_data.get("content", "")
        model = message_data.get("model") if isinstance(message_data, dict) else None
        ts_raw = obj.get("timestamp", "")

        yield SessionEntry(
            entry_type=entry_type,
            uuid=obj.get("uuid", ""),
            parent_uuid=obj.get("parentUuid"),
            timestamp_raw=ts_raw,
            timestamp=_parse_timestamp(ts_raw),
            model=model,
            message_data=message_data if isinstance(message_data, dict) else {},
            content=content,
            is_compact_summary=obj.get("isCompactSummary", False),
            is_meta=obj.get("isMeta", False),
            is_sidechain=obj.get("isSidechain", False),
            raw=obj,
        )


def iter_session_entries(path: str | Path) -> Iterator[SessionEntry]:
    """Iterate over entries in a session JSONL file.

    Yields SessionEntry objects for each user, assistant, or progress entry.
    Skips malformed lines and non-message entries (summary, system, etc.).

    The first yielded entry will have header metadata available in its raw dict.

    Args:
        path: Path to the JSONL session file

    Yields:
        SessionEntry for each relevant line in the file
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

            entry_type = obj.get("type")

            # Handle progress records
            if entry_type == "progress":
                parent_tool_id = obj.get("parentToolUseID")
                agent_data = obj.get("data", {})
                progress_agent_id = (
                    agent_data.get("agentId") if isinstance(agent_data, dict) else None
                )
                if parent_tool_id and progress_agent_id:
                    yield SessionEntry(
                        entry_type="progress",
                        raw=obj,
                        progress_parent_tool_id=parent_tool_id,
                        progress_agent_id=progress_agent_id,
                    )
                continue

            if entry_type not in ("user", "assistant"):
                continue

            message_data = obj.get("message", {})
            content = message_data.get("content", "")
            model = (
                message_data.get("model") if isinstance(message_data, dict) else None
            )
            ts_raw = obj.get("timestamp", "")

            yield SessionEntry(
                entry_type=entry_type,
                uuid=obj.get("uuid", ""),
                parent_uuid=obj.get("parentUuid"),
                timestamp_raw=ts_raw,
                timestamp=_parse_timestamp(ts_raw),
                model=model,
                message_data=message_data if isinstance(message_data, dict) else {},
                content=content,
                is_compact_summary=obj.get("isCompactSummary", False),
                is_meta=obj.get("isMeta", False),
                is_sidechain=obj.get("isSidechain", False),
                raw=obj,
            )


# Mapping from meta entry type to the field that holds the value
_META_VALUE_FIELDS = {
    "custom-title": "customTitle",
    "agent-name": "agentName",
    "permission-mode": "permissionMode",
}


def iter_all_session_entries(
    path: str | Path,
) -> Iterator[SessionEntry | SessionSystemEntry | SessionAttachment | SessionMetaEntry]:
    """Iterate over ALL entries in a session JSONL file.

    Unlike iter_session_entries() which only yields user/assistant/progress,
    this function also yields system entries, attachments, and metadata entries.

    Args:
        path: Path to the JSONL session file

    Yields:
        SessionEntry, SessionSystemEntry, SessionAttachment, or SessionMetaEntry
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

            entry_type = obj.get("type")

            # System entries (turn_duration, stop_hook_summary, bridge_status)
            if entry_type == "system":
                subtype = obj.get("subtype", "")
                if subtype:
                    ts_raw = obj.get("timestamp", "")
                    yield SessionSystemEntry(
                        subtype=subtype,
                        timestamp=_parse_timestamp(ts_raw),
                        data=obj,
                        raw=obj,
                    )
                continue

            # Attachment entries (diagnostics, hook_success, etc.)
            if entry_type == "attachment":
                attachment_data = obj.get("attachment", {})
                att_type = attachment_data.get("type", "")
                if att_type:
                    ts_raw = obj.get("timestamp", "")
                    yield SessionAttachment(
                        attachment_type=att_type,
                        timestamp=_parse_timestamp(ts_raw),
                        data=attachment_data,
                        raw=obj,
                    )
                continue

            # Meta entries (custom-title, agent-name, permission-mode)
            if entry_type in _META_VALUE_FIELDS:
                value_field = _META_VALUE_FIELDS[entry_type]
                yield SessionMetaEntry(
                    meta_type=entry_type,
                    value=obj.get(value_field, ""),
                    raw=obj,
                )
                continue

            # Progress records (same as iter_session_entries)
            if entry_type == "progress":
                parent_tool_id = obj.get("parentToolUseID")
                agent_data = obj.get("data", {})
                progress_agent_id = (
                    agent_data.get("agentId") if isinstance(agent_data, dict) else None
                )
                if parent_tool_id and progress_agent_id:
                    yield SessionEntry(
                        entry_type="progress",
                        raw=obj,
                        progress_parent_tool_id=parent_tool_id,
                        progress_agent_id=progress_agent_id,
                    )
                continue

            # User/assistant messages
            if entry_type not in ("user", "assistant"):
                continue

            message_data = obj.get("message", {})
            content = message_data.get("content", "")
            model = (
                message_data.get("model") if isinstance(message_data, dict) else None
            )
            ts_raw = obj.get("timestamp", "")

            # Extract usage from assistant messages
            usage = None
            if entry_type == "assistant" and isinstance(message_data, dict):
                usage = message_data.get("usage")

            yield SessionEntry(
                entry_type=entry_type,
                uuid=obj.get("uuid", ""),
                parent_uuid=obj.get("parentUuid"),
                timestamp_raw=ts_raw,
                timestamp=_parse_timestamp(ts_raw),
                model=model,
                message_data=message_data if isinstance(message_data, dict) else {},
                content=content,
                is_compact_summary=obj.get("isCompactSummary", False),
                is_meta=obj.get("isMeta", False),
                is_sidechain=obj.get("isSidechain", False),
                entrypoint=obj.get("entrypoint"),
                usage=usage,
                raw=obj,
            )
