"""Write parsed JSONL session entries to Parquet (Tier 1 of the four-tier ETL).

Per-session layout:
    parquet_lake/projects/<project_slug>/sessions/<session_id>/
        log_entries.parquet     -- one row per JSONL line (typed envelope + JSON payloads)
        session_meta.parquet    -- one row of session-level metadata

Schema design:
- Envelope fields (session_id, uuid, parent_uuid, timestamp, type, etc.)
  are typed Parquet columns -- the columns analytical queries scan.
- Polymorphic per-entry-type payloads (message, toolUseResult, attachment,
  data, etc.) are stored as JSON columns. Forward-compat: new fields don't
  require schema migration.
- Lineage columns (etl_run_id, parsed_at, parser_version, record_source)
  stamp every row.
- model_extra fields (any unknown top-level keys Pydantic preserved) land
  in `extras_json` for full retention.

Trunc-and-reload safe: rerun overwrites the per-session Parquet files.
JSONL is the immutable audit log; Parquet is a typed-columnar cache.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from ccutils.parsers.models import (
    _Envelope,
    AssistantEntry,
    AttachmentEntry,
    PermissionModeEntry,
    ProgressEntry,
    SessionLogEntry,
    SystemEntry,
    UserEntry,
    parse_log_entry,
)
from ccutils.schemas.star.utils import generate_dimension_key


PARSER_VERSION = "0.15.0-dev"
RECORD_SOURCE_CLAUDE_CODE_JSONL = "claude_code_jsonl"


# Envelope columns kept as typed Parquet columns. Anything else lands in JSON.
_LOG_ENTRY_SCHEMA = pa.schema([
    # Provenance / lineage
    pa.field("etl_run_id", pa.string(), nullable=False),
    pa.field("parsed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("parser_version", pa.string(), nullable=False),
    pa.field("record_source", pa.string(), nullable=False),
    # Per-row identity
    pa.field("entry_id", pa.string(), nullable=False),
    pa.field("source_path", pa.string(), nullable=False),
    pa.field("sequence_num", pa.int32(), nullable=False),
    # Entry envelope
    pa.field("type", pa.string(), nullable=False),
    pa.field("uuid", pa.string()),
    pa.field("parent_uuid", pa.string()),
    pa.field("session_id", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("cwd", pa.string()),
    pa.field("git_branch", pa.string()),
    pa.field("slug", pa.string()),
    pa.field("version", pa.string()),
    pa.field("user_type", pa.string()),
    pa.field("entrypoint", pa.string()),
    pa.field("is_sidechain", pa.bool_()),
    pa.field("is_meta", pa.bool_()),
    pa.field("agent_id", pa.string()),
    # Polymorphic payloads (JSON; type-specific shape inside)
    pa.field("message_json", pa.string()),
    pa.field("tool_use_result_json", pa.string()),
    pa.field("attachment_json", pa.string()),
    pa.field("progress_data_json", pa.string()),
    pa.field("system_subtype", pa.string()),
    pa.field("system_payload_json", pa.string()),
    pa.field("meta_payload_json", pa.string()),  # for permission-mode / custom-title / etc.
    # Forward-compat: any extra top-level keys Pydantic preserved
    pa.field("extras_json", pa.string()),
    # Raw entry as JSON for full audit trail
    pa.field("raw_json", pa.string()),
])


_SESSION_META_SCHEMA = pa.schema([
    pa.field("etl_run_id", pa.string(), nullable=False),
    pa.field("parsed_at", pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("parser_version", pa.string(), nullable=False),
    pa.field("record_source", pa.string(), nullable=False),
    pa.field("session_id", pa.string(), nullable=False),
    pa.field("source_path", pa.string(), nullable=False),
    pa.field("source_size_bytes", pa.int64(), nullable=False),
    pa.field("entry_count", pa.int64(), nullable=False),
    pa.field("first_timestamp", pa.string()),
    pa.field("last_timestamp", pa.string()),
    pa.field("project_slug", pa.string()),
    pa.field("cwd", pa.string()),
    pa.field("git_branch", pa.string()),
    pa.field("agent_id", pa.string()),
])


def make_etl_run_id() -> str:
    """Generate a unique ETL run id (UUID4 hex)."""
    return uuid.uuid4().hex


# Envelope keys derived from the Pydantic _Envelope model + the polymorphic
# payload keys we project into separate JSON columns. Used to filter unknown
# meta entries' raw fields when building meta_payload_json.
_ENVELOPE_KEYS: set[str] = {
    f.alias or name for name, f in _Envelope.model_fields.items()
} | {"message", "toolUseResult", "attachment", "data", "subtype"}


def _entry_to_row(
    entry: SessionLogEntry,
    sequence_num: int,
    raw: dict[str, Any],
    source_path: str,
    etl_run_id: str,
    parsed_at: datetime,
    record_source: str,
) -> dict[str, Any]:
    """Project one Pydantic entry into a flat dict matching _LOG_ENTRY_SCHEMA."""
    # Envelope fields exist on all entry types via _Envelope; meta entries
    # may have many None'd. getattr keeps things defensive.
    base: dict[str, Any] = {
        "etl_run_id": etl_run_id,
        "parsed_at": parsed_at,
        "parser_version": PARSER_VERSION,
        "record_source": record_source,
        "entry_id": generate_dimension_key(source_path, sequence_num),
        "source_path": source_path,
        "sequence_num": sequence_num,
        "type": getattr(entry, "type", "unknown") or "unknown",
        "uuid": getattr(entry, "uuid", None),
        "parent_uuid": getattr(entry, "parent_uuid", None),
        "session_id": getattr(entry, "session_id", None),
        "timestamp": getattr(entry, "timestamp", None),
        "cwd": getattr(entry, "cwd", None),
        "git_branch": getattr(entry, "git_branch", None),
        "slug": getattr(entry, "slug", None),
        "version": getattr(entry, "version", None),
        "user_type": getattr(entry, "user_type", None),
        "entrypoint": getattr(entry, "entrypoint", None),
        "is_sidechain": getattr(entry, "is_sidechain", False),
        "is_meta": getattr(entry, "is_meta", False),
        "agent_id": getattr(entry, "agent_id", None),
        "message_json": None,
        "tool_use_result_json": None,
        "attachment_json": None,
        "progress_data_json": None,
        "system_subtype": None,
        "system_payload_json": None,
        "meta_payload_json": None,
        "extras_json": None,
        "raw_json": json.dumps(raw, ensure_ascii=False, default=str),
    }

    # Type-specific payload extraction
    if isinstance(entry, (UserEntry, AssistantEntry)):
        base["message_json"] = json.dumps(entry.message, ensure_ascii=False, default=str)
        if isinstance(entry, UserEntry) and entry.tool_use_result is not None:
            base["tool_use_result_json"] = json.dumps(
                entry.tool_use_result, ensure_ascii=False, default=str
            )
    elif isinstance(entry, AttachmentEntry):
        base["attachment_json"] = json.dumps(entry.attachment, ensure_ascii=False, default=str)
    elif isinstance(entry, ProgressEntry):
        base["progress_data_json"] = json.dumps(entry.data, ensure_ascii=False, default=str)
    elif isinstance(entry, SystemEntry):
        base["system_subtype"] = entry.subtype
        base["system_payload_json"] = json.dumps(raw, ensure_ascii=False, default=str)
    elif isinstance(entry, PermissionModeEntry):
        base["meta_payload_json"] = json.dumps(
            {"permission_mode": entry.permission_mode}, ensure_ascii=False, default=str
        )
    else:
        # custom-title / agent-name / last-prompt / queue-operation / etc.
        # Capture the whole raw entry minus envelope as the meta payload.
        meta_payload = {k: v for k, v in raw.items() if k not in _ENVELOPE_KEYS}
        if meta_payload:
            base["meta_payload_json"] = json.dumps(meta_payload, ensure_ascii=False, default=str)

    # Forward-compat extras (anything Pydantic put in model_extra not already projected)
    extras = entry.model_extra if hasattr(entry, "model_extra") else None
    if extras:
        base["extras_json"] = json.dumps(extras, ensure_ascii=False, default=str)

    return base


def _iter_raw_and_typed(jsonl_path: Path) -> Iterable[tuple[dict[str, Any], SessionLogEntry]]:
    """Yield (raw_dict, typed_entry) pairs in source order."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield raw, parse_log_entry(raw)


def write_session_to_parquet(
    jsonl_path: str | Path,
    output_root: str | Path,
    etl_run_id: str | None = None,
    project_slug: str | None = None,
    record_source: str = RECORD_SOURCE_CLAUDE_CODE_JSONL,
) -> tuple[Path, Path]:
    """Write one session's JSONL to per-session Parquet files.

    Args:
        jsonl_path: Source JSONL file (~/.claude/projects/<slug>/<session>.jsonl
                    or a nested subagent transcript path).
        output_root: parquet_lake root directory.
        etl_run_id: Caller-supplied ETL run id (UUID hex). Generates one
                    if omitted.
        project_slug: Project slug for partitioning. Inferred from the
                      jsonl_path's parent dir name if omitted.
        record_source: Provenance label stamped on every row.

    Returns:
        (log_entries_path, session_meta_path) -- the two Parquet files written.
    """
    jsonl_path = Path(jsonl_path)
    output_root = Path(output_root)
    if etl_run_id is None:
        etl_run_id = make_etl_run_id()
    parsed_at = datetime.now(tz=timezone.utc)

    if project_slug is None:
        project_slug = jsonl_path.parent.name

    session_id = jsonl_path.stem
    session_dir = output_root / "projects" / project_slug / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    log_entries_path = session_dir / "log_entries.parquet"
    session_meta_path = session_dir / "session_meta.parquet"

    rows: list[dict[str, Any]] = []
    first_ts: str | None = None
    last_ts: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    agent_id: str | None = None

    for sequence_num, (raw, typed) in enumerate(_iter_raw_and_typed(jsonl_path)):
        row = _entry_to_row(
            typed, sequence_num, raw, str(jsonl_path), etl_run_id, parsed_at, record_source,
        )
        rows.append(row)
        ts = row["timestamp"]
        if ts:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        if cwd is None:
            cwd = row["cwd"]
        if git_branch is None:
            git_branch = row["git_branch"]
        if agent_id is None:
            agent_id = row["agent_id"]

    if not rows:
        raise ValueError(f"No valid JSON log entries found in {jsonl_path}")

    # Subagent identity (real Claude Code contract): agent transcript
    # entries carry the PARENT's sessionId on every line, so raw capture
    # would leave log_entries.parquet disagreeing with session_meta.parquet
    # (whose session_id is the filename stem). Stamp rows with the file's
    # identity at Tier 1 so the lake is internally consistent; staging
    # keeps an equivalent override to repair lakes written before this.
    if re.search(r"/subagents/agent-[^/]+\.jsonl$", str(jsonl_path)):
        for row in rows:
            row["session_id"] = session_id

    # Write log_entries.parquet
    table = pa.Table.from_pylist(rows, schema=_LOG_ENTRY_SCHEMA)
    pq.write_table(table, log_entries_path, compression="zstd")

    # Write session_meta.parquet
    meta_row = {
        "etl_run_id": etl_run_id,
        "parsed_at": parsed_at,
        "parser_version": PARSER_VERSION,
        "record_source": record_source,
        "session_id": session_id,
        "source_path": str(jsonl_path),
        "source_size_bytes": jsonl_path.stat().st_size,
        "entry_count": len(rows),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "project_slug": project_slug,
        "cwd": cwd,
        "git_branch": git_branch,
        "agent_id": agent_id,
    }
    meta_table = pa.Table.from_pylist([meta_row], schema=_SESSION_META_SCHEMA)
    pq.write_table(meta_table, session_meta_path, compression="zstd")

    return log_entries_path, session_meta_path
