"""ETL functions for simple schema.

This module provides functions to export session data to DuckDB and JSON
using the simple 4-table schema.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ...parsers.jsonl_reader import iter_loglines, iter_session_entries
from ...sanitize import PathSanitizer
from ..star.extractors import estimate_tokens
from .schema import create_duckdb_schema


@dataclass
class SimpleExtractionResult:
    """Shared extraction result for both DuckDB and JSON export paths."""

    session_id: str = ""
    project_path: str = ""
    project_name: str = ""
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    first_timestamp_raw: str = ""
    last_timestamp_raw: str = ""
    cwd: str | None = None
    git_branch: str | None = None
    version: str | None = None
    is_agent: bool = False
    agent_id: str | None = None
    parent_session_id: str | None = None

    messages: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    thinking_blocks: list[dict] = field(default_factory=list)

    user_message_count: int = 0
    assistant_message_count: int = 0
    tool_use_count: int = 0
    total_estimated_tokens: int = 0

    # Remaining tool uses that never got a result
    orphan_tool_uses: list[dict] = field(default_factory=list)


def _extract_session_core(
    session_path,
    include_thinking=False,
    truncate_output=2000,
    private=False,
    loglines=None,
    session_id_override=None,
    project_name=None,
):
    """Core extraction logic shared by DuckDB and JSON export paths.

    Args:
        session_path: Path to the JSONL session file (ignored if loglines provided)
        include_thinking: Whether to export thinking blocks
        truncate_output: Max characters for tool output (default 2000)
        private: If True, sanitize paths to remove sensitive directory info
        loglines: Optional pre-parsed logline dicts (skips file reading)
        session_id_override: Optional session ID (used with loglines instead of path.stem)
        project_name: Optional project name override

    Returns:
        SimpleExtractionResult with all extracted data
    """
    if loglines is not None:
        entries_iter = iter_loglines(loglines)
        session_path_str = "claude.ai"
    else:
        session_path = Path(session_path)
        entries_iter = iter_session_entries(session_path)
        session_path_str = str(session_path)

    result = SimpleExtractionResult()
    result.project_path = session_path_str
    result.project_name = project_name or (
        session_path.parent.name if hasattr(session_path, "parent") else ""
    )

    sanitizer = None
    tool_use_map = {}
    thinking_id = 0
    is_first = True

    for entry in entries_iter:
        if entry.entry_type == "progress":
            continue

        # Extract metadata from first entry
        if is_first:
            is_first = False
            result.session_id = session_id_override or (
                session_path.stem if hasattr(session_path, "stem") else "unknown"
            )
            result.cwd = entry.raw.get("cwd")
            result.git_branch = entry.raw.get("gitBranch")
            result.version = entry.raw.get("version")
            result.agent_id = entry.raw.get("agentId")
            result.is_agent = result.agent_id is not None
            if result.is_agent:
                result.parent_session_id = entry.raw.get("sessionId")

            if private:
                sanitizer = PathSanitizer(result.cwd)
                result.cwd = sanitizer.sanitize_cwd()
                result.project_path = sanitizer.sanitize_project_path(
                    result.project_path
                )

        uuid = entry.uuid
        parent_uuid = entry.parent_uuid
        timestamp = entry.timestamp
        timestamp_raw = entry.timestamp_raw
        content = entry.content
        model = entry.model

        if timestamp is not None:
            if result.first_timestamp is None:
                result.first_timestamp = timestamp
                result.first_timestamp_raw = timestamp_raw
            result.last_timestamp = timestamp
            result.last_timestamp_raw = timestamp_raw

        # Extract content blocks
        has_tool_use = False
        has_tool_result = False
        has_thinking = False
        text_content = ""

        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")

                if block_type == "text":
                    text_parts.append(block.get("text", ""))

                elif block_type == "tool_use":
                    has_tool_use = True
                    result.tool_use_count += 1
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})

                    input_json_str = json.dumps(tool_input)
                    result.total_estimated_tokens += estimate_tokens(input_json_str)

                    if sanitizer:
                        input_json_str = sanitizer.sanitize_json_string(input_json_str)
                        tool_input = json.loads(input_json_str)

                    input_summary = input_json_str[:truncate_output]

                    tool_use_map[tool_id] = {
                        "tool_use_id": tool_id,
                        "session_id": result.session_id,
                        "message_id": uuid,
                        "tool_name": tool_name,
                        "input_json": tool_input,
                        "input_json_str": input_json_str,
                        "input_summary": input_summary,
                        "timestamp": timestamp,
                        "timestamp_raw": timestamp_raw,
                    }

                elif block_type == "tool_result":
                    has_tool_result = True
                    tool_id = block.get("tool_use_id", "")
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        output_text = result_content[:truncate_output]
                    else:
                        output_text = str(result_content)[:truncate_output]

                    result.total_estimated_tokens += estimate_tokens(
                        result_content
                        if isinstance(result_content, str)
                        else str(result_content)
                    )

                    if sanitizer:
                        output_text = sanitizer.sanitize_text(output_text)

                    if tool_id in tool_use_map:
                        tool_info = tool_use_map.pop(tool_id)
                        tool_info["result_message_id"] = uuid
                        tool_info["output_text"] = output_text
                        result.tool_calls.append(tool_info)

                elif block_type == "thinking":
                    has_thinking = True
                    thinking_text = block.get("thinking", "")
                    result.total_estimated_tokens += estimate_tokens(thinking_text)
                    if include_thinking:
                        thinking_id += 1
                        result.thinking_blocks.append(
                            {
                                "id": thinking_id,
                                "session_id": result.session_id,
                                "message_id": uuid,
                                "thinking_text": thinking_text,
                                "timestamp": timestamp,
                                "timestamp_raw": timestamp_raw,
                            }
                        )

            text_content = " ".join(text_parts)

        # Estimate tokens for text content (thinking and tool I/O estimated above)
        result.total_estimated_tokens += estimate_tokens(text_content)

        # Count messages
        if entry.entry_type == "user":
            result.user_message_count += 1
        else:
            result.assistant_message_count += 1

        # Serialize content for storage
        content_json = json.dumps(content) if isinstance(content, list) else None
        if sanitizer and content_json:
            content_json = sanitizer.sanitize_json_string(content_json)

        result.messages.append(
            {
                "id": uuid,
                "session_id": result.session_id,
                "parent_id": parent_uuid,
                "type": entry.entry_type,
                "timestamp": timestamp,
                "timestamp_raw": timestamp_raw,
                "model": model,
                "content": text_content,
                "content_json": content_json,
                "has_tool_use": has_tool_use,
                "has_tool_result": has_tool_result,
                "has_thinking": has_thinking,
                "is_sidechain": entry.is_sidechain,
            }
        )

    # Collect any remaining tool uses (no result received)
    for tool_info in tool_use_map.values():
        tool_info["result_message_id"] = None
        tool_info["output_text"] = None
        result.orphan_tool_uses.append(tool_info)

    return result


def export_session_to_duckdb(
    conn,
    session_path,
    project_name,
    include_thinking=False,
    truncate_output=2000,
    loglines=None,
    session_id_override=None,
    private=False,
):
    """Export a single session to DuckDB.

    Args:
        conn: DuckDB connection
        session_path: Path to the JSONL session file (ignored if loglines provided)
        project_name: Name of the project
        include_thinking: Whether to export thinking blocks
        truncate_output: Max characters for tool output (default 2000)
        loglines: Optional pre-parsed logline dicts (skips file reading)
        session_id_override: Optional session ID (used with loglines instead of path.stem)
        private: If True, sanitize paths to remove sensitive directory info
    """
    result = _extract_session_core(
        session_path,
        include_thinking=include_thinking,
        truncate_output=truncate_output,
        private=private,
        loglines=loglines,
        session_id_override=session_id_override,
        project_name=project_name,
    )

    if not result.session_id:
        return

    # Insert tool calls
    for tc in result.tool_calls:
        conn.execute(
            """
            INSERT INTO tool_calls (
                tool_use_id, session_id, message_id,
                result_message_id, tool_name, input_json,
                input_summary, output_text, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                tc["tool_use_id"],
                result.session_id,
                tc["message_id"],
                tc["result_message_id"],
                tc["tool_name"],
                tc["input_json_str"],
                tc["input_summary"],
                tc["output_text"],
                tc["timestamp"],
            ],
        )

    # Insert orphan tool uses (tool_use with no matching tool_result)
    for tc in result.orphan_tool_uses:
        conn.execute(
            """
            INSERT INTO tool_calls (
                tool_use_id, session_id, message_id,
                result_message_id, tool_name, input_json,
                input_summary, output_text, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                tc["tool_use_id"],
                result.session_id,
                tc["message_id"],
                None,
                tc["tool_name"],
                tc["input_json_str"],
                tc["input_summary"],
                None,
                tc["timestamp"],
            ],
        )

    # Insert thinking blocks
    for tb in result.thinking_blocks:
        conn.execute(
            """
            INSERT INTO thinking (id, session_id, message_id, thinking_text, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """,
            [
                tb["id"],
                result.session_id,
                tb["message_id"],
                tb["thinking_text"],
                tb["timestamp"],
            ],
        )

    # Insert messages
    for msg in result.messages:
        conn.execute(
            """
            INSERT INTO messages (
                id, session_id, parent_id, type, timestamp, model,
                content, content_json, has_tool_use, has_tool_result, has_thinking,
                is_sidechain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                msg["id"],
                result.session_id,
                msg["parent_id"],
                msg["type"],
                msg["timestamp"],
                msg["model"],
                msg["content"],
                msg["content_json"],
                msg["has_tool_use"],
                msg["has_tool_result"],
                msg["has_thinking"],
                msg["is_sidechain"],
            ],
        )

    # Insert session metadata
    message_count = result.user_message_count + result.assistant_message_count
    conn.execute(
        """
        INSERT INTO sessions (
            session_id, project_path, project_name, first_timestamp, last_timestamp,
            message_count, user_message_count, assistant_message_count,
            tool_use_count, estimated_tokens, cwd, git_branch, version,
            is_agent, agent_id, parent_session_id, depth_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        [
            result.session_id,
            result.project_path,
            result.project_name,
            result.first_timestamp,
            result.last_timestamp,
            message_count,
            result.user_message_count,
            result.assistant_message_count,
            result.tool_use_count,
            result.total_estimated_tokens,
            result.cwd,
            result.git_branch,
            result.version,
            result.is_agent,
            result.agent_id,
            result.parent_session_id,
            0,  # depth_level - will be set by multi-session export
        ],
    )


def _extract_session_data(
    session_path, include_thinking=False, truncate_output=2000, private=False
):
    """Extract session data from a JSONL file.

    Args:
        session_path: Path to the JSONL session file
        include_thinking: Whether to include thinking blocks
        truncate_output: Max characters for tool output
        private: If True, sanitize paths to remove sensitive directory info

    Returns:
        dict with session, messages, tool_calls, thinking keys
    """
    result = _extract_session_core(
        session_path,
        include_thinking=include_thinking,
        truncate_output=truncate_output,
        private=private,
    )

    session_meta = {
        "session_id": result.session_id,
        "project_name": result.project_name,
        "project_path": result.project_path,
        "cwd": result.cwd,
        "git_branch": result.git_branch,
        "version": result.version,
        "first_timestamp": result.first_timestamp_raw or None,
        "last_timestamp": result.last_timestamp_raw or None,
        "message_count": result.user_message_count + result.assistant_message_count,
        "user_message_count": result.user_message_count,
        "assistant_message_count": result.assistant_message_count,
        "tool_use_count": result.tool_use_count,
        "estimated_tokens": result.total_estimated_tokens,
        "is_agent": result.is_agent,
        "agent_id": result.agent_id,
        "parent_session_id": result.parent_session_id,
    }

    # Convert messages to use raw timestamps for JSON serialization
    messages = []
    for msg in result.messages:
        messages.append(
            {
                "id": msg["id"],
                "session_id": msg["session_id"],
                "parent_id": msg["parent_id"],
                "type": msg["type"],
                "timestamp": msg["timestamp_raw"],
                "model": msg["model"],
                "content": msg["content"],
                "content_json": msg["content_json"],
                "has_tool_use": msg["has_tool_use"],
                "has_tool_result": msg["has_tool_result"],
                "has_thinking": msg["has_thinking"],
                "is_sidechain": msg["is_sidechain"],
            }
        )

    # Convert tool calls to use raw timestamps and original input format
    tool_calls = []
    for tc in result.tool_calls:
        tool_calls.append(
            {
                "tool_use_id": tc["tool_use_id"],
                "session_id": tc["session_id"],
                "message_id": tc["message_id"],
                "result_message_id": tc["result_message_id"],
                "tool_name": tc["tool_name"],
                "input_json": tc["input_json"],
                "input_summary": tc["input_summary"],
                "output_text": tc["output_text"],
                "timestamp": tc["timestamp_raw"],
            }
        )

    # Add orphan tool uses (no result received)
    for tc in result.orphan_tool_uses:
        tool_calls.append(
            {
                "tool_use_id": tc["tool_use_id"],
                "session_id": tc["session_id"],
                "message_id": tc["message_id"],
                "result_message_id": None,
                "tool_name": tc["tool_name"],
                "input_json": tc["input_json"],
                "input_summary": tc["input_summary"],
                "output_text": None,
                "timestamp": tc["timestamp_raw"],
            }
        )

    # Convert thinking blocks to use raw timestamps
    thinking_blocks = []
    for tb in result.thinking_blocks:
        thinking_blocks.append(
            {
                "id": tb["id"],
                "session_id": tb["session_id"],
                "message_id": tb["message_id"],
                "thinking_text": tb["thinking_text"],
                "timestamp": tb["timestamp_raw"],
            }
        )

    return {
        "session": session_meta,
        "messages": messages,
        "tool_calls": tool_calls,
        "thinking": thinking_blocks,
    }


def export_sessions_to_json(
    session_paths,
    output_path,
    include_thinking=False,
    truncate_output=2000,
    private=False,
):
    """Export sessions to JSON format (simple schema).

    Args:
        session_paths: List of paths to JSONL session files
        output_path: Path for output JSON file
        include_thinking: Whether to include thinking blocks
        truncate_output: Max characters for tool output (default 2000)
        private: If True, sanitize paths to remove sensitive directory info
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sessions = []
    messages = []
    tool_calls = []
    thinking_blocks = []

    for session_path in session_paths:
        session_data = _extract_session_data(
            session_path, include_thinking, truncate_output, private=private
        )
        sessions.append(session_data["session"])
        messages.extend(session_data["messages"])
        tool_calls.extend(session_data["tool_calls"])
        thinking_blocks.extend(session_data["thinking"])

    result = {
        "version": "1.0",
        "schema_type": "simple",
        "exported_at": datetime.now().astimezone().isoformat(),
        "tables": {
            "sessions": sessions,
            "messages": messages,
            "tool_calls": tool_calls,
            "thinking": thinking_blocks,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
