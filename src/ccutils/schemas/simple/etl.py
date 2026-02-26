"""ETL functions for simple schema.

This module provides functions to export session data to DuckDB and JSON
using the simple 4-table schema.
"""

import json
from datetime import datetime
from pathlib import Path

from ...parsers.jsonl_reader import iter_loglines, iter_session_entries
from .schema import create_duckdb_schema


def export_session_to_duckdb(
    conn,
    session_path,
    project_name,
    include_thinking=False,
    truncate_output=2000,
    loglines=None,
    session_id_override=None,
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
    """
    if loglines is not None:
        entries_iter = iter_loglines(loglines)
        session_path_str = "claude.ai"
    else:
        session_path = Path(session_path)
        entries_iter = iter_session_entries(session_path)
        session_path_str = str(session_path)
    session_id = None
    cwd = None
    git_branch = None
    version = None
    first_timestamp = None
    last_timestamp = None
    user_count = 0
    assistant_count = 0
    tool_use_count = 0

    # Agent metadata
    is_agent = False
    agent_id = None
    parent_session_id = None

    # Maps to link tool_use to tool_result
    tool_use_map = {}
    thinking_id = 0

    for entry in entries_iter:
        if entry.entry_type == "progress":
            continue

        # Extract metadata from first entry
        if session_id is None:
            session_id = session_id_override or (
                session_path.stem if hasattr(session_path, "stem") else "unknown"
            )
            cwd = entry.raw.get("cwd")
            git_branch = entry.raw.get("gitBranch")
            version = entry.raw.get("version")
            agent_id = entry.raw.get("agentId")
            is_agent = agent_id is not None
            if is_agent:
                parent_session_id = entry.raw.get("sessionId")

        uuid = entry.uuid
        parent_uuid = entry.parent_uuid
        timestamp = entry.timestamp
        content = entry.content
        model = entry.model

        if timestamp is not None:
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

        # Extract content
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
                    tool_use_count += 1
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})

                    if isinstance(tool_input, dict):
                        input_summary = json.dumps(tool_input)[:truncate_output]
                    else:
                        input_summary = str(tool_input)[:truncate_output]

                    tool_use_map[tool_id] = {
                        "message_id": uuid,
                        "tool_name": tool_name,
                        "input_json": json.dumps(tool_input),
                        "input_summary": input_summary,
                        "timestamp": timestamp,
                    }

                elif block_type == "tool_result":
                    has_tool_result = True
                    tool_id = block.get("tool_use_id", "")
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        output_text = result_content[:truncate_output]
                    else:
                        output_text = str(result_content)[:truncate_output]

                    if tool_id in tool_use_map:
                        tool_info = tool_use_map[tool_id]
                        conn.execute(
                            """
                            INSERT INTO tool_calls (
                                tool_use_id, session_id, message_id,
                                result_message_id, tool_name, input_json,
                                input_summary, output_text, timestamp
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            [
                                tool_id,
                                session_id,
                                tool_info["message_id"],
                                uuid,
                                tool_info["tool_name"],
                                tool_info["input_json"],
                                tool_info["input_summary"],
                                output_text,
                                tool_info["timestamp"],
                            ],
                        )

                elif block_type == "thinking":
                    has_thinking = True
                    if include_thinking:
                        thinking_text = block.get("thinking", "")
                        thinking_id += 1
                        conn.execute(
                            """
                            INSERT INTO thinking (id, session_id, message_id, thinking_text, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """,
                            [
                                thinking_id,
                                session_id,
                                uuid,
                                thinking_text,
                                timestamp,
                            ],
                        )

            text_content = " ".join(text_parts)

        # Count messages
        if entry.entry_type == "user":
            user_count += 1
        else:
            assistant_count += 1

        # Insert message
        conn.execute(
            """
            INSERT INTO messages (
                id, session_id, parent_id, type, timestamp, model,
                content, content_json, has_tool_use, has_tool_result, has_thinking,
                is_sidechain
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                uuid,
                session_id,
                parent_uuid,
                entry.entry_type,
                timestamp,
                model,
                text_content,
                json.dumps(content) if isinstance(content, list) else None,
                has_tool_use,
                has_tool_result,
                has_thinking,
                entry.is_sidechain,
            ],
        )

    # Insert session metadata
    if session_id:
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, project_path, project_name, first_timestamp, last_timestamp,
                message_count, user_message_count, assistant_message_count,
                tool_use_count, cwd, git_branch, version,
                is_agent, agent_id, parent_session_id, depth_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                session_id,
                session_path_str,
                project_name,
                first_timestamp,
                last_timestamp,
                user_count + assistant_count,
                user_count,
                assistant_count,
                tool_use_count,
                cwd,
                git_branch,
                version,
                is_agent,
                agent_id,
                parent_session_id,
                0,  # depth_level - will be set by multi-session export
            ],
        )


def _extract_session_data(session_path, include_thinking=False, truncate_output=2000):
    """Extract session data from a JSONL file.

    Args:
        session_path: Path to the JSONL session file
        include_thinking: Whether to include thinking blocks
        truncate_output: Max characters for tool output

    Returns:
        dict with session, messages, tool_calls, thinking keys
    """
    session_path = Path(session_path)
    session_id = session_path.stem
    project_name = session_path.parent.name

    session_meta = {
        "session_id": session_id,
        "project_name": project_name,
        "project_path": str(session_path),
        "cwd": None,
        "git_branch": None,
        "version": None,
        "first_timestamp": None,
        "last_timestamp": None,
        "message_count": 0,
        "user_message_count": 0,
        "assistant_message_count": 0,
        "tool_use_count": 0,
        "is_agent": False,
        "agent_id": None,
        "parent_session_id": None,
    }

    messages = []
    tool_calls = []
    thinking_blocks = []
    tool_use_map = {}  # tool_use_id -> tool info
    thinking_id = 0
    is_first = True

    for entry in iter_session_entries(session_path):
        if entry.entry_type == "progress":
            continue

        # Extract metadata from first entry
        if is_first:
            is_first = False
            session_meta["cwd"] = entry.raw.get("cwd")
            session_meta["git_branch"] = entry.raw.get("gitBranch")
            session_meta["version"] = entry.raw.get("version")
            agent_id = entry.raw.get("agentId")
            session_meta["agent_id"] = agent_id
            session_meta["is_agent"] = agent_id is not None
            if agent_id:
                session_meta["parent_session_id"] = entry.raw.get("sessionId")

        uuid = entry.uuid
        parent_uuid = entry.parent_uuid
        timestamp_str = entry.timestamp_raw
        content = entry.content
        model = entry.model
        is_sidechain = entry.is_sidechain

        # Track timestamps
        if entry.timestamp is not None:
            if session_meta["first_timestamp"] is None:
                session_meta["first_timestamp"] = timestamp_str
            session_meta["last_timestamp"] = timestamp_str

        # Extract content
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
                    session_meta["tool_use_count"] += 1
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})

                    if isinstance(tool_input, dict):
                        input_summary = json.dumps(tool_input)[:truncate_output]
                    else:
                        input_summary = str(tool_input)[:truncate_output]

                    tool_use_map[tool_id] = {
                        "tool_use_id": tool_id,
                        "session_id": session_id,
                        "message_id": uuid,
                        "tool_name": tool_name,
                        "input_json": tool_input,
                        "input_summary": input_summary,
                        "timestamp": timestamp_str,
                    }

                elif block_type == "tool_result":
                    has_tool_result = True
                    tool_id = block.get("tool_use_id", "")
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        output_text = result_content[:truncate_output]
                    else:
                        output_text = str(result_content)[:truncate_output]

                    if tool_id in tool_use_map:
                        tool_info = tool_use_map[tool_id]
                        tool_info["result_message_id"] = uuid
                        tool_info["output_text"] = output_text
                        tool_calls.append(tool_info)
                        del tool_use_map[tool_id]

                elif block_type == "thinking":
                    has_thinking = True
                    if include_thinking:
                        thinking_text = block.get("thinking", "")
                        thinking_id += 1
                        thinking_blocks.append(
                            {
                                "id": thinking_id,
                                "session_id": session_id,
                                "message_id": uuid,
                                "thinking_text": thinking_text,
                                "timestamp": timestamp_str,
                            }
                        )

            text_content = " ".join(text_parts)

        # Count messages
        if entry.entry_type == "user":
            session_meta["user_message_count"] += 1
        else:
            session_meta["assistant_message_count"] += 1

        messages.append(
            {
                "id": uuid,
                "session_id": session_id,
                "parent_id": parent_uuid,
                "type": entry.entry_type,
                "timestamp": timestamp_str,
                "model": model,
                "content": text_content,
                "content_json": content if isinstance(content, list) else None,
                "has_tool_use": has_tool_use,
                "has_tool_result": has_tool_result,
                "has_thinking": has_thinking,
                "is_sidechain": is_sidechain,
            }
        )

    # Add any remaining tool uses (no result received)
    for tool_info in tool_use_map.values():
        tool_info["result_message_id"] = None
        tool_info["output_text"] = None
        tool_calls.append(tool_info)

    session_meta["message_count"] = (
        session_meta["user_message_count"] + session_meta["assistant_message_count"]
    )

    return {
        "session": session_meta,
        "messages": messages,
        "tool_calls": tool_calls,
        "thinking": thinking_blocks,
    }


def export_sessions_to_json(
    session_paths, output_path, include_thinking=False, truncate_output=2000
):
    """Export sessions to JSON format (simple schema).

    Args:
        session_paths: List of paths to JSONL session files
        output_path: Path for output JSON file
        include_thinking: Whether to include thinking blocks
        truncate_output: Max characters for tool output (default 2000)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sessions = []
    messages = []
    tool_calls = []
    thinking_blocks = []

    for session_path in session_paths:
        session_data = _extract_session_data(
            session_path, include_thinking, truncate_output
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
