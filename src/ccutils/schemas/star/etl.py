"""ETL pipeline for loading session data into star schema."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ...parsers.jsonl_reader import iter_session_entries
from ...sanitize import PathSanitizer
from .extractors import (
    calculate_conversation_depth,
    count_words,
    detect_language_from_extension,
    estimate_tokens,
    extract_code_blocks,
    extract_entities,
    extract_file_info,
    extract_file_path_from_tool,
    get_operation_type,
)
from .heuristics import (
    classify_complexity,
    classify_domain,
    classify_error_type,
    classify_intent,
    classify_outcome,
)
from .utils import (
    generate_dimension_key,
    get_model_family,
    get_time_of_day,
    get_tool_category,
)


@dataclass
class StarExtractionResult:
    """Bundles all data extracted from a session file for star schema loading."""

    # Session metadata
    cwd: str | None = None
    git_branch: str | None = None
    version: str | None = None
    slug: str | None = None
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None

    # Agent metadata
    is_agent: bool = False
    agent_id: str | None = None
    parent_session_id: str | None = None

    # Counters
    user_count: int = 0
    assistant_count: int = 0
    total_content_blocks: int = 0
    thinking_count: int = 0

    # Data lists
    messages_data: list = field(default_factory=list)
    content_blocks_data: list = field(default_factory=list)
    tool_calls_data: list = field(default_factory=list)
    file_operations_data: list = field(default_factory=list)
    code_blocks_data: list = field(default_factory=list)
    errors_data: list = field(default_factory=list)
    entity_mentions_data: list = field(default_factory=list)
    tool_chain_data: list = field(default_factory=list)
    tool_input_params_data: list = field(default_factory=list)

    # Dimension tracking sets
    tools_seen: set = field(default_factory=set)
    models_seen: set = field(default_factory=set)
    dates_seen: set = field(default_factory=set)

    # Dimension tracking dicts
    files_seen: dict = field(default_factory=dict)
    task_agent_map: dict = field(default_factory=dict)

    # Token estimation breakdown
    thinking_estimated_tokens: int = 0
    tool_io_estimated_tokens: int = 0

    # For heuristic classification
    first_user_message: str | None = None
    last_assistant_message: str | None = None
    file_extensions_seen: list = field(default_factory=list)
    max_depth: int = 0


def run_star_schema_etl(
    conn,
    session_path,
    project_name,
    include_thinking=False,
    truncate_output=2000,
    private=False,
):
    """Run ETL to populate star schema from a session file.

    This function:
    1. Extracts raw data from the session file
    2. Transforms and loads dimension tables (with deduplication)
    3. Transforms and loads fact tables
    4. Runs heuristic classification on the session

    Args:
        conn: DuckDB connection
        session_path: Path to the JSONL session file
        project_name: Name of the project
        include_thinking: Whether to include thinking blocks
        truncate_output: Max characters for tool output (default 2000)
        private: If True, sanitize paths to remove sensitive directory info
    """
    session_path = Path(session_path)
    session_id = session_path.stem
    session_key = generate_dimension_key(session_id)
    project_path = str(session_path.parent)
    project_key = generate_dimension_key(project_path)

    result = _extract_star_data(
        session_path,
        session_key,
        project_key,
        include_thinking,
        truncate_output,
        private,
    )

    # Create sanitizer for dimension loading (display values only)
    sanitizer = PathSanitizer(result.cwd) if private else None

    _load_dimensions(
        conn,
        project_key,
        project_path,
        project_name,
        session_key,
        session_id,
        result,
        sanitizer,
    )
    _load_facts(conn, session_key, project_key, result)


@dataclass
class BlockContext:
    """Per-message state passed to block handler functions."""

    message_id: str
    session_key: str
    date_key: int | None
    time_key: int | None
    timestamp: datetime | None
    truncate_output: int
    idx: int  # block index
    sanitizer: PathSanitizer | None = None


def _handle_tool_use_block(result, block, ctx, tool_use_map, prev_tool_call):
    """Handle a tool_use content block during extraction.

    Populates tool_use_map, tool_input_params, file operations, and tool chain data.
    Returns updated prev_tool_call tuple.
    """
    tool_use_id = block.get("id")
    tool_name = block.get("name", "unknown")
    tool_input = block.get("input", {})
    result.tools_seen.add(tool_name)

    input_json = json.dumps(tool_input)
    result.tool_io_estimated_tokens += estimate_tokens(input_json)

    # Extract common parameters for direct columns
    extracted_file_path = extract_file_path_from_tool(tool_name, tool_input)
    extracted_command = (
        tool_input.get("command") if tool_name.lower() == "bash" else None
    )
    extracted_pattern = (
        tool_input.get("pattern") if tool_name.lower() == "grep" else None
    )
    extracted_query = tool_input.get("query") or tool_input.get("url")

    # Sanitize paths if private mode
    if ctx.sanitizer:
        input_json = ctx.sanitizer.sanitize_json_string(input_json)
        if extracted_file_path:
            extracted_file_path = ctx.sanitizer.sanitize_path(extracted_file_path)
        if extracted_command:
            extracted_command = ctx.sanitizer.sanitize_text(extracted_command)

    input_summary = input_json[: ctx.truncate_output]

    tool_use_map[tool_use_id] = {
        "message_id": ctx.message_id,
        "tool_name": tool_name,
        "tool_key": generate_dimension_key(tool_name),
        "input_json": input_json,
        "input_summary": input_summary,
        "input_char_count": len(input_json),
        "timestamp": ctx.timestamp,
        "date_key": ctx.date_key,
        "time_key": ctx.time_key,
        "file_path": extracted_file_path,
        "command": extracted_command,
        "pattern": extracted_pattern,
        "query_text": extracted_query,
    }

    # Populate tool_input_params for granular exploration
    for param_key, param_value in tool_input.items():
        if param_value is None:
            continue
        param_id = f"{tool_use_id}-{param_key}"
        param_text = None
        param_number = None
        param_bool = None

        if isinstance(param_value, bool):
            param_bool = param_value
        elif isinstance(param_value, (int, float)):
            param_number = float(param_value)
        elif isinstance(param_value, str):
            param_text = param_value[:2000]
            if ctx.sanitizer:
                param_text = ctx.sanitizer.sanitize_text(param_text)
        else:
            param_text = json.dumps(param_value)[:2000]
            if ctx.sanitizer:
                param_text = ctx.sanitizer.sanitize_json_string(param_text)

        result.tool_input_params_data.append(
            {
                "param_id": param_id,
                "tool_call_id": tool_use_id,
                "session_key": ctx.session_key,
                "param_key": param_key,
                "param_value_text": param_text,
                "param_value_number": param_number,
                "param_value_bool": param_bool,
            }
        )

    # Track file operations (use sanitized path for display, original for keys)
    file_path = extract_file_path_from_tool(tool_name, tool_input)
    if file_path:
        file_info = extract_file_info(file_path)
        if file_info and file_path not in result.files_seen:
            result.files_seen[file_path] = file_info
            # Track file extensions for domain classification
            ext = file_info.get("file_extension", "")
            if ext:
                result.file_extensions_seen.append(ext)

        operation_type = get_operation_type(tool_name)
        file_content = tool_input.get("content", "")
        file_size = len(file_content) if isinstance(file_content, str) else 0

        result.file_operations_data.append(
            {
                "file_operation_id": f"{tool_use_id}-file",
                "tool_call_id": tool_use_id,
                "session_key": ctx.session_key,
                "file_key": file_info["file_key"] if file_info else None,
                "tool_key": generate_dimension_key(tool_name),
                "date_key": ctx.date_key,
                "time_key": ctx.time_key,
                "operation_type": operation_type,
                "file_size_chars": file_size,
                "timestamp": ctx.timestamp,
            }
        )

    # Track tool chain
    tool_key = generate_dimension_key(tool_name)
    chain_id = f"{ctx.session_key}-chain"
    step_position = len(result.tool_chain_data)

    time_since_prev = None
    prev_tool_key_val = None
    if prev_tool_call and ctx.timestamp:
        prev_ts = prev_tool_call[2]
        if prev_ts:
            time_since_prev = (ctx.timestamp - prev_ts).total_seconds()
        prev_tool_key_val = prev_tool_call[1]

    result.tool_chain_data.append(
        {
            "chain_step_id": f"{chain_id}-{step_position}",
            "session_key": ctx.session_key,
            "chain_id": chain_id,
            "tool_call_id": tool_use_id,
            "tool_key": tool_key,
            "step_position": step_position,
            "prev_tool_key": prev_tool_key_val,
            "next_tool_key": None,  # backfilled after extraction
            "is_error": False,  # backfilled after tool_result
            "time_since_prev_seconds": time_since_prev,
        }
    )

    _add_content_block(
        result.content_blocks_data,
        ctx.message_id,
        ctx.session_key,
        "tool_use",
        ctx.idx,
        ctx.date_key,
        ctx.time_key,
        input_summary,
        ctx.truncate_output,
        block,
    )
    result.total_content_blocks += 1

    return (tool_use_id, tool_key, ctx.timestamp)


def _handle_tool_result_block(result, block, ctx, tool_use_map):
    """Handle a tool_result content block during extraction.

    Resolves tool_use_map entries, appends to tool_calls_data and errors_data.
    """
    tool_use_id = block.get("tool_use_id")
    result_content = block.get("content", "")
    is_error = block.get("is_error", False)

    if isinstance(result_content, list):
        result_text = " ".join(
            str(item.get("text", ""))
            for item in result_content
            if isinstance(item, dict)
        )
    else:
        result_text = str(result_content)

    if ctx.sanitizer:
        result_text = ctx.sanitizer.sanitize_text(result_text)

    output_text = result_text[: ctx.truncate_output]
    output_char_count = len(result_text)
    result.tool_io_estimated_tokens += estimate_tokens(result_text)

    if tool_use_id and tool_use_id in tool_use_map:
        tool_info = tool_use_map.pop(tool_use_id)

        # Calculate duration between invoke and result
        duration_seconds = None
        if tool_info["timestamp"] and ctx.timestamp:
            duration_seconds = (ctx.timestamp - tool_info["timestamp"]).total_seconds()

        result.tool_calls_data.append(
            {
                "tool_call_id": tool_use_id,
                "session_key": ctx.session_key,
                "tool_key": tool_info["tool_key"],
                "date_key": tool_info["date_key"],
                "time_key": tool_info["time_key"],
                "invoke_message_id": tool_info["message_id"],
                "result_message_id": ctx.message_id,
                "timestamp": tool_info["timestamp"],
                "input_char_count": tool_info["input_char_count"],
                "output_char_count": output_char_count,
                "is_error": is_error,
                "duration_seconds": duration_seconds,
                "input_json": tool_info["input_json"],
                "input_summary": tool_info["input_summary"],
                "output_text": output_text,
                "file_path": tool_info["file_path"],
                "command": tool_info["command"],
                "pattern": tool_info["pattern"],
                "query_text": tool_info["query_text"],
            }
        )

        if is_error:
            error_type = classify_error_type(output_text)
            result.errors_data.append(
                {
                    "error_id": f"{tool_use_id}-error",
                    "tool_call_id": tool_use_id,
                    "session_key": ctx.session_key,
                    "tool_key": tool_info["tool_key"],
                    "error_type": error_type,
                    "date_key": tool_info["date_key"],
                    "time_key": tool_info["time_key"],
                    "error_message": output_text,
                    "timestamp": tool_info["timestamp"],
                }
            )

            # Backfill is_error on the corresponding chain step
            for step in result.tool_chain_data:
                if step["tool_call_id"] == tool_use_id:
                    step["is_error"] = True
                    break

    _add_content_block(
        result.content_blocks_data,
        ctx.message_id,
        ctx.session_key,
        "tool_result",
        ctx.idx,
        ctx.date_key,
        ctx.time_key,
        output_text,
        ctx.truncate_output,
        block,
    )
    result.total_content_blocks += 1


def _extract_star_data(
    session_path,
    session_key,
    project_key,
    include_thinking,
    truncate_output,
    private=False,
):
    """Extract all data from a session file for star schema loading.

    Returns a StarExtractionResult with all extracted data ready for
    dimension and fact table loading.
    """
    result = StarExtractionResult()

    tool_use_map = {}
    message_timestamps = {}
    depth_map = {}
    prev_tool_call = None
    is_first = True
    sanitizer = None

    for entry in iter_session_entries(session_path):
        # Capture progress records for deterministic agent delegation linking
        if entry.entry_type == "progress":
            if entry.progress_parent_tool_id and entry.progress_agent_id:
                result.task_agent_map[entry.progress_parent_tool_id] = (
                    entry.progress_agent_id
                )
            continue

        # Extract metadata from first entry
        if is_first:
            is_first = False
            result.cwd = entry.raw.get("cwd")
            result.git_branch = entry.raw.get("gitBranch")
            result.version = entry.raw.get("version")
            result.slug = entry.raw.get("slug")
            result.agent_id = entry.raw.get("agentId")
            result.is_agent = result.agent_id is not None
            if result.is_agent:
                result.parent_session_id = entry.raw.get("sessionId")

            if private:
                sanitizer = PathSanitizer(result.cwd)

        message_id = entry.uuid
        parent_id = entry.parent_uuid
        timestamp = entry.timestamp
        model = entry.model
        content = entry.content

        # Derive date/time keys from parsed timestamp
        date_key = None
        time_key = None
        if timestamp is not None:
            if result.first_timestamp is None:
                result.first_timestamp = timestamp
            result.last_timestamp = timestamp
            date_key = int(timestamp.strftime("%Y%m%d"))
            time_key = int(timestamp.strftime("%H%M"))
            result.dates_seen.add(date_key)

        if model:
            result.models_seen.add(model)

        # Process content
        has_tool_use = False
        has_tool_result = False
        has_thinking = False
        text_content = ""
        content_json = json.dumps(content)
        content_block_count = 0
        msg_thinking_tokens = 0
        msg_tool_io_tokens = 0

        if isinstance(content, str):
            text_content = content
            content_block_count = 1
            block_id = f"{message_id}-0"
            result.content_blocks_data.append(
                {
                    "content_block_id": block_id,
                    "message_id": message_id,
                    "session_key": session_key,
                    "block_type": "text",
                    "date_key": date_key,
                    "time_key": time_key,
                    "block_index": 0,
                    "content_length": len(content),
                    "content_text": content[:truncate_output] if content else "",
                    "content_json": json.dumps({"type": "text", "text": content}),
                }
            )
            result.total_content_blocks += 1

        elif isinstance(content, list):
            texts = []
            msg_thinking_tokens = 0
            msg_tool_io_tokens = 0
            for idx, block in enumerate(content):
                if not isinstance(block, dict):
                    continue

                block_type = block.get("type")
                content_block_count += 1

                should_track = True
                if block_type == "thinking" and not include_thinking:
                    should_track = False

                ctx = BlockContext(
                    message_id=message_id,
                    session_key=session_key,
                    date_key=date_key,
                    time_key=time_key,
                    timestamp=timestamp,
                    truncate_output=truncate_output,
                    idx=idx,
                    sanitizer=sanitizer,
                )

                if block_type == "text":
                    text = block.get("text", "")
                    texts.append(text)
                    if should_track:
                        _add_content_block(
                            result.content_blocks_data,
                            message_id,
                            session_key,
                            "text",
                            idx,
                            date_key,
                            time_key,
                            text,
                            truncate_output,
                            block,
                        )
                        result.total_content_blocks += 1

                elif block_type == "tool_use":
                    has_tool_use = True
                    # Track tool I/O tokens before handler (which updates session accumulator)
                    tool_input_json = json.dumps(block.get("input", {}))
                    msg_tool_io_tokens += estimate_tokens(tool_input_json)
                    prev_tool_call = _handle_tool_use_block(
                        result, block, ctx, tool_use_map, prev_tool_call
                    )

                elif block_type == "tool_result":
                    has_tool_result = True
                    # Track tool result tokens before handler
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_text = " ".join(
                            str(item.get("text", ""))
                            for item in result_content
                            if isinstance(item, dict)
                        )
                    else:
                        result_text = str(result_content)
                    msg_tool_io_tokens += estimate_tokens(result_text)
                    _handle_tool_result_block(result, block, ctx, tool_use_map)

                elif block_type == "thinking":
                    has_thinking = True
                    result.thinking_count += 1
                    thinking_text = block.get("thinking", "")
                    thinking_tokens = estimate_tokens(thinking_text)
                    msg_thinking_tokens += thinking_tokens
                    result.thinking_estimated_tokens += thinking_tokens

                    if should_track:
                        _add_content_block(
                            result.content_blocks_data,
                            message_id,
                            session_key,
                            "thinking",
                            idx,
                            date_key,
                            time_key,
                            thinking_text,
                            truncate_output,
                            block,
                        )
                        result.total_content_blocks += 1

                elif block_type == "image":
                    if should_track:
                        result.content_blocks_data.append(
                            {
                                "content_block_id": f"{message_id}-{idx}",
                                "message_id": message_id,
                                "session_key": session_key,
                                "block_type": "image",
                                "date_key": date_key,
                                "time_key": time_key,
                                "block_index": idx,
                                "content_length": 0,
                                "content_text": "[image]",
                                "content_json": json.dumps(
                                    {"type": "image", "note": "content omitted"}
                                ),
                            }
                        )
                        result.total_content_blocks += 1

            text_content = " ".join(texts)

        # Extract code blocks
        if text_content:
            extracted_blocks = extract_code_blocks(text_content)
            for cb_idx, cb in enumerate(extracted_blocks):
                language = cb["language"]
                result.code_blocks_data.append(
                    {
                        "code_block_id": f"{message_id}-code-{cb_idx}",
                        "message_id": message_id,
                        "session_key": session_key,
                        "language": language,
                        "date_key": date_key,
                        "time_key": time_key,
                        "block_index": cb_idx,
                        "line_count": cb["line_count"],
                        "char_count": cb["char_count"],
                        "code_text": cb["code"][:truncate_output],
                    }
                )

            entities = extract_entities(text_content, message_id, session_key)
            result.entity_mentions_data.extend(entities)

        # Track first user message and last assistant message for heuristics
        if entry.entry_type == "user":
            result.user_count += 1
            if result.first_user_message is None and text_content:
                result.first_user_message = text_content
        else:
            result.assistant_count += 1
            if text_content:
                result.last_assistant_message = text_content

        word_cnt = count_words(text_content)
        text_token_est = estimate_tokens(text_content)
        token_est = text_token_est + msg_thinking_tokens + msg_tool_io_tokens

        response_time = None
        if parent_id and parent_id in message_timestamps and timestamp:
            parent_ts = message_timestamps[parent_id]
            if parent_ts:
                response_time = (timestamp - parent_ts).total_seconds()

        conversation_depth = calculate_conversation_depth(
            message_id, parent_id, depth_map
        )
        depth_map[message_id] = conversation_depth
        if conversation_depth > result.max_depth:
            result.max_depth = conversation_depth

        if timestamp:
            message_timestamps[message_id] = timestamp

        model_key = generate_dimension_key(model) if model else None

        result.messages_data.append(
            {
                "message_id": message_id,
                "session_key": session_key,
                "project_key": project_key,
                "message_type": entry.entry_type,
                "model_key": model_key,
                "date_key": date_key,
                "time_key": time_key,
                "parent_message_id": parent_id,
                "timestamp": timestamp,
                "content_length": len(text_content),
                "content_block_count": content_block_count,
                "has_tool_use": has_tool_use,
                "has_tool_result": has_tool_result,
                "has_thinking": has_thinking,
                "word_count": word_cnt,
                "estimated_tokens": token_est,
                "response_time_seconds": response_time,
                "conversation_depth": conversation_depth,
                "content_text": (
                    text_content[:truncate_output] if text_content else ""
                ),
                "content_json": content_json,
                "is_sidechain": entry.is_sidechain,
            }
        )

    # Backfill next_tool_key on chain steps
    for i in range(len(result.tool_chain_data) - 1):
        result.tool_chain_data[i]["next_tool_key"] = result.tool_chain_data[i + 1][
            "tool_key"
        ]

    # Include orphan tool uses (tool_use with no matching tool_result)
    for tool_use_id, tool_info in tool_use_map.items():
        result.tool_calls_data.append(
            {
                "tool_call_id": tool_use_id,
                "session_key": session_key,
                "tool_key": tool_info["tool_key"],
                "date_key": tool_info["date_key"],
                "time_key": tool_info["time_key"],
                "invoke_message_id": tool_info["message_id"],
                "result_message_id": None,
                "timestamp": tool_info["timestamp"],
                "input_char_count": tool_info["input_char_count"],
                "output_char_count": 0,
                "is_error": False,
                "duration_seconds": None,
                "input_json": tool_info["input_json"],
                "input_summary": tool_info["input_summary"],
                "output_text": None,
                "file_path": tool_info["file_path"],
                "command": tool_info["command"],
                "pattern": tool_info["pattern"],
                "query_text": tool_info["query_text"],
            }
        )

    return result


def _add_content_block(
    blocks_data,
    message_id,
    session_key,
    block_type,
    idx,
    date_key,
    time_key,
    content_text,
    truncate_output,
    block,
):
    """Helper to add a content block to the list."""
    blocks_data.append(
        {
            "content_block_id": f"{message_id}-{idx}",
            "message_id": message_id,
            "session_key": session_key,
            "block_type": block_type,
            "date_key": date_key,
            "time_key": time_key,
            "block_index": idx,
            "content_length": len(content_text) if content_text else 0,
            "content_text": content_text[:truncate_output] if content_text else "",
            "content_json": json.dumps(block),
        }
    )


def _load_dimensions(
    conn,
    project_key,
    project_path,
    project_name,
    session_key,
    session_id,
    result,
    sanitizer=None,
):
    """Load all dimension tables."""

    # dim_project (sanitize project_path for display)
    display_project_path = (
        sanitizer.sanitize_project_path(project_path) if sanitizer else project_path
    )
    if not conn.execute(
        "SELECT 1 FROM dim_project WHERE project_key = ?", [project_key]
    ).fetchone():
        conn.execute(
            "INSERT INTO dim_project VALUES (?, ?, ?)",
            [project_key, display_project_path, project_name],
        )

    # dim_session with heuristic classification
    parent_session_key = (
        generate_dimension_key(result.parent_session_id)
        if result.parent_session_id
        else None
    )
    depth_level = 0
    if result.is_agent and parent_session_key:
        parent_depth = conn.execute(
            "SELECT depth_level FROM dim_session WHERE session_key = ?",
            [parent_session_key],
        ).fetchone()
        if parent_depth is not None:
            depth_level = parent_depth[0] + 1

    display_cwd = sanitizer.sanitize_cwd() if sanitizer else result.cwd

    # Heuristic classification
    error_rate = (
        len(result.errors_data) / len(result.tool_calls_data)
        if result.tool_calls_data
        else 0.0
    )
    agent_depth = 0
    if result.is_agent:
        agent_depth = depth_level

    intent = classify_intent(result.first_user_message)
    complexity = classify_complexity(
        tool_count=len(result.tool_calls_data),
        msg_count=result.user_count + result.assistant_count,
        agent_depth=agent_depth,
        error_count=len(result.errors_data),
    )
    outcome = classify_outcome(result.last_assistant_message, error_rate=error_rate)
    domain = classify_domain(result.file_extensions_seen)

    if not conn.execute(
        "SELECT 1 FROM dim_session WHERE session_key = ?", [session_key]
    ).fetchone():
        conn.execute(
            """INSERT INTO dim_session
               (session_key, session_id, project_key, cwd, git_branch, version,
                slug, first_timestamp, last_timestamp, is_agent, agent_id,
                parent_session_key, depth_level, chain_key,
                intent, complexity, outcome, domain,
                first_user_message, last_assistant_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                session_key,
                session_id,
                project_key,
                display_cwd,
                result.git_branch,
                result.version,
                result.slug,
                result.first_timestamp,
                result.last_timestamp,
                result.is_agent,
                result.agent_id,
                parent_session_key,
                depth_level,
                None,  # chain_key - populated by batch export
                intent,
                complexity,
                outcome,
                domain,
                result.first_user_message[:500] if result.first_user_message else None,
                (
                    result.last_assistant_message[:500]
                    if result.last_assistant_message
                    else None
                ),
            ],
        )

    # dim_tool
    for tool_name in result.tools_seen:
        tool_key = generate_dimension_key(tool_name)
        if not conn.execute(
            "SELECT 1 FROM dim_tool WHERE tool_key = ?", [tool_key]
        ).fetchone():
            category = get_tool_category(tool_name)
            conn.execute(
                "INSERT INTO dim_tool VALUES (?, ?, ?)",
                [tool_key, tool_name, category],
            )

    # dim_model
    for model_name in result.models_seen:
        model_key = generate_dimension_key(model_name)
        if not conn.execute(
            "SELECT 1 FROM dim_model WHERE model_key = ?", [model_key]
        ).fetchone():
            family = get_model_family(model_name)
            conn.execute(
                "INSERT INTO dim_model VALUES (?, ?, ?)",
                [model_key, model_name, family],
            )

    # dim_date (with week_of_year)
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    for date_key in result.dates_seen:
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
                    """INSERT INTO dim_date VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        date_key,
                        full_date.date(),
                        year,
                        month,
                        day,
                        day_of_week,
                        day_names[day_of_week],
                        month_names[month - 1],
                        quarter,
                        is_weekend,
                        week_of_year,
                    ],
                )
            except ValueError:
                pass

    # dim_time
    times_seen = {msg["time_key"] for msg in result.messages_data if msg["time_key"]}
    for time_key in times_seen:
        if not conn.execute(
            "SELECT 1 FROM dim_time WHERE time_key = ?", [time_key]
        ).fetchone():
            hour = time_key // 100
            minute = time_key % 100
            time_of_day = get_time_of_day(hour)
            conn.execute(
                "INSERT INTO dim_time VALUES (?, ?, ?, ?)",
                [time_key, hour, minute, time_of_day],
            )

    # dim_file (with language inferred from extension)
    for _file_path, file_info in result.files_seen.items():
        if not conn.execute(
            "SELECT 1 FROM dim_file WHERE file_key = ?", [file_info["file_key"]]
        ).fetchone():
            display_file_path = (
                sanitizer.sanitize_path(file_info["file_path"])
                if sanitizer
                else file_info["file_path"]
            )
            display_dir_path = (
                sanitizer.sanitize_path(file_info["directory_path"])
                if sanitizer
                else file_info["directory_path"]
            )
            language = detect_language_from_extension(file_info["file_path"])
            conn.execute(
                "INSERT INTO dim_file VALUES (?, ?, ?, ?, ?, ?)",
                [
                    file_info["file_key"],
                    display_file_path,
                    file_info["file_name"],
                    file_info["file_extension"],
                    display_dir_path,
                    language,
                ],
            )


def _load_facts(conn, session_key, project_key, result):
    """Load all fact tables."""

    # fact_messages (message_type is now degenerate VARCHAR)
    for msg in result.messages_data:
        conn.execute(
            """INSERT INTO fact_messages
               (message_id, session_key, project_key, message_type, model_key,
                date_key, time_key, parent_message_id, timestamp, content_length,
                content_block_count, has_tool_use, has_tool_result, has_thinking,
                word_count, estimated_tokens, response_time_seconds, conversation_depth,
                content_text, content_json, is_sidechain)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                msg["message_id"],
                msg["session_key"],
                msg["project_key"],
                msg["message_type"],
                msg["model_key"],
                msg["date_key"],
                msg["time_key"],
                msg["parent_message_id"],
                msg["timestamp"],
                msg["content_length"],
                msg["content_block_count"],
                msg["has_tool_use"],
                msg["has_tool_result"],
                msg["has_thinking"],
                msg["word_count"],
                msg["estimated_tokens"],
                msg["response_time_seconds"],
                msg["conversation_depth"],
                msg["content_text"],
                msg["content_json"],
                msg["is_sidechain"],
            ],
        )

    # fact_content_blocks (block_type is now degenerate VARCHAR)
    for block in result.content_blocks_data:
        conn.execute(
            """INSERT INTO fact_content_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                block["content_block_id"],
                block["message_id"],
                block["session_key"],
                block["block_type"],
                block["date_key"],
                block["time_key"],
                block["block_index"],
                block["content_length"],
                block["content_text"],
                block["content_json"],
            ],
        )

    # fact_tool_calls (with duration_seconds)
    for tc in result.tool_calls_data:
        conn.execute(
            """INSERT INTO fact_tool_calls VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                tc["tool_call_id"],
                tc["session_key"],
                tc["tool_key"],
                tc["date_key"],
                tc["time_key"],
                tc["invoke_message_id"],
                tc["result_message_id"],
                tc["timestamp"],
                tc["input_char_count"],
                tc["output_char_count"],
                tc["is_error"],
                tc["duration_seconds"],
                tc["input_json"],
                tc["input_summary"],
                tc["output_text"],
                tc["file_path"],
                tc["command"],
                tc["pattern"],
                tc["query_text"],
            ],
        )

    # fact_session_summary (enhanced metrics)
    session_duration = 0
    if result.first_timestamp and result.last_timestamp:
        session_duration = int(
            (result.last_timestamp - result.first_timestamp).total_seconds()
        )

    first_date_key = None
    first_time_key = None
    if result.first_timestamp:
        first_date_key = int(result.first_timestamp.strftime("%Y%m%d"))
        first_time_key = int(result.first_timestamp.strftime("%H%M"))

    # Message-level tokens now include text + thinking + tool I/O per message,
    # so session total is just the sum across messages
    total_estimated_tokens = sum(
        msg.get("estimated_tokens", 0) for msg in result.messages_data
    )

    total_tool_calls = len(result.tool_calls_data)
    total_errors = len(result.errors_data)

    conn.execute(
        """INSERT INTO fact_session_summary VALUES
           (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            session_key,
            project_key,
            first_date_key,
            first_time_key,
            result.user_count + result.assistant_count,
            result.user_count,
            result.assistant_count,
            total_tool_calls,
            result.thinking_count,
            result.total_content_blocks,
            total_errors,
            len(result.tools_seen),
            len(result.files_seen),
            result.max_depth,
            total_estimated_tokens,
            result.thinking_estimated_tokens,
            result.tool_io_estimated_tokens,
            session_duration,
            result.first_timestamp,
            result.last_timestamp,
            # _incl_agents columns: initialized to own values,
            # rollup happens in finalize_star_schema()
            total_estimated_tokens,
            total_tool_calls,
            total_errors,
            session_duration,
        ],
    )

    # fact_file_operations
    for fop in result.file_operations_data:
        conn.execute(
            """INSERT INTO fact_file_operations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                fop["file_operation_id"],
                fop["tool_call_id"],
                fop["session_key"],
                fop["file_key"],
                fop["tool_key"],
                fop["date_key"],
                fop["time_key"],
                fop["operation_type"],
                fop["file_size_chars"],
                fop["timestamp"],
            ],
        )

    # fact_code_blocks (language is now degenerate VARCHAR)
    for cb in result.code_blocks_data:
        conn.execute(
            """INSERT INTO fact_code_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                cb["code_block_id"],
                cb["message_id"],
                cb["session_key"],
                cb["language"],
                cb["date_key"],
                cb["time_key"],
                cb["block_index"],
                cb["line_count"],
                cb["char_count"],
                cb["code_text"],
            ],
        )

    # fact_errors (error_type is now degenerate VARCHAR from heuristic)
    for err in result.errors_data:
        conn.execute(
            """INSERT INTO fact_errors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                err["error_id"],
                err["tool_call_id"],
                err["session_key"],
                err["tool_key"],
                err["error_type"],
                err["date_key"],
                err["time_key"],
                err["error_message"],
                err["timestamp"],
            ],
        )

    # fact_entity_mentions (entity_type is now degenerate VARCHAR)
    for em in result.entity_mentions_data:
        conn.execute(
            """INSERT INTO fact_entity_mentions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                em["mention_id"],
                em["message_id"],
                em["session_key"],
                em["entity_type"],
                em["entity_text"],
                em["entity_normalized"],
                em["context_snippet"],
                em["position_start"],
                em["position_end"],
            ],
        )

    # fact_tool_chain_steps (with next_tool_key and is_error)
    for tc in result.tool_chain_data:
        conn.execute(
            """INSERT INTO fact_tool_chain_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                tc["chain_step_id"],
                tc["session_key"],
                tc["chain_id"],
                tc["tool_call_id"],
                tc["tool_key"],
                tc["step_position"],
                tc["prev_tool_key"],
                tc["next_tool_key"],
                tc["is_error"],
                tc["time_since_prev_seconds"],
            ],
        )

    # fact_tool_input_params
    for param in result.tool_input_params_data:
        conn.execute(
            """INSERT INTO fact_tool_input_params VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                param["param_id"],
                param["tool_call_id"],
                param["session_key"],
                param["param_key"],
                param["param_value_text"],
                param["param_value_number"],
                param["param_value_bool"],
            ],
        )

    # stg_task_agent_map (progress record links: tool_use_id -> agent_id)
    if result.task_agent_map:
        for tool_use_id, agent_id in result.task_agent_map.items():
            conn.execute(
                "INSERT INTO stg_task_agent_map VALUES (?, ?, ?)",
                [tool_use_id, agent_id, session_key],
            )
