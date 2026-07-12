"""Markdown export for Claude Code session transcripts.

Renders one plain-markdown file per session: user/assistant headings,
tool calls as collapsible <details> blocks with fenced inputs and
truncated results, and thinking blocks as blockquotes. No templates,
no CSS -- plain string building on top of the same parsing entry
points the HTML exporter uses.
"""

import json
import re
from pathlib import Path

from ..parsers import (
    extract_rich_metadata,
    extract_text_from_content,
    find_all_sessions,
    get_session_summary,
    parse_session_file,
)
from ..parsers.session import extract_header_fields
from .html import _sanitize_loglines, is_tool_result_message

# Tool results longer than this are truncated inside the details block.
TOOL_RESULT_MAX_CHARS = 1500

# Tool input keys checked (in order) for the <summary> label.
_SUMMARY_INPUT_KEYS = ("file_path", "command", "path", "pattern", "url", "description")

_SUMMARY_VALUE_MAX_CHARS = 80

_BACKTICK_RUN = re.compile(r"`+")


def _fence(text):
    """Return a backtick fence longer than any backtick run in text."""
    longest = max((len(m.group()) for m in _BACKTICK_RUN.finditer(text)), default=0)
    return "`" * max(3, longest + 1)


def _escape_summary(text):
    """Escape HTML-sensitive characters for use inside <summary>."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _summary_label(tool_name, tool_input):
    """Build the <summary> label: tool name plus its key input, if any."""
    if isinstance(tool_input, dict):
        for key in _SUMMARY_INPUT_KEYS:
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                value = value.strip().splitlines()[0]
                if len(value) > _SUMMARY_VALUE_MAX_CHARS:
                    value = value[:_SUMMARY_VALUE_MAX_CHARS] + "..."
                return f"{tool_name}: {value}"
    return tool_name


def _result_text(result_block):
    """Flatten a tool_result content payload (str or block list) to text."""
    content = result_block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _render_thinking(thinking_text):
    """Render a thinking block as a blockquoted subsection."""
    lines = ["> **Thinking**", ">"]
    for line in thinking_text.strip().splitlines():
        lines.append(f"> {line}".rstrip())
    return "\n".join(lines)


def _render_tool_use(block, results_by_id):
    """Render a tool_use block (and its paired result) as a details block."""
    tool_name = block.get("name", "Unknown tool")
    tool_input = block.get("input", {})
    label = _escape_summary(_summary_label(tool_name, tool_input))

    input_json = json.dumps(tool_input, indent=2, ensure_ascii=False)
    input_fence = _fence(input_json)

    parts = [
        "<details>",
        f"<summary>{label}</summary>",
        "",
        f"{input_fence}json",
        input_json,
        input_fence,
    ]

    result_block = results_by_id.get(block.get("id"))
    if result_block is not None:
        result_text = _result_text(result_block).strip()
        if len(result_text) > TOOL_RESULT_MAX_CHARS:
            result_text = (
                result_text[:TOOL_RESULT_MAX_CHARS] + "\n... [truncated]"
            )
        result_label = (
            "Result (error):" if result_block.get("is_error") else "Result:"
        )
        result_fence = _fence(result_text)
        parts += [
            "",
            result_label,
            "",
            result_fence,
            result_text,
            result_fence,
        ]

    parts.append("</details>")
    return "\n".join(parts)


def _collect_tool_results(loglines):
    """Map tool_use_id -> tool_result block across the whole session."""
    results = {}
    for entry in loglines:
        content = entry.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if tool_use_id:
                    results[tool_use_id] = block
    return results


def _render_header(loglines, title, session_id=None):
    """Render the session header: title plus session id / date if present."""
    lines = [f"# {title}", ""]
    first_ts = None
    for entry in loglines:
        session_id = session_id or entry.get("sessionId")
        first_ts = first_ts or entry.get("timestamp")
        if session_id and first_ts:
            break
    if session_id:
        lines.append(f"- Session: `{session_id}`")
    if first_ts:
        lines.append(f"- Date: {first_ts}")
    if session_id or first_ts:
        lines.append("")
    return lines


def render_session_markdown(loglines, *, title, session_id=None, include_thinking=True):
    """Render parsed loglines to a markdown document string."""
    results_by_id = _collect_tool_results(loglines)
    lines = _render_header(loglines, title, session_id=session_id)

    for entry in loglines:
        log_type = entry.get("type")
        message = entry.get("message", {})
        if not message:
            continue

        if log_type == "user":
            # Tool-result carriers are rendered inside their tool's details
            # block, not as standalone user messages.
            if is_tool_result_message(message):
                continue
            text = extract_text_from_content(message.get("content", ""))
            if not text:
                continue
            lines += ["## User", "", text, ""]

        elif log_type == "assistant":
            content = message.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            if not isinstance(content, list):
                continue

            rendered = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text", "").strip()
                    if text:
                        rendered.append(text)
                elif block_type == "thinking":
                    if include_thinking and block.get("thinking", "").strip():
                        rendered.append(_render_thinking(block["thinking"]))
                elif block_type == "tool_use":
                    rendered.append(_render_tool_use(block, results_by_id))

            if rendered:
                lines += ["## Assistant", ""]
                for part in rendered:
                    lines += [part, ""]

    return "\n".join(lines).rstrip() + "\n"


def generate_markdown(
    session_path,
    output_path,
    *,
    include_thinking=True,
    include_subagents=True,
    private=False,
):
    """Generate a markdown transcript from a session file.

    Args:
        session_path: Path to the JSON/JSONL session file.
        output_path: Target .md file, or a directory (the file is then
            named after the session stem).
        include_thinking: Include thinking blocks as blockquoted sections.
        include_subagents: Accepted for CLI parity with the HTML path;
            subagent sessions live in separate JSONL files, so selection
            happens at the CLI layer, not per-session here.
        private: Sanitize paths with the same PathSanitizer treatment the
            HTML exporter applies.

    Returns:
        Path to the written .md file.
    """
    del include_subagents  # selection happens at the CLI layer (see docstring)
    session_path = Path(session_path)
    output_path = Path(output_path)

    data = parse_session_file(session_path)
    loglines = data.get("loglines", [])

    # JSONL-normalized loglines carry only type/timestamp/message; session id
    # and cwd come from the metadata extractor, with the shared header scan
    # as fallback for non-JSONL inputs it doesn't cover.
    meta = extract_rich_metadata(session_path, session_path.parent.name)
    session_id, cwd = meta.session_id, meta.cwd
    if session_id is None or cwd is None:
        scanned_id, scanned_cwd = extract_header_fields(session_path)
        session_id = session_id or scanned_id
        cwd = cwd or scanned_cwd

    if private:
        loglines = _sanitize_loglines(loglines, cwd=cwd)

    summary = get_session_summary(session_path)
    title = summary if summary and summary != "(no summary)" else session_path.stem

    if output_path.suffix == ".md":
        md_path = output_path
    else:
        md_path = output_path / f"{session_path.stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)

    content = render_session_markdown(
        loglines,
        title=title,
        session_id=session_id,
        include_thinking=include_thinking,
    )
    md_path.write_text(content, encoding="utf-8")
    return md_path


def generate_batch_markdown(
    source_folder,
    output_dir,
    include_agents=False,
    include_thinking=True,
    private=False,
    progress_callback=None,
):
    """Generate a markdown archive for all sessions in a projects folder.

    Mirrors generate_batch_html's per-project directory layout
    (output_dir/<project>/<session>.md) but writes no index pages.

    Returns statistics dict with total_projects, total_sessions,
    failed_sessions, output_dir.
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = find_all_sessions(source_folder, include_agents=include_agents)

    total_session_count = sum(len(p["sessions"]) for p in projects)
    processed_count = 0
    successful_sessions = 0
    failed_sessions = []

    for project in projects:
        project_dir = output_dir / project["name"]
        project_dir.mkdir(exist_ok=True)

        for session in project["sessions"]:
            session_name = session["path"].stem
            try:
                generate_markdown(
                    session["path"],
                    project_dir,
                    include_thinking=include_thinking,
                    private=private,
                )
                successful_sessions += 1
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project["name"],
                        "session": session_name,
                        "error": str(e),
                    }
                )

            processed_count += 1
            if progress_callback:
                progress_callback(
                    project["name"], session_name, processed_count, total_session_count
                )

    return {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
    }
