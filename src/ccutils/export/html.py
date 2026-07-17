"""HTML generation for Claude Code session transcripts.

This module provides functions for rendering Claude Code sessions as HTML,
including message rendering, CSS styling, and pagination.
"""

import html
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import markdown
import nh3
from jinja2 import Environment, PackageLoader

from ..parsers import (
    extract_searchable_content,
    extract_text_from_content,
    find_all_sessions,
    get_session_summary,
    parse_session_file,
    PROMPTS_PER_PAGE,
)
from ..parsers.discovery import curate_projects
from ..parsers.session import RENDERED_NON_MESSAGE_TYPES, extract_header_fields

# Display labels for the non-message entry types. Keys must match
# RENDERED_NON_MESSAGE_TYPES (validated below); the renderer falls back
# to a title-cased version of the entry type if a label is missing.
_ENTRY_ROLE_LABELS = {
    "system": "System",
    "attachment": "Attachment",
    "meta": "Meta",
    "file-history-snapshot": "File history snapshot",
    "queue-operation": "Queue",
    "pr-link": "PR",
    "summary": "Summary",
    "last-prompt": "Last prompt",
    "permission-mode": "Permission mode",
    "custom-title": "Custom title",
    "agent-name": "Agent",
}
assert set(_ENTRY_ROLE_LABELS) == set(RENDERED_NON_MESSAGE_TYPES), (
    "RENDERED_NON_MESSAGE_TYPES drifted from _ENTRY_ROLE_LABELS"
)

# Set up Jinja2 environment
_jinja_env = Environment(
    loader=PackageLoader("ccutils", "templates"),
    autoescape=True,
)

# Load macros template and expose macros
_macros_template = _jinja_env.get_template("macros.html")
_macros = _macros_template.module


def get_template(name):
    """Get a Jinja2 template by name."""
    return _jinja_env.get_template(name)


# Regex to match git commit output: [branch hash] message
COMMIT_PATTERN = re.compile(r"\[[\w\-/]+ ([a-f0-9]{7,})\] (.+?)(?:\n|$)")

# Regex to detect GitHub repo from git push output
GITHUB_REPO_PATTERN = re.compile(
    r"github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)/pull/new/"
)

LONG_TEXT_THRESHOLD = (
    300  # Characters - text blocks longer than this are shown in index
)

# Module-level variable for GitHub repo (set by generate_html)
_github_repo = None


def set_github_repo(repo):
    """Set the module-level GitHub repo for rendering commit links."""
    global _github_repo
    _github_repo = repo


def get_github_repo():
    """Get the current GitHub repo setting."""
    return _github_repo


def detect_github_repo_from_cwd():
    """Detect GitHub repo from current working directory's git remote.

    Runs `git remote get-url origin` and parses the GitHub URL.
    Supports both HTTPS and SSH URL formats.

    Returns repo in format "owner/repo" or None if not detected.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # HTTPS: https://github.com/owner/repo.git
            if "github.com" in url:
                # Remove .git suffix if present
                url = url.rstrip("/").removesuffix(".git")
                # Extract owner/repo
                if "github.com/" in url:
                    return url.split("github.com/")[1]
                elif "github.com:" in url:  # SSH format: git@github.com:owner/repo
                    return url.split("github.com:")[1]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def detect_github_repo(loglines):
    """Detect GitHub repo from session loglines.

    Looks for GitHub URLs in git push output within tool results.
    Returns repo in format "owner/repo" or None if not detected.
    """
    for entry in loglines:
        message = entry.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    match = GITHUB_REPO_PATTERN.search(result_content)
                    if match:
                        return match.group(1)
    return None


def format_json(obj):
    """Format an object as JSON HTML."""
    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
        formatted = json.dumps(obj, indent=2, ensure_ascii=False)
        return f'<pre class="json">{html.escape(formatted)}</pre>'
    except (json.JSONDecodeError, TypeError):
        return f"<pre>{html.escape(str(obj))}</pre>"


def render_markdown_text(text):
    """Render markdown text to HTML, with sanitization to prevent XSS."""
    if not text:
        return ""
    raw = markdown.markdown(text, extensions=["fenced_code", "tables"])
    return nh3.clean(raw, attributes={"code": {"class"}})


def is_json_like(text):
    """Check if text appears to be JSON."""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    return (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    )


def render_todo_write(tool_input, tool_id):
    """Render TodoWrite tool calls."""
    todos = tool_input.get("todos", [])
    if not todos:
        return ""
    return _macros.todo_list(todos, tool_id)


def render_write_tool(tool_input, tool_id):
    """Render Write tool calls with file path header and content preview."""
    file_path = tool_input.get("file_path", "Unknown file")
    content = tool_input.get("content", "")
    return _macros.write_tool(file_path, content, tool_id)


def render_edit_tool(tool_input, tool_id):
    """Render Edit tool calls with diff-like old/new display."""
    file_path = tool_input.get("file_path", "Unknown file")
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    replace_all = tool_input.get("replace_all", False)
    return _macros.edit_tool(file_path, old_string, new_string, replace_all, tool_id)


def render_bash_tool(tool_input, tool_id):
    """Render Bash tool calls with command as plain text."""
    command = tool_input.get("command", "")
    description = tool_input.get("description", "")
    return _macros.bash_tool(command, description, tool_id)


def _render_generic_tool_use(tool_name, tool_input, tool_id):
    """Generic tool_use renderer for any tool without dedicated styling.

    Shared by client-side `tool_use`, server-side `server_tool_use`, and
    `mcp_tool_use`. Pulls `description` out of input for the macro header
    and renders the rest as pretty JSON.
    """
    description = tool_input.get("description", "")
    display_input = {k: v for k, v in tool_input.items() if k != "description"}
    input_json = json.dumps(display_input, indent=2, ensure_ascii=False)
    return _macros.tool_use(tool_name, description, input_json, tool_id)


def render_content_block(block):
    """Render a single content block to HTML."""
    if not isinstance(block, dict):
        return f"<p>{html.escape(str(block))}</p>"
    block_type = block.get("type", "")
    if block_type == "image":
        source = block.get("source", {})
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return _macros.image_block(media_type, data)
    elif block_type == "thinking":
        content_html = render_markdown_text(block.get("thinking", ""))
        return _macros.thinking(content_html)
    elif block_type == "redacted_thinking":
        # Anthropic-side safety redaction on Opus thinking. We render a
        # small banner so the user can see redaction happened.
        return _macros.entry_banner(
            "Redacted thinking",
            "Thinking content was redacted by the API safety layer.",
            "thinking",
        )
    elif block_type == "text":
        content_html = render_markdown_text(block.get("text", ""))
        return _macros.assistant_text(content_html)
    elif block_type == "tool_use":
        tool_name = block.get("name", "Unknown tool")
        tool_input = block.get("input", {})
        tool_id = block.get("id", "")
        if tool_name == "TodoWrite":
            return render_todo_write(tool_input, tool_id)
        if tool_name == "Write":
            return render_write_tool(tool_input, tool_id)
        if tool_name == "Edit":
            return render_edit_tool(tool_input, tool_id)
        if tool_name == "Bash":
            return render_bash_tool(tool_input, tool_id)
        return _render_generic_tool_use(tool_name, tool_input, tool_id)
    elif block_type in ("server_tool_use", "mcp_tool_use"):
        # Server-side (Anthropic-hosted) and MCP tool calls. Same shape
        # as a regular tool_use, no client-side dispatch (no Bash/Edit/etc).
        return _render_generic_tool_use(
            block.get("name", block_type),
            block.get("input", {}),
            block.get("id", ""),
        )
    elif block_type in (
        "web_search_tool_result",
        "code_execution_tool_result",
        "mcp_tool_result",
    ):
        content = block.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, ensure_ascii=False)
        return _macros.entry_banner(
            block_type.replace("_", " ").title(),
            content,
            "tool",
        )
    elif block_type == "tool_result":
        content = block.get("content", "")
        is_error = block.get("is_error", False)
        has_images = False

        # Check for git commits and render with styled cards
        if isinstance(content, str):
            commits_found = list(COMMIT_PATTERN.finditer(content))
            if commits_found:
                # Build commit cards + remaining content
                parts = []
                last_end = 0
                for match in commits_found:
                    # Add any content before this commit
                    before = content[last_end : match.start()].strip()
                    if before:
                        parts.append(f"<pre>{html.escape(before)}</pre>")

                    commit_hash = match.group(1)
                    commit_msg = match.group(2)
                    parts.append(
                        _macros.commit_card(commit_hash, commit_msg, _github_repo)
                    )
                    last_end = match.end()

                # Add any remaining content after last commit
                after = content[last_end:].strip()
                if after:
                    parts.append(f"<pre>{html.escape(after)}</pre>")

                content_html = "".join(parts)
            else:
                content_html = f"<pre>{html.escape(content)}</pre>"
        elif isinstance(content, list):
            # Handle tool result content that contains multiple blocks
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            parts.append(f"<pre>{html.escape(text)}</pre>")
                    elif item_type == "image":
                        source = item.get("source", {})
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        if data:
                            parts.append(_macros.image_block(media_type, data))
                            has_images = True
                    else:
                        # Unknown type, render as JSON
                        parts.append(format_json(item))
                else:
                    # Non-dict item, escape as text
                    parts.append(f"<pre>{html.escape(str(item))}</pre>")
            content_html = "".join(parts) if parts else format_json(content)
        elif is_json_like(content):
            content_html = format_json(content)
        else:
            content_html = format_json(content)
        return _macros.tool_result(content_html, is_error, has_images)
    else:
        return format_json(block)


def render_user_message_content(message_data):
    """Render user message content to HTML."""
    content = message_data.get("content", "")
    if isinstance(content, str):
        if is_json_like(content):
            return _macros.user_content(format_json(content))
        return _macros.user_content(render_markdown_text(content))
    elif isinstance(content, list):
        return "".join(render_content_block(block) for block in content)
    return f"<p>{html.escape(str(content))}</p>"


def render_assistant_message(message_data):
    """Render assistant message content to HTML."""
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return f"<p>{html.escape(str(content))}</p>"
    return "".join(render_content_block(block) for block in content)


def make_msg_id(timestamp):
    """Create a message ID from a timestamp."""
    return f"msg-{timestamp.replace(':', '-').replace('.', '-')}"


def analyze_conversation(messages):
    """Analyze messages in a conversation to extract stats and long texts."""
    tool_counts = {}  # tool_name -> count
    long_texts = []
    commits = []  # list of (hash, message, timestamp)

    for log_type, message_json, timestamp in messages:
        if not message_json:
            continue
        try:
            message_data = json.loads(message_json)
        except json.JSONDecodeError:
            continue

        content = message_data.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")

            if block_type == "tool_use":
                tool_name = block.get("name", "Unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            elif block_type == "tool_result":
                # Check for git commit output
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    for match in COMMIT_PATTERN.finditer(result_content):
                        commits.append((match.group(1), match.group(2), timestamp))
            elif block_type == "text":
                text = block.get("text", "")
                if len(text) >= LONG_TEXT_THRESHOLD:
                    long_texts.append(text)

    return {
        "tool_counts": tool_counts,
        "long_texts": long_texts,
        "commits": commits,
    }


def format_tool_stats(tool_counts):
    """Format tool counts into a concise summary string."""
    if not tool_counts:
        return ""

    # Abbreviate common tool names
    abbrev = {
        "Bash": "bash",
        "Read": "read",
        "Write": "write",
        "Edit": "edit",
        "Glob": "glob",
        "Grep": "grep",
        "Task": "task",
        "TodoWrite": "todo",
        "WebFetch": "fetch",
        "WebSearch": "search",
    }

    parts = []
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        short_name = abbrev.get(name, name.lower())
        parts.append(f"{count} {short_name}")

    return " . ".join(parts)


def is_tool_result_message(message_data):
    """Check if a message contains only tool_result blocks."""
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return False
    if not content:
        return False
    return all(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def render_message(log_type, message_json, timestamp):
    """Render a single message to HTML."""
    if not message_json:
        return ""
    try:
        message_data = json.loads(message_json)
    except json.JSONDecodeError:
        return ""
    if log_type == "user":
        content_html = render_user_message_content(message_data)
        # Check if this is a tool result message
        if is_tool_result_message(message_data):
            role_class, role_label = "tool-reply", "Tool reply"
        else:
            role_class, role_label = "user", "User"
    elif log_type == "assistant":
        content_html = render_assistant_message(message_data)
        role_class, role_label = "assistant", "Assistant"
    elif log_type in RENDERED_NON_MESSAGE_TYPES:
        # `message_data` is actually the raw JSONL entry for non-message
        # types (parser stores the raw obj under `_raw`, the body loop
        # passes it through this slot to keep render_message's signature
        # the same).
        content_html = render_non_message_entry(log_type, message_data)
        role_class = f"entry-{log_type}"
        role_label = _ENTRY_ROLE_LABELS[log_type]
    else:
        return ""
    if not content_html.strip():
        return ""
    msg_id = make_msg_id(timestamp)
    return _macros.message(role_class, role_label, msg_id, timestamp, content_html)


def render_non_message_entry(log_type, raw_obj):
    """Render a non-message JSONL entry (system, attachment, meta, etc.).

    Phase 1: styled banners for a few high-signal top-level types and
    subtypes; collapsed <details> fallback for everything else. Phase 2
    will add styled renderers for diagnostics, plan-mode, hook progress,
    and more attachment subtypes.
    """
    # Top-level types that carry a single high-signal field.
    if log_type == "permission-mode":
        return _macros.entry_banner(
            "Permission mode", str(raw_obj.get("permissionMode", "")), "system"
        )
    if log_type == "custom-title":
        return _macros.entry_banner(
            "Custom title", str(raw_obj.get("customTitle", "")), "user"
        )
    if log_type == "agent-name":
        return _macros.entry_banner(
            "Agent", str(raw_obj.get("agentName", "")), "user"
        )
    if log_type == "last-prompt":
        return _macros.entry_banner(
            "Queued prompt", str(raw_obj.get("lastPrompt", "")), "user"
        )
    if log_type == "pr-link":
        url = raw_obj.get("url") or raw_obj.get("prUrl") or ""
        return _macros.entry_banner("Linked PR", url, "user")
    if log_type == "summary":
        return _macros.entry_banner(
            "Summary", raw_obj.get("summary", ""), "system"
        )
    if log_type == "queue-operation":
        op = raw_obj.get("operation", "")
        content = raw_obj.get("content", "") or ""
        # Queue payloads can be huge (entire subagent reports). Truncate.
        if len(content) > 400:
            content = content[:400] + " ... (truncated)"
        label = f"Queue: {op}" if op else "Queue"
        return _macros.entry_banner(label, content, "system")
    if log_type == "file-history-snapshot":
        snap = raw_obj.get("snapshot") or {}
        backups = snap.get("trackedFileBackups") or {}
        is_update = raw_obj.get("isSnapshotUpdate", False)
        label = "Snapshot update" if is_update else "Snapshot"
        if not backups:
            detail = "No tracked file backups."
        else:
            detail = f"{len(backups)} tracked file(s): " + ", ".join(
                list(backups.keys())[:5]
            )
            if len(backups) > 5:
                detail += f" (+{len(backups) - 5} more)"
        return _macros.entry_banner(label, detail, "system")

    if log_type == "system":
        subtype = raw_obj.get("subtype") or ""
        if subtype == "turn_duration":
            duration_ms = raw_obj.get("durationMs")
            message_count = raw_obj.get("messageCount")
            if duration_ms is not None:
                detail = f"{duration_ms} ms"
                if message_count is not None:
                    detail += f" / {message_count} messages"
                return _macros.entry_banner("Turn duration", detail, "system")
        if subtype == "stop_hook_summary":
            hook_count = raw_obj.get("hookCount", 0)
            prevented = raw_obj.get("preventedContinuation", False)
            stop_reason = raw_obj.get("stopReason", "") or ""
            detail_parts = [f"{hook_count} hook(s)"]
            if stop_reason:
                detail_parts.append(f"reason: {stop_reason}")
            if prevented:
                detail_parts.append("continuation prevented")
            return _macros.entry_banner(
                "Stop hook", "; ".join(detail_parts), "system"
            )
        if subtype == "api_error":
            err = raw_obj.get("error") or raw_obj.get("message") or "API error"
            return _macros.entry_banner("API error", str(err), "thinking")
        if subtype == "compact_boundary":
            return _macros.entry_banner(
                "Compact boundary",
                "Conversation context was compacted at this point.",
                "system",
            )

    if log_type == "attachment":
        att = raw_obj.get("attachment") or {}
        sub = att.get("type") or ""
        if sub == "diagnostics":
            diagnostics = att.get("diagnostics") or []
            return _macros.entry_banner(
                "Diagnostics",
                f"{len(diagnostics)} diagnostic(s) reported by LSP.",
                "thinking",
            )
        if sub == "hook_success":
            hook_name = att.get("hookName", "")
            duration = att.get("durationMs")
            detail = hook_name
            if duration is not None:
                detail += f" ({duration} ms)"
            return _macros.entry_banner("Hook", detail, "system")

    # Fallback: render the raw JSON inside a collapsed <details>.
    # Truncate to keep page sizes sane on attachment-heavy sessions.
    label = log_type.replace("-", " ").title()
    raw_str = json.dumps(raw_obj, indent=2, ensure_ascii=False)
    if len(raw_str) > 2000:
        raw_str = raw_str[:2000] + "\n... (truncated)"
    return _macros.entry_fallback(label, raw_str)


def generate_pagination_html(current_page, total_pages):
    """Generate pagination HTML for a transcript page."""
    return _macros.pagination(current_page, total_pages)


def generate_index_pagination_html(total_pages):
    """Generate pagination HTML for the index page."""
    return _macros.index_pagination(total_pages)


# Load static assets via importlib.resources (works with zip/wheel installs)
from importlib.resources import files as _resource_files

_STATIC = _resource_files("ccutils") / "static"
CSS = (_STATIC / "transcript.css").read_text(encoding="utf-8")
JS = (_STATIC / "transcript.js").read_text(encoding="utf-8")


def generate_batch_html(
    source_folder,
    output_dir,
    include_agents=False,
    progress_callback=None,
    no_search_index=False,
    private=False,
    projects=None,
):
    """Generate HTML archive for all sessions in a Claude projects folder.

    Creates:
    - Master index.html listing all projects
    - Per-project directories with index.html listing sessions
    - Per-session directories with transcript pages
    - search-index.js for full-text search (unless no_search_index=True)

    Args:
        source_folder: Path to the Claude projects folder
        output_dir: Path for output archive
        include_agents: Whether to include agent-* session files
        progress_callback: Optional callback(project_name, session_name, current, total)
            called after each session is processed
        no_search_index: If True, skip generating the search index

    Returns statistics dict with total_projects, total_sessions, failed_sessions, output_dir.
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / "transcript.css").write_text(CSS, encoding="utf-8")
    (output_dir / "transcript.js").write_text(JS, encoding="utf-8")

    # Find all sessions. A pre-scanned list (already project-filtered by
    # the CLI) skips the rescan; either way the render-format curation
    # rule applies -- html never renders warmup/no-summary sessions.
    if projects is None:
        projects = find_all_sessions(source_folder, include_agents=include_agents)
    else:
        projects = curate_projects(projects)

    # Calculate total for progress tracking
    total_session_count = sum(len(p["sessions"]) for p in projects)
    processed_count = 0
    successful_sessions = 0
    failed_sessions = []

    # Process each project
    for project in projects:
        project_dir = output_dir / project["name"]
        project_dir.mkdir(exist_ok=True)

        # Process each session
        for session in project["sessions"]:
            session_name = session["path"].stem
            session_dir = project_dir / session_name

            # Generate transcript HTML with error handling
            try:
                generate_html(session["path"], session_dir, private=private, rel_path="../../")
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

            # Call progress callback if provided
            if progress_callback:
                progress_callback(
                    project["name"], session_name, processed_count, total_session_count
                )

        # Generate project index
        _generate_project_index(project, project_dir)

    # Generate master index (with search UI if search index will be generated)
    has_search_index = not no_search_index
    _generate_master_index(projects, output_dir, has_search_index=has_search_index)

    # Generate search index (unless disabled)
    if has_search_index:
        _generate_search_index(projects, output_dir)

    return {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
    }


def _generate_project_index(project, output_dir):
    """Generate index.html for a single project."""
    template = get_template("project_index.html")

    # Format sessions for template
    sessions_data = []
    for session in project["sessions"]:
        mod_time = datetime.fromtimestamp(session["mtime"])
        sessions_data.append(
            {
                "name": session["path"].stem,
                "summary": session["summary"],
                "date": mod_time.strftime("%Y-%m-%d %H:%M"),
                "size_kb": session["size"] / 1024,
            }
        )

    content = template.render(
        rel_path="../",
        project_name=project["name"],
        sessions=sessions_data,
        session_count=len(sessions_data),
    )
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def _generate_master_index(projects, output_dir, has_search_index=False):
    """Generate master index.html listing all projects."""
    template = get_template("master_index.html")

    # Format projects for template
    projects_data = []
    for project in projects:
        if not project["sessions"]:
            continue
        most_recent = datetime.fromtimestamp(project["sessions"][0]["mtime"])
        projects_data.append(
            {
                "name": project["name"],
                "session_count": len(project["sessions"]),
                "recent_date": most_recent.strftime("%Y-%m-%d %H:%M"),
            }
        )

    total_sessions = sum(p["session_count"] for p in projects_data)

    if has_search_index:
        global_search_js = _jinja_env.get_template("global_search.js").render()
        (output_dir / "global_search.js").write_text(global_search_js, encoding="utf-8")

    content = template.render(
        rel_path="",
        projects=projects_data,
        total_projects=len(projects_data),
        total_sessions=total_sessions,
        has_search_index=has_search_index,
    )
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def _generate_search_index(projects, output_dir):
    """Generate search-index.js with searchable content from all sessions."""
    all_documents = []

    for project in projects:
        for session in project["sessions"]:
            session_path = session["path"]
            try:
                # Parse session file and extract documents
                data = parse_session_file(session_path)
                loglines = data.get("loglines", [])
                session_docs = extract_searchable_content(
                    loglines, project["name"], session_path.stem
                )
                if session_docs:
                    all_documents.extend(session_docs)
            except Exception:
                # Skip sessions that fail to parse
                pass

    # Build index with version info
    search_index = {
        "version": 1,
        "documents": all_documents,
    }

    # Write as JavaScript file
    js_content = f"var SEARCH_INDEX = {json.dumps(search_index, ensure_ascii=False)};"
    (output_dir / "search-index.js").write_text(js_content, encoding="utf-8")


def generate_multi_session_index(
    output_dir,
    sessions,
    agent_map=None,
    title="Sessions",
):
    """Generate an index page for multiple sessions.

    Args:
        output_dir: Directory to write index.html
        sessions: List of session Paths
        agent_map: Optional dict mapping parent session Path to list of agent Paths
        title: Page title

    Returns:
        Path to generated index.html
    """
    output_dir = Path(output_dir)
    agent_map = agent_map or {}
    template = get_template("multi_session_index.html")

    # Format sessions for template
    sessions_data = []
    for session_path in sessions:
        session_path = Path(session_path)
        stat = session_path.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        summary = get_session_summary(session_path)

        # Check if this is an agent session
        is_agent = session_path.name.startswith("agent-")

        # Get agent count for parent sessions
        agent_count = len(agent_map.get(session_path, []))

        sessions_data.append(
            {
                "name": session_path.stem,
                "summary": summary,
                "date": mod_time.strftime("%Y-%m-%d %H:%M"),
                "size_kb": stat.st_size / 1024,
                "is_agent": is_agent,
                "agent_count": agent_count,
            }
        )

    (output_dir / "transcript.css").write_text(CSS, encoding="utf-8")
    (output_dir / "transcript.js").write_text(JS, encoding="utf-8")

    content = template.render(
        rel_path="",
        title=title,
        sessions=sessions_data,
    )

    index_path = output_dir / "index.html"
    index_path.write_text(content, encoding="utf-8")
    return index_path


def _resolve_private_cwd(source_path, loglines):
    """Resolve the working directory to sanitize --private paths against.

    Prefers a scan of the session file (normalized loglines drop cwd), then
    falls back to any cwd carried on a logline (dict-format .json inputs).
    Returns None when neither yields one -- the caller must then fail loud,
    NOT silently ship unsanitized output.
    """
    if source_path is not None:
        _, cwd = extract_header_fields(source_path)
        if cwd:
            return cwd
    for entry in loglines or []:
        cwd = entry.get("cwd")
        if cwd:
            return cwd
    return None


def _warn_private_unresolved(label):
    """Loud stderr warning: --private was requested but could not sanitize.

    The privacy contract has failed silently twice; a cwd-less session
    (agent transcripts, .json/claude.ai exports, hand-assembled loglines)
    would otherwise no-op with exit 0. Fail loud instead.
    """
    print(
        f"WARNING: --private could not determine a working directory for "
        f"{label}; file paths were NOT sanitized in the output. "
        f"Review the output before sharing.",
        file=sys.stderr,
    )


def _sanitize_loglines(loglines, cwd=None):
    """Sanitize paths in loglines for private mode.

    Extracts cwd from the first logline's raw data (unless passed
    explicitly), creates a PathSanitizer, then deep-walks all loglines to
    sanitize tool_use input dicts and tool_result content strings.

    NOTE: best-effort. Only a subset of channels is walked (tool_use input
    file_path/command/content/path and string tool_result content); message
    text, thinking blocks, and non-message entries are NOT sanitized. See
    the --private known-limitations note in README / CHANGELOG.
    """
    from ..sanitize import PathSanitizer

    if not loglines:
        return loglines

    # Extract cwd from first logline unless the caller provided one
    if not cwd:
        for entry in loglines:
            cwd = entry.get("cwd")
            if cwd:
                break

    if not cwd:
        return loglines

    sanitizer = PathSanitizer(cwd)

    for entry in loglines:
        message_data = entry.get("message", {})
        content = message_data.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "tool_use":
                tool_input = block.get("input", {})
                if isinstance(tool_input, dict):
                    # Sanitize known path fields
                    if "file_path" in tool_input and isinstance(
                        tool_input["file_path"], str
                    ):
                        tool_input["file_path"] = sanitizer.sanitize_path(
                            tool_input["file_path"]
                        )
                    if "command" in tool_input and isinstance(
                        tool_input["command"], str
                    ):
                        tool_input["command"] = sanitizer.sanitize_text(
                            tool_input["command"]
                        )
                    if "content" in tool_input and isinstance(
                        tool_input["content"], str
                    ):
                        tool_input["content"] = sanitizer.sanitize_text(
                            tool_input["content"]
                        )
                    if "path" in tool_input and isinstance(tool_input["path"], str):
                        tool_input["path"] = sanitizer.sanitize_path(tool_input["path"])

            elif block_type == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    block["content"] = sanitizer.sanitize_text(result_content)

    return loglines


def generate_html(
    json_path=None,
    output_dir=None,
    github_repo=None,
    loglines=None,
    private=False,
    rel_path="",
):
    """Generate HTML transcript from a session file or pre-parsed loglines.

    Args:
        json_path: Path to JSON/JSONL session file (required unless loglines provided)
        output_dir: Directory to write HTML files
        github_repo: Optional GitHub repo for commit links (format: "owner/repo")
        loglines: Optional pre-parsed list of log entries (skips file parsing)
        private: If True, sanitize paths to remove sensitive directory info

    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load session file (supports both JSON and JSONL) or use provided loglines
    if loglines is None:
        if json_path is None:
            raise ValueError("Either json_path or loglines must be provided")
        data = parse_session_file(json_path)
        loglines = data.get("loglines", [])

    # Sanitize paths in loglines if private mode. Normalized JSONL loglines
    # carry only type/timestamp/message -- no cwd -- so resolve cwd from the
    # session file (or a dict-format logline). Fail loud if unresolvable
    # rather than shipping unsanitized output with exit 0.
    if private:
        cwd = _resolve_private_cwd(json_path, loglines)
        if cwd is None:
            _warn_private_unresolved(str(json_path) if json_path else "session")
        loglines = _sanitize_loglines(loglines, cwd=cwd)

    # Auto-detect GitHub repo if not provided
    if github_repo is None:
        # First try to detect from session content (git push output)
        github_repo = detect_github_repo(loglines)
        if not github_repo:
            # Fallback: detect from current working directory's git remote
            github_repo = detect_github_repo_from_cwd()

    # Save previous global so back-to-back generate_html() calls (e.g. from
    # generate_batch_html) don't leak state between sessions.
    if rel_path == "":
        (output_dir / "transcript.css").write_text(CSS, encoding="utf-8")
        (output_dir / "transcript.js").write_text(JS, encoding="utf-8")

    _saved_github_repo = get_github_repo()
    set_github_repo(github_repo)
    try:
        return _generate_html_body(loglines, output_dir, rel_path=rel_path)
    finally:
        set_github_repo(_saved_github_repo)


def _generate_html_body(loglines, output_dir, rel_path=""):
    """Render-side body of generate_html. Reads _github_repo via get_github_repo()."""
    # Group messages into conversations (each starting with a user prompt)
    conversations = []
    current_conv = None

    for entry in loglines:
        log_type = entry.get("type")
        timestamp = entry.get("timestamp", "")
        is_compact_summary = entry.get("isCompactSummary", False)
        message_data = entry.get("message", {})

        is_message = log_type in ("user", "assistant")

        # Non-message entries (system / attachment / meta / file-history-snapshot /
        # queue-operation / pr-link / summary / last-prompt) ride along with the
        # current conversation so the renderer can dispatch on type. Skip if no
        # conversation has started yet -- they have nowhere to attach.
        if not is_message:
            if current_conv is not None:
                raw_json = json.dumps(entry.get("_raw", {}))
                current_conv["messages"].append((log_type, raw_json, timestamp))
            continue

        if not message_data:
            continue

        # Convert message dict to JSON string for compatibility with existing render functions
        message_json = json.dumps(message_data)

        # Check if this is a new user prompt
        is_user_prompt = False
        user_text = None

        if log_type == "user":
            content = message_data.get("content", "")
            text = extract_text_from_content(content)
            if text:
                is_user_prompt = True
                user_text = text

        if is_user_prompt:
            # Start a new conversation
            if current_conv:
                conversations.append(current_conv)
            current_conv = {
                "user_text": user_text,
                "timestamp": timestamp,
                "messages": [(log_type, message_json, timestamp)],
                "is_continuation": bool(is_compact_summary),
            }
        elif current_conv:
            # Add to current conversation
            current_conv["messages"].append((log_type, message_json, timestamp))

    # Don't forget the last conversation
    if current_conv:
        conversations.append(current_conv)

    # Calculate pagination
    total_convs = len(conversations)
    total_pages = (total_convs + PROMPTS_PER_PAGE - 1) // PROMPTS_PER_PAGE

    # Generate each page
    for page_num in range(1, total_pages + 1):
        start_idx = (page_num - 1) * PROMPTS_PER_PAGE
        end_idx = min(start_idx + PROMPTS_PER_PAGE, total_convs)
        page_convs = conversations[start_idx:end_idx]

        messages_html = []
        for conv in page_convs:
            is_first = True
            for log_type, message_json, timestamp in conv["messages"]:
                msg_html = render_message(log_type, message_json, timestamp)
                if msg_html:
                    # Wrap continuation summaries in collapsed details
                    if is_first and conv.get("is_continuation"):
                        msg_html = _macros.continuation(msg_html)
                    messages_html.append(msg_html)
                is_first = False

        pagination_html = generate_pagination_html(page_num, total_pages)

        page_template = get_template("page.html")
        page_content = page_template.render(
            rel_path=rel_path,
            page_num=page_num,
            total_pages=total_pages,
            pagination_html=pagination_html,
            messages_html="".join(messages_html),
        )

        (output_dir / f"page-{page_num:03d}.html").write_text(
            page_content, encoding="utf-8"
        )

    # Calculate overall stats and collect all commits for timeline
    total_tool_counts = {}
    total_messages = 0
    all_commits = []  # (timestamp, hash, message, page_num, conv_index)

    for i, conv in enumerate(conversations):
        total_messages += len(conv["messages"])
        stats = analyze_conversation(conv["messages"])
        for tool, count in stats["tool_counts"].items():
            total_tool_counts[tool] = total_tool_counts.get(tool, 0) + count
        page_num = (i // PROMPTS_PER_PAGE) + 1
        for commit_hash, commit_msg, commit_ts in stats["commits"]:
            all_commits.append((commit_ts, commit_hash, commit_msg, page_num, i))

    total_tool_calls = sum(total_tool_counts.values())
    total_commits = len(all_commits)

    # Build timeline items: prompts and commits merged by timestamp
    timeline_items = []

    # Add prompts
    prompt_num = 0
    for i, conv in enumerate(conversations):
        if conv.get("is_continuation"):
            continue
        if conv["user_text"].startswith("Stop hook feedback:"):
            continue
        prompt_num += 1
        page_num = (i // PROMPTS_PER_PAGE) + 1
        msg_id = make_msg_id(conv["timestamp"])
        link = f"page-{page_num:03d}.html#{msg_id}"
        rendered_content = render_markdown_text(conv["user_text"])

        # Collect all messages including from subsequent continuation conversations
        all_messages = list(conv["messages"])
        for j in range(i + 1, len(conversations)):
            if not conversations[j].get("is_continuation"):
                break
            all_messages.extend(conversations[j]["messages"])

        # Analyze conversation for stats
        stats = analyze_conversation(all_messages)
        tool_stats_str = format_tool_stats(stats["tool_counts"])

        long_texts_html = ""
        for lt in stats["long_texts"]:
            rendered_lt = render_markdown_text(lt)
            long_texts_html += _macros.index_long_text(rendered_lt)

        stats_html = _macros.index_stats(tool_stats_str, long_texts_html)

        item_html = _macros.index_item(
            prompt_num, link, conv["timestamp"], rendered_content, stats_html
        )
        timeline_items.append((conv["timestamp"], "prompt", item_html))

    # Add commits as separate timeline items
    for commit_ts, commit_hash, commit_msg, page_num, conv_idx in all_commits:
        item_html = _macros.index_commit(
            commit_hash, commit_msg, commit_ts, get_github_repo()
        )
        timeline_items.append((commit_ts, "commit", item_html))

    # Sort by timestamp
    timeline_items.sort(key=lambda x: x[0])
    index_items = [item[2] for item in timeline_items]

    # Generate index page
    index_pagination = generate_index_pagination_html(total_pages)
    index_template = get_template("index.html")
    index_content = index_template.render(
        rel_path=rel_path,
        pagination_html=index_pagination,
        prompt_num=prompt_num,
        total_messages=total_messages,
        total_tool_calls=total_tool_calls,
        total_commits=total_commits,
        total_pages=total_pages,
        index_items_html="".join(index_items),
    )

    index_path = output_dir / "index.html"
    index_path.write_text(index_content, encoding="utf-8")

    search_js = _jinja_env.get_template("search.js").render(total_pages=total_pages)
    (output_dir / "search.js").write_text(search_js, encoding="utf-8")

    return output_dir
