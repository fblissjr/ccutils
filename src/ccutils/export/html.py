"""HTML generation for Claude Code session transcripts.

This module provides functions for rendering Claude Code sessions as HTML,
including message rendering, CSS styling, and pagination.
"""

import html
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import markdown
from jinja2 import Environment, PackageLoader

from ..parsers import (
    extract_searchable_content,
    extract_text_from_content,
    find_all_sessions,
    get_session_summary,
    parse_session_file,
    PROMPTS_PER_PAGE,
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
    """Render markdown text to HTML."""
    if not text:
        return ""
    return markdown.markdown(text, extensions=["fenced_code", "tables"])


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
        description = tool_input.get("description", "")
        display_input = {k: v for k, v in tool_input.items() if k != "description"}
        input_json = json.dumps(display_input, indent=2, ensure_ascii=False)
        return _macros.tool_use(tool_name, description, input_json, tool_id)
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
    else:
        return ""
    if not content_html.strip():
        return ""
    msg_id = make_msg_id(timestamp)
    return _macros.message(role_class, role_label, msg_id, timestamp, content_html)


def generate_pagination_html(current_page, total_pages):
    """Generate pagination HTML for a transcript page."""
    return _macros.pagination(current_page, total_pages)


def generate_index_pagination_html(total_pages):
    """Generate pagination HTML for the index page."""
    return _macros.index_pagination(total_pages)


# Load static assets from files
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
CSS = (_STATIC_DIR / "transcript.css").read_text(encoding="utf-8")
JS = (_STATIC_DIR / "transcript.js").read_text(encoding="utf-8")


def generate_batch_html(
    source_folder,
    output_dir,
    include_agents=False,
    progress_callback=None,
    no_search_index=False,
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

    # Find all sessions
    projects = find_all_sessions(source_folder, include_agents=include_agents)

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
                generate_html(session["path"], session_dir)
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
        css=CSS,
        project_name=project["name"],
        sessions=sessions_data,
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
                "most_recent": most_recent.strftime("%Y-%m-%d %H:%M"),
            }
        )

    content = template.render(
        css=CSS,
        projects=projects_data,
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

    content = template.render(
        css=CSS,
        js=JS,
        title=title,
        sessions=sessions_data,
    )

    index_path = output_dir / "index.html"
    index_path.write_text(content, encoding="utf-8")
    return index_path


def generate_html(json_path=None, output_dir=None, github_repo=None, loglines=None):
    """Generate HTML transcript from a session file or pre-parsed loglines.

    Args:
        json_path: Path to JSON/JSONL session file (required unless loglines provided)
        output_dir: Directory to write HTML files
        github_repo: Optional GitHub repo for commit links (format: "owner/repo")
        loglines: Optional pre-parsed list of log entries (skips file parsing)

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

    # Auto-detect GitHub repo if not provided
    if github_repo is None:
        # First try to detect from session content (git push output)
        github_repo = detect_github_repo(loglines)
        if not github_repo:
            # Fallback: detect from current working directory's git remote
            github_repo = detect_github_repo_from_cwd()

    # Set module-level variable for render functions
    set_github_repo(github_repo)

    # Group messages into conversations (each starting with a user prompt)
    conversations = []
    current_conv = None

    for entry in loglines:
        log_type = entry.get("type")
        timestamp = entry.get("timestamp", "")
        is_compact_summary = entry.get("isCompactSummary", False)
        message_data = entry.get("message", {})

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
                        msg_html = f'<details class="continuation"><summary>Session continuation summary</summary>{msg_html}</details>'
                    messages_html.append(msg_html)
                is_first = False

        pagination_html = generate_pagination_html(page_num, total_pages)

        page_template = get_template("page.html")
        page_content = page_template.render(
            css=CSS,
            js=JS,
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
        css=CSS,
        js=JS,
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

    return output_dir
