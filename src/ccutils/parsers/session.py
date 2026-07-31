"""Session file parsing utilities.

Handles parsing of Claude Code session files in both JSON and JSONL formats.
"""

import json
import re
from pathlib import Path


def parse_session_file(filepath):
    """Parse a session file and return normalized data.

    Supports both JSON and JSONL formats.
    Returns a dict with 'loglines' key containing the normalized entries.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".jsonl":
        return _parse_jsonl_file(filepath)
    else:
        # Standard JSON format
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


# Single source of truth for the JSONL entry types HTML rendering preserves.
# Parser keeps them, renderer dispatches on them, role labels are keyed by them.
# Order matches the dispatch order in `ccutils.export.html.render_message`.
RENDERED_NON_MESSAGE_TYPES = (
    "system",
    "attachment",
    "meta",
    "file-history-snapshot",
    "queue-operation",
    "pr-link",
    "summary",
    "last-prompt",
    "permission-mode",
    "custom-title",
    "agent-name",
)
_RENDERED_NON_MESSAGE_TYPES = frozenset(RENDERED_NON_MESSAGE_TYPES)


def _parse_jsonl_file(filepath):
    """Parse JSONL file and convert to standard format.

    Preserves user/assistant message entries plus the contextual entry
    types Claude Code 2.x emits around them (system, attachment, meta,
    file-history-snapshot, queue-operation, pr-link, summary,
    last-prompt). Progress entries are skipped here -- a real session
    can carry tens of thousands of them and they're not useful inline.
    Renderers downstream dispatch on `type` and fall back to a collapsed
    <details> for types they don't have a styled renderer for yet, so
    no captured entry is silently invisible.
    """
    loglines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = obj.get("type")
            timestamp = obj.get("timestamp", "")

            if entry_type in ("user", "assistant"):
                entry = {
                    "type": entry_type,
                    "timestamp": timestamp,
                    "message": obj.get("message", {}),
                }
                if obj.get("isCompactSummary"):
                    entry["isCompactSummary"] = True
                loglines.append(entry)
            elif entry_type in _RENDERED_NON_MESSAGE_TYPES:
                # Carry the raw entry through under `_raw` so the
                # renderer can introspect subtype + payload without
                # re-parsing the JSONL line.
                loglines.append(
                    {
                        "type": entry_type,
                        "timestamp": timestamp,
                        "message": {},
                        "_raw": obj,
                    }
                )
            # `progress` and any future unknown types fall through

    return {"loglines": loglines}


def extract_text_from_content(content):
    """Extract plain text from message content.

    Handles both string content (older format) and array content (newer format).

    Args:
        content: Either a string or a list of content blocks like
                 [{"type": "text", "text": "..."}, {"type": "image", ...}]

    Returns:
        The extracted text as a string, or empty string if no text found.
    """
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        # Extract text from content blocks of type "text"
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)
        return " ".join(texts).strip()
    return ""


def extract_snippet(content, max_length=200, query=None):
    """Extract a relevant snippet from content.

    If query is provided, centers snippet around first match.

    Args:
        content: The text content to extract a snippet from
        max_length: Maximum length of the snippet
        query: Optional query to center the snippet around

    Returns:
        A snippet string, potentially with "..." added
    """
    if len(content) <= max_length:
        return content

    if query:
        # Find first occurrence of query (case-insensitive)
        pos = content.lower().find(query.lower())
        if pos != -1:
            # Center window around match
            start = max(0, pos - max_length // 2)
            end = min(len(content), start + max_length)
            # Adjust start if we're near the end
            if end == len(content):
                start = max(0, end - max_length)
            snippet = content[start:end]
            if start > 0:
                snippet = "..." + snippet
            if end < len(content):
                snippet = snippet + "..."
            return snippet

    # Default: first N characters
    return content[:max_length] + "..."


def get_session_summary(filepath, max_length=200):
    """Extract a human-readable summary from a session file.

    Supports both JSON and JSONL formats.
    Returns a summary string or "(no summary)" if none found.
    """
    filepath = Path(filepath)
    try:
        if filepath.suffix == ".jsonl":
            return _get_jsonl_summary(filepath, max_length)
        else:
            # For JSON files, try to get first user message
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            loglines = data.get("loglines", [])
            for entry in loglines:
                if entry.get("type") == "user":
                    msg = entry.get("message", {})
                    content = msg.get("content", "")
                    text = extract_text_from_content(content)
                    if text:
                        if len(text) > max_length:
                            return text[: max_length - 3] + "..."
                        return text
            return "(no summary)"
    except Exception:
        return "(no summary)"


def _get_jsonl_summary(filepath, max_length=200):
    """Extract summary from JSONL file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # First priority: summary type entries
                    if obj.get("type") == "summary" and obj.get("summary"):
                        summary = obj["summary"]
                        if len(summary) > max_length:
                            return summary[: max_length - 3] + "..."
                        return summary
                except json.JSONDecodeError:
                    continue

        # Second pass: find first non-meta user message
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if (
                        obj.get("type") == "user"
                        and not obj.get("isMeta")
                        and obj.get("message", {}).get("content")
                    ):
                        content = obj["message"]["content"]
                        text = extract_text_from_content(content)
                        if text and not text.startswith("<"):
                            if len(text) > max_length:
                                return text[: max_length - 3] + "..."
                            return text
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return "(no summary)"


def iter_jsonl_dicts(session_path, must_contain=None):
    """Yield parsed dict entries from a JSONL session file.

    The single tolerant JSONL scanner shared by every header/identity
    extractor (extract_session_metadata, extract_header_fields,
    extract_rich_metadata). Skips blank lines, lines that fail to parse,
    and lines that parse to a NON-dict (a bare scalar in a hand-edited or
    pretty-printed file would otherwise crash `.get()`). No line cap:
    sessions can open with an arbitrarily long headerless prefix
    (accumulated `summary` entries after repeated compaction), so a fixed
    window silently loses identity on long-lived sessions.

    `must_contain`: when set, only lines containing this substring are
    JSON-parsed -- a cheap pre-filter so per-file hot paths (the picker)
    don't fully parse multi-MB message lines just to read a header key.
    """
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
                if must_contain is not None and must_contain not in line:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def extract_session_metadata(session_path):
    """Extract session identity metadata from a session file.

    Returns the first entry that carries a sessionId (agentId/isSidechain
    ride the same entry in real Claude Code data). Tolerates an arbitrary
    headerless prefix.

    Returns dict with:
        - sessionId: The session's unique ID
        - agentId: Agent ID if this is an agent session (None otherwise)
        - isSidechain: True if this is an agent/sidechain session
        - cwd: The session's working directory

    Returns empty dict if file is empty or unreadable; the all-None dict
    if readable but no sessionId is present anywhere.
    """
    identity = None
    cwd = None
    for data in iter_jsonl_dicts(session_path, must_contain='"sessionId"'):
        if data.get("sessionId") is None:
            continue
        if identity is None:
            identity = {
                "sessionId": data.get("sessionId"),
                "agentId": data.get("agentId"),
                "isSidechain": data.get("isSidechain", False),
            }
        # cwd is captured independently, on first truthy occurrence -- some
        # entry types (queue-operation, last-prompt) carry sessionId without
        # cwd and can precede the user/attachment entries that do, so it
        # must not be pinned to whichever line supplies the identity fields.
        if cwd is None and data.get("cwd") is not None:
            cwd = data.get("cwd")
        if identity is not None and cwd is not None:
            break

    if identity is not None:
        identity["cwd"] = cwd
        return identity
    # Distinguish readable-but-headerless from empty/unreadable only if a
    # caller needs it; discovery uses .get(), which treats both alike.
    if Path(session_path).exists() and Path(session_path).stat().st_size > 0:
        return {
            "sessionId": None, "agentId": None, "isSidechain": False, "cwd": None,
        }
    return {}


def extract_header_fields(session_path):
    """Scan a JSONL session for its header identity: (session_id, cwd).

    First (truthy) occurrence of each field wins -- cwd can change
    mid-session, and the header keeps the starting one. Tolerates
    headerless leading lines and non-dict lines. Returns (None, None) on
    unreadable files.
    """
    session_id = None
    cwd = None
    for obj in iter_jsonl_dicts(session_path):
        session_id = session_id or obj.get("sessionId")
        cwd = cwd or obj.get("cwd")
        if session_id and cwd:
            break
    return session_id, cwd


def extract_session_slug(session_path):
    """Extract slug from session file (links related sessions).

    Sessions that are resumed/continued share the same slug field.
    This allows grouping related sessions into conversation chains.

    Args:
        session_path: Path to the session file

    Returns:
        The slug string if found, None otherwise.
    """
    session_path = Path(session_path)
    try:
        with open(session_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "slug" in data:
                        return data["slug"]
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return None


# Constant for pagination
PROMPTS_PER_PAGE = 5


def extract_searchable_content(loglines, project_name, session_name):
    """Extract searchable documents from a session's loglines.

    Args:
        loglines: List of log entries from parse_session_file()
        project_name: Name of the project for document IDs
        session_name: Name of the session for document IDs

    Returns:
        List of document dicts for the search index, each containing:
        - id: Unique document ID
        - project: Project name
        - session: Session name
        - page: Page filename (e.g., "page-001.html")
        - anchor: Anchor ID for linking
        - type: Document type (user, assistant, tool_use, tool_result)
        - timestamp: ISO timestamp
        - content: Searchable text content
        - snippet: Short preview text
    """
    documents = []
    page_num = 1
    user_prompt_count = 0

    def make_anchor(timestamp):
        """Create anchor ID from timestamp."""
        if not timestamp:
            return ""
        # Convert to safe ID: msg-2025-01-01T10-00-00-000Z
        return "msg-" + timestamp.replace(":", "-").replace(".", "-")

    def add_document(doc_type, timestamp, content):
        """Helper to add a document to the list."""
        if not content or not content.strip():
            return
        documents.append(
            {
                "id": f"{project_name}/{session_name}/page-{page_num:03d}#{make_anchor(timestamp)}",
                "project": project_name,
                "session": session_name,
                "page": f"page-{page_num:03d}.html",
                "anchor": make_anchor(timestamp),
                "type": doc_type,
                "timestamp": timestamp,
                "content": content.strip(),
                "snippet": extract_snippet(content.strip()),
            }
        )

    for entry in loglines:
        log_type = entry.get("type")
        timestamp = entry.get("timestamp", "")
        message_data = entry.get("message", {})
        content = message_data.get("content", "")

        if log_type == "user":
            # Track page number based on user prompts
            user_prompt_count += 1
            if user_prompt_count > PROMPTS_PER_PAGE:
                page_num += 1
                user_prompt_count = 1

            # Handle different content formats
            if isinstance(content, str):
                add_document("user", timestamp, content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            add_document("user", timestamp, block.get("text", ""))
                        elif block_type == "tool_result":
                            # Tool results - truncate to 500 chars
                            result_content = block.get("content", "")
                            if isinstance(result_content, str):
                                truncated = result_content[:500]
                                add_document("tool_result", timestamp, truncated)

        elif log_type == "assistant":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        if block_type == "text":
                            add_document("assistant", timestamp, block.get("text", ""))
                        elif block_type == "tool_use":
                            # Index tool name and input
                            tool_name = block.get("name", "")
                            tool_input = block.get("input", {})
                            if isinstance(tool_input, dict):
                                input_str = json.dumps(tool_input)[:1000]
                            else:
                                input_str = str(tool_input)[:1000]
                            tool_content = f"{tool_name}: {input_str}"
                            add_document("tool_use", timestamp, tool_content)

    return documents


def extract_repo_from_session(session):
    """Extract GitHub repo from session metadata.

    Looks in session_context.outcomes for git_info.repo,
    or parses from session_context.sources URL.

    Returns repo as "owner/name" or None.
    """
    context = session.get("session_context", {})

    # Try outcomes first (has clean repo format)
    outcomes = context.get("outcomes", [])
    for outcome in outcomes:
        if outcome.get("type") == "git_repository":
            git_info = outcome.get("git_info", {})
            repo = git_info.get("repo")
            if repo:
                return repo

    # Fall back to sources URL
    sources = context.get("sources", [])
    for source in sources:
        if source.get("type") == "git_repository":
            url = source.get("url", "")
            if "github.com/" in url:
                match = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?$", url)
                if match:
                    return match.group(1)

    return None
