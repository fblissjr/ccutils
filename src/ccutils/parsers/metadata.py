# path-privacy: skip-file -- references universal Claude Code data paths (not personal)
"""Rich metadata extraction from Claude Code session files.

Provides fast metadata extraction for session selection UI,
including project name, model, git branch, duration, and summaries.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .session import extract_text_from_content

# Messages to skip when looking for a meaningful summary
SKIP_SUMMARY_PATTERNS = [
    "[request interrupted",
    "[error",
    "error processing",
    "an error occurred",
    "api error",
    "rate limit",
    "warmup",
]


@dataclass
class SessionMetadata:
    """Rich metadata for a single session file."""

    path: Path
    session_id: str | None = None
    cwd: str | None = None
    project_name: str = ""
    project_path: str = ""  # parent folder name (grouping key)
    git_branch: str | None = None
    model: str | None = None
    model_short: str = ""
    slug: str | None = None
    summary: str = "(no summary)"
    mtime: float = 0.0
    size: int = 0
    user_msg_count: int = 0
    assistant_msg_count: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    duration_minutes: int | None = None
    version: str | None = None


def shorten_model_name(model_id: str | None) -> str:
    """Shorten a Claude model ID to a human-friendly name.

    Examples:
        claude-opus-4-6 -> opus-4.6
        claude-sonnet-4-5-20250929 -> sonnet-4.5
        claude-haiku-4-5-20251001 -> haiku-4.5
        claude-sonnet-4-20250514 -> sonnet-4
        claude-3-5-sonnet-20241022 -> sonnet-3.5
        None -> ""
    """
    if not model_id:
        return ""

    model_id = model_id.lower().strip()

    # claude-{family}-{version} patterns (Claude 4+ naming)
    # e.g. claude-opus-4-6, claude-sonnet-4-5-20250929, claude-sonnet-4-20250514
    for family in ("opus", "sonnet", "haiku"):
        prefix = f"claude-{family}-"
        if model_id.startswith(prefix):
            rest = model_id[len(prefix) :]
            # Extract version numbers before any date suffix (8+ digits)
            parts = rest.split("-")
            version_parts = []
            for part in parts:
                if part.isdigit() and len(part) <= 2:
                    version_parts.append(part)
                else:
                    break
            if version_parts:
                return f"{family}-{'.'.join(version_parts)}"
            return family

    # claude-3-5-sonnet-20241022 pattern (Claude 3.x naming)
    if model_id.startswith("claude-3"):
        for family in ("opus", "sonnet", "haiku"):
            if family in model_id:
                # Extract version from claude-3-5-sonnet -> 3.5
                parts = model_id.split(f"-{family}")[0].replace("claude-", "")
                version = parts.replace("-", ".")
                return f"{family}-{version}"

    # Fallback: return as-is but strip "claude-" prefix
    if model_id.startswith("claude-"):
        return model_id[7:]
    return model_id


def derive_project_name(cwd: str | None, folder_name: str) -> str:
    """Derive a readable project name from cwd or folder name.

    Prefers cwd (e.g., /Users/dev/workspace/myproject -> myproject)
    because it's the actual directory name, not an encoded path.

    Falls back to folder_name parsing (same logic as get_project_display_name).
    """
    if cwd:
        # Use the last component of the cwd path
        parts = cwd.rstrip("/").split("/")
        if parts and parts[-1]:
            return parts[-1]

    # Fall back to folder name parsing (imported from discovery)
    from .discovery import get_project_display_name

    return get_project_display_name(folder_name)


def _is_skip_summary(text: str) -> bool:
    """Check if text matches a pattern we should skip for summaries."""
    text_lower = text.lower().strip()
    if not text_lower:
        return True
    # Skip XML-like content
    if text_lower.startswith("<"):
        return True
    for pattern in SKIP_SUMMARY_PATTERNS:
        if text_lower.startswith(pattern):
            return True
    return False


def get_meaningful_summary(filepath: Path, max_length: int = 120) -> str:
    """Extract the first meaningful user message as a summary.

    Skips:
    - Empty messages
    - XML-prefixed messages (system prompts)
    - "[Request interrupted...]" messages
    - Error/API messages
    - isMeta messages
    - warmup messages

    Args:
        filepath: Path to session JSONL file
        max_length: Maximum summary length

    Returns:
        First meaningful user text, truncated to max_length.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)

                    # Check for summary type entries first
                    if obj.get("type") == "summary" and obj.get("summary"):
                        summary = obj["summary"]
                        if not _is_skip_summary(summary):
                            if len(summary) > max_length:
                                return summary[: max_length - 3] + "..."
                            return summary

                    if obj.get("type") != "user":
                        continue
                    if obj.get("isMeta"):
                        continue

                    content = obj.get("message", {}).get("content", "")
                    text = extract_text_from_content(content)
                    if not text or _is_skip_summary(text):
                        continue

                    if len(text) > max_length:
                        return text[: max_length - 3] + "..."
                    return text
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass

    return "(no summary)"


def extract_rich_metadata(filepath: Path, folder_name: str) -> SessionMetadata:
    """Extract rich metadata from a session file by scanning it.

    Reads through the file to extract:
    - Session ID, cwd, git branch, slug, version (from first few lines)
    - Model (from first assistant message)
    - First/last timestamps (for duration estimation)
    - User/assistant message counts
    - Meaningful summary

    Args:
        filepath: Path to the session JSONL file
        folder_name: Parent folder name (for project grouping)

    Returns:
        Populated SessionMetadata instance.
    """
    stat = filepath.stat()
    meta = SessionMetadata(
        path=filepath,
        project_path=folder_name,
        mtime=stat.st_mtime,
        size=stat.st_size,
    )

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            got_header = False
            got_model = False
            got_summary = False
            first_ts = None
            last_ts = None
            user_count = 0
            assistant_count = 0

            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = obj.get("type")
                ts = obj.get("timestamp")
                if ts:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

                # Header fields: first OCCURRENCE of each, not first entry.
                # Sessions can open with headerless lines (summary, ...);
                # latching on line 1 lost sessionId/cwd and silently
                # disabled --private downstream.
                if not got_header:
                    meta.session_id = meta.session_id or obj.get("sessionId")
                    meta.cwd = meta.cwd or obj.get("cwd")
                    meta.git_branch = meta.git_branch or obj.get("gitBranch")
                    meta.version = meta.version or obj.get("version")
                    if obj.get("slug"):
                        meta.slug = obj["slug"]
                    if meta.session_id and meta.cwd:
                        got_header = True

                # Slug can appear on any line
                if not meta.slug and obj.get("slug"):
                    meta.slug = obj["slug"]

                # Extract model from first assistant message
                if not got_model and entry_type == "assistant":
                    msg = obj.get("message", {})
                    if isinstance(msg, dict) and msg.get("model"):
                        meta.model = msg["model"]
                        meta.model_short = shorten_model_name(meta.model)
                        got_model = True

                # Count messages
                if entry_type == "user" and not obj.get("isMeta"):
                    user_count += 1
                    # Extract summary from first meaningful user message
                    if not got_summary:
                        content = obj.get("message", {}).get("content", "")
                        text = extract_text_from_content(content)
                        if text and not _is_skip_summary(text):
                            if len(text) > 120:
                                meta.summary = text[:117] + "..."
                            else:
                                meta.summary = text
                            got_summary = True

                elif entry_type == "assistant":
                    assistant_count += 1

                # Also check for summary type entries
                if not got_summary and entry_type == "summary" and obj.get("summary"):
                    summary_text = obj["summary"]
                    if not _is_skip_summary(summary_text):
                        if len(summary_text) > 120:
                            meta.summary = summary_text[:117] + "..."
                        else:
                            meta.summary = summary_text
                        got_summary = True

            meta.user_msg_count = user_count
            meta.assistant_msg_count = assistant_count
            meta.first_timestamp = first_ts
            meta.last_timestamp = last_ts

    except OSError:
        pass

    # Derive project name from cwd or folder
    meta.project_name = derive_project_name(meta.cwd, folder_name)

    # Estimate duration
    meta.duration_minutes = _estimate_duration(
        meta.first_timestamp, meta.last_timestamp
    )

    return meta


def _estimate_duration(first_ts: str | None, last_ts: str | None) -> int | None:
    """Estimate session duration in minutes from ISO timestamps."""
    if not first_ts or not last_ts:
        return None
    try:
        from datetime import datetime

        # Parse ISO timestamps (handle both Z and +00:00 suffixes)
        fmt_options = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"]
        first_dt = None
        last_dt = None
        for fmt in fmt_options:
            try:
                first_dt = datetime.strptime(first_ts, fmt)
                break
            except ValueError:
                continue
        for fmt in fmt_options:
            try:
                last_dt = datetime.strptime(last_ts, fmt)
                break
            except ValueError:
                continue

        if first_dt and last_dt:
            delta = last_dt - first_dt
            minutes = int(delta.total_seconds() / 60)
            return max(0, minutes)
    except Exception:
        pass
    return None


def format_duration(minutes: int | None) -> str:
    """Format duration in minutes to human-readable string.

    Examples:
        None -> ""
        0 -> "<1m"
        5 -> "5m"
        65 -> "1h 5m"
        125 -> "2h 5m"
    """
    if minutes is None:
        return ""
    if minutes == 0:
        return "<1m"
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining = minutes % 60
    if remaining == 0:
        return f"{hours}h"
    return f"{hours}h {remaining}m"
