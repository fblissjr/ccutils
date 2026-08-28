"""Session discovery and project management utilities.

This module provides functions for finding and organizing Claude Code sessions
across project directories. Display and selection functions have been moved
to the tui/ package; thin wrappers here maintain backward compatibility.
"""

import tempfile
from pathlib import Path

from .metadata import (
    extract_rich_metadata,
)
from .session import extract_session_metadata, extract_session_slug, get_session_summary

# ---------------------------------------------------------------------------
# Session discovery functions (data only, no display)
# ---------------------------------------------------------------------------


def find_local_sessions(folder, limit=10, project_filter=None):
    """Find recent JSONL session files in the given folder.

    Returns a list of (Path, summary, slug) tuples sorted by modification time.
    Excludes agent files and warmup/empty sessions.
    Sessions with the same slug are part of the same conversation chain (resumed sessions).

    Args:
        folder: Path to the projects folder
        limit: Maximum number of sessions to return
        project_filter: Optional filter for project names (partial, case-insensitive)
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    results = []
    for f in folder.glob("**/*.jsonl"):
        if f.name.startswith("agent-"):
            continue
        if project_filter and not matches_project_filter(f.parent.name, project_filter):
            continue
        summary = get_session_summary(f)
        # Skip boring/empty sessions
        if summary.lower() == "warmup" or summary == "(no summary)":
            continue
        slug = extract_session_slug(f)
        results.append((f, summary, slug))

    # Sort by modification time, most recent first
    results.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    return results[:limit]


def flatten_selected_sessions(selected):
    """Flatten selected sessions which may include chains (lists) or single paths.

    In collapsed chain mode, selecting a chain returns a list of all session paths.
    This function flattens such mixed selections into a single list of paths.

    Args:
        selected: List of items - each item is either a Path or a list of Paths

    Returns:
        Flattened list of all session Paths.
    """
    result = []
    for item in selected:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def find_agent_sessions(session_paths, recursive=True):
    """Find the subagent transcripts belonging to each given session.

    A session's agents live in a directory named after the session, not
    beside it::

        <project>/<parent-uuid>.jsonl
        <project>/<parent-uuid>/subagents/agent-<id>.jsonl
        <project>/<parent-uuid>/subagents/workflows/<wf-id>/agent-<id>.jsonl

    so the search is rooted at ``<parent>/<stem>/subagents`` and walks down
    from there -- workflow-tool agents nest one level deeper, and any future
    grouping directory is covered by the same walk. This function previously
    globbed the session's OWN directory, which is the pre-2026 layout: it
    matched nothing on real data, and `local` exported no subagents at all.

    Ownership comes from the directory, never from the transcript's
    ``sessionId`` -- agent entries carry the PARENT's sessionId on every
    line, so that field cannot distinguish one agent from another.

    Args:
        session_paths: List of parent session Paths.
        recursive: Retained for compatibility and no longer selects
            anything. Nested agents are siblings of their spawner in the
            same ``subagents`` directory and carry the same root sessionId,
            so depth is not derivable from the files scanned here. Stated
            depth lives in each agent's ``.meta.json`` ``spawnDepth``.

    Returns:
        Dict mapping each given session Path to its list of agent Paths.
    """
    if not session_paths:
        return {}

    result = {}
    for session_path in (Path(p) for p in session_paths):
        agent_root = session_path.parent / session_path.stem / "subagents"
        agents = (
            sorted(f for f in agent_root.rglob("agent-*.jsonl") if f.is_file())
            if agent_root.is_dir()
            else []
        )
        result[session_path] = agents

    return result


def get_project_display_name(folder_name):
    """Convert encoded folder name to readable project name.

    Claude Code stores projects in folders like:
    - -home-user-projects-myproject -> myproject
    - -mnt-c-Users-name-Projects-app -> app

    For nested paths under common roots (home, projects, code, Users, etc.),
    extracts the meaningful project portion.
    """
    # Common path prefixes to strip
    prefixes_to_strip = [
        "-home-",
        "-mnt-c-Users-",
        "-mnt-c-users-",
        "-Users-",
    ]

    name = folder_name
    for prefix in prefixes_to_strip:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix) :]
            break

    # Split on dashes and find meaningful parts
    parts = name.split("-")

    # Common intermediate directories to skip
    skip_dirs = {"projects", "code", "repos", "src", "dev", "work", "documents"}

    # Find the first meaningful part (after skipping username and common dirs)
    meaningful_parts = []
    found_project = False

    for i, part in enumerate(parts):
        if not part:
            continue
        # Skip the first part if it looks like a username (before common dirs)
        if i == 0 and not found_project:
            # Check if next parts contain common dirs
            remaining = [p.lower() for p in parts[i + 1 :]]
            if any(d in remaining for d in skip_dirs):
                continue
        if part.lower() in skip_dirs:
            found_project = True
            continue
        meaningful_parts.append(part)
        found_project = True

    if meaningful_parts:
        return "-".join(meaningful_parts)

    # Fallback: return last non-empty part or original
    for part in reversed(parts):
        if part:
            return part
    return folder_name


def is_temp_dir_cwd(cwd: str | None) -> bool:
    """True when a session's cwd resolves under the OS temp directory.

    Sessions run from a temp dir are sandboxed/ephemeral tooling (eval
    harnesses, CI scratch runs) rather than real projects -- ingesting them
    pollutes session counts, intent/domain classification, and facets with
    synthetic, non-representative activity. Only a real prefix match counts
    (a project literally named "my-tmp-experiments" must not false-positive).
    """
    if not cwd:
        return False
    normalized = cwd.rstrip("/") + "/"
    prefixes = [
        "/tmp/", "/private/tmp/",
        "/var/folders/", "/private/var/folders/",
    ]
    temp_dir = tempfile.gettempdir()
    if temp_dir:
        prefixes.append(temp_dir.rstrip("/") + "/")
    return any(normalized.startswith(p) for p in prefixes)


def matches_project_filter(folder_name: str, project_filter: str | None) -> bool:
    """Check if project folder matches filter (partial, case-insensitive).

    Matches against both the display name AND the raw folder name for
    better discoverability (e.g., searching "claude-code" will match
    even if display name is "workspace-claude-transcripts").

    Args:
        folder_name: The raw folder name (e.g., "-home-user-projects-myproject")
        project_filter: Filter string to match against, or None for no filtering

    Returns:
        True if the filter matches or is None/empty, False otherwise
    """
    if not project_filter:
        return True
    filter_lower = project_filter.lower()
    display_name = get_project_display_name(folder_name)
    # Match against display name OR raw folder name
    return filter_lower in display_name.lower() or filter_lower in folder_name.lower()


def find_local_sessions_rich(
    folder, limit=100, project_filter=None, include_temp_sessions=False
):
    """Find recent JSONL sessions with rich metadata extraction.

    Returns a list of SessionMetadata objects sorted by modification time.
    Excludes agent files and warmup/empty sessions.

    Args:
        folder: Path to the projects folder
        limit: Maximum number of sessions to return
        project_filter: Optional filter for project names (partial, case-insensitive)
        include_temp_sessions: When False (default), sessions whose cwd is
            under the OS temp directory are excluded -- see is_temp_dir_cwd.
            Filtered before the limit is applied so real recent sessions
            aren't crowded out of the top-N by sandboxed/ephemeral runs.

    Returns:
        List of SessionMetadata objects, sorted by mtime (most recent first).
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    results = []
    for f in sorted(
        folder.glob("**/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        if f.name.startswith("agent-"):
            continue
        if project_filter and not matches_project_filter(f.parent.name, project_filter):
            continue

        meta = extract_rich_metadata(f, f.parent.name)

        if not include_temp_sessions and is_temp_dir_cwd(meta.cwd):
            continue

        # Skip boring/empty sessions
        if meta.summary.lower() == "warmup" or meta.summary == "(no summary)":
            continue

        results.append(meta)
        if len(results) >= limit:
            break

    return results


def group_by_project(sessions):
    """Group SessionMetadata list by project_path.

    Args:
        sessions: List of SessionMetadata objects

    Returns:
        Dict mapping project_path to list of SessionMetadata, ordered by most
        recent session activity.
    """
    groups = {}
    for s in sessions:
        if s.project_path not in groups:
            groups[s.project_path] = []
        groups[s.project_path].append(s)

    # Sort each group by mtime (most recent first)
    for group in groups.values():
        group.sort(key=lambda s: s.mtime, reverse=True)

    # Sort groups by most recent session
    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: groups[k][0].mtime if groups[k] else 0,
        reverse=True,
    )
    return {k: groups[k] for k in sorted_keys}


def is_curated_out(summary):
    """The render-format curation rule: warmup / no-summary sessions are
    skipped by browsable exports (html/markdown). Warehouse batch paths
    ingest everything. Single source of truth for both sides."""
    return summary.lower() == "warmup" or summary == "(no summary)"


def curate_projects(projects):
    """Apply the render-format curation rule to a find_all_sessions list.

    Returns a new list with curated-out sessions removed and projects
    left empty by the filter dropped. Used by the html/markdown batch
    exporters when handed a pre-scanned (possibly complete) list.
    """
    curated = []
    for project in projects:
        sessions = [
            s for s in project["sessions"] if not is_curated_out(s["summary"])
        ]
        if sessions:
            curated.append({**project, "sessions": sessions})
    return curated


def find_all_sessions(
    folder, include_agents=False, project_filter=None, include_unsummarized=False,
    include_temp_sessions=False,
):
    """Find all sessions in a Claude projects folder, grouped by project.

    Returns a list of project dicts, each containing:
    - name: display name for the project
    - path: Path to the project folder
    - sessions: list of session dicts with path, summary, mtime, size

    Sessions are sorted by modification time (most recent first) within each project.
    Projects are sorted by their most recent session.

    Project attribution mirrors the warehouse's ``project_dir_sql``
    (etl/utils.py): the file's parent directory, walking up past any
    ``<seg>/subagents`` layers. Subagent files live at
    ``<project>/<parent-uuid>/subagents/agent-*.jsonl`` -- grouping by
    bare parent dir would lump every subagent across all projects into a
    synthetic "subagents" project (and hide them from ``project_filter``).
    Keeping the two rules identical means the picker/-p taxonomy and
    ``dim_project`` cannot disagree, whatever layout is scanned.

    Args:
        folder: Path to the projects folder
        include_agents: Whether to include agent-* session files
        project_filter: Optional filter for project names (partial, case-insensitive)
        include_unsummarized: When True, keep sessions whose summary is
            "warmup" or "(no summary)". The curated default suits browsable
            exports; warehouse batch runs pass True so coverage is complete.
        include_temp_sessions: When False (default), sessions whose cwd is
            under the OS temp directory are excluded -- see is_temp_dir_cwd.
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    projects = {}

    for session_file in folder.glob("**/*.jsonl"):
        # Skip agent files unless requested
        if not include_agents and session_file.name.startswith("agent-"):
            continue

        # Python mirror of etl/utils.py::project_dir_sql -- keep in sync.
        # Cut at the LAST "subagents" segment each pass (a grouping dir may
        # sit below it -- workflow agents live in subagents/workflows/<id>/)
        # and loop for nested delegation.
        project_folder = session_file.parent
        while "subagents" in project_folder.parts:
            parts = project_folder.parts
            cut = len(parts) - 1 - parts[::-1].index("subagents")
            project_folder = Path(*parts[: cut - 1])
        project_key = project_folder.name

        # Skip projects that don't match filter
        if project_filter and not matches_project_filter(
            project_key, project_filter
        ):
            continue

        if not include_temp_sessions and is_temp_dir_cwd(
            extract_session_metadata(session_file).get("cwd")
        ):
            continue

        # Get summary and (by default) skip boring sessions
        summary = get_session_summary(session_file)
        if not include_unsummarized and is_curated_out(summary):
            continue

        if project_key not in projects:
            projects[project_key] = {
                "name": get_project_display_name(project_key),
                "path": project_folder,
                "sessions": [],
            }

        stat = session_file.stat()
        projects[project_key]["sessions"].append(
            {
                "path": session_file,
                "summary": summary,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        )

    # Sort sessions within each project by mtime (most recent first)
    for project in projects.values():
        project["sessions"].sort(key=lambda s: s["mtime"], reverse=True)

    # Convert to list and sort projects by most recent session
    result = list(projects.values())
    result.sort(
        key=lambda p: p["sessions"][0]["mtime"] if p["sessions"] else 0, reverse=True
    )

    return result
