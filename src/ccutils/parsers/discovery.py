"""Session discovery and project management utilities.

This module provides functions for finding and organizing Claude Code sessions
across project directories. Display and selection functions have been moved
to the tui/ package; thin wrappers here maintain backward compatibility.
"""

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
    """Find all agent sessions related to given parent sessions.

    Agent sessions are identified by:
    - Filename pattern: agent-{agentId}.jsonl
    - Contains sessionId field linking to parent session
    - Has isSidechain: true flag

    Args:
        session_paths: List of parent session Paths
        recursive: If True, also discover agents spawned by agents

    Returns:
        Dict mapping parent session Path to list of agent session Paths.
        When recursive=True, nested agents are flattened under the original parent.
    """
    if not session_paths:
        return {}

    session_paths = [Path(p) for p in session_paths]
    original_set = set(session_paths)
    result = {p: [] for p in session_paths}

    # Build a map of sessionId -> session_path for quick lookup
    session_id_map = {}
    for p in session_paths:
        meta = extract_session_metadata(p)
        if meta.get("sessionId"):
            session_id_map[meta["sessionId"]] = p
        # Also map by file stem
        session_id_map[p.stem] = p

    # Track which original parent each path traces back to
    # (for recursive flattening)
    root_parent_map = {p: p for p in session_paths}

    # Get all directories containing the sessions
    dirs = set(p.parent for p in session_paths)

    # Find all agent files in those directories
    agent_files = []
    for d in dirs:
        agent_files.extend(d.glob("agent-*.jsonl"))

    # Multiple passes to handle recursive discovery
    found_new = True
    processed_agents = set()

    while found_new:
        found_new = False

        for agent_path in agent_files:
            if agent_path in processed_agents:
                continue

            meta = extract_session_metadata(agent_path)
            parent_session_id = meta.get("sessionId")

            if not parent_session_id:
                processed_agents.add(agent_path)
                continue

            # Find the parent session
            parent_path = session_id_map.get(parent_session_id)

            if parent_path is not None:
                processed_agents.add(agent_path)
                found_new = True

                # Find the root parent (original session, not an agent)
                root_parent = root_parent_map.get(parent_path, parent_path)

                # Add agent to the appropriate parent
                if recursive and root_parent in original_set:
                    # Flatten to original parent
                    if agent_path not in result[root_parent]:
                        result[root_parent].append(agent_path)
                    # Track this agent's root parent
                    root_parent_map[agent_path] = root_parent
                else:
                    # Non-recursive: add to immediate parent only
                    if parent_path in result:
                        if agent_path not in result[parent_path]:
                            result[parent_path].append(agent_path)

                # Register this agent in session_id_map so its children can find it
                if recursive:
                    agent_stem = agent_path.stem
                    if agent_stem not in session_id_map:
                        session_id_map[agent_stem] = agent_path

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


def find_local_sessions_rich(folder, limit=100, project_filter=None):
    """Find recent JSONL sessions with rich metadata extraction.

    Returns a list of SessionMetadata objects sorted by modification time.
    Excludes agent files and warmup/empty sessions.

    Args:
        folder: Path to the projects folder
        limit: Maximum number of sessions to return
        project_filter: Optional filter for project names (partial, case-insensitive)

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


def find_all_sessions(folder, include_agents=False, project_filter=None):
    """Find all sessions in a Claude projects folder, grouped by project.

    Returns a list of project dicts, each containing:
    - name: display name for the project
    - path: Path to the project folder
    - sessions: list of session dicts with path, summary, mtime, size

    Sessions are sorted by modification time (most recent first) within each project.
    Projects are sorted by their most recent session.

    Args:
        folder: Path to the projects folder
        include_agents: Whether to include agent-* session files
        project_filter: Optional filter for project names (partial, case-insensitive)
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    projects = {}

    for session_file in folder.glob("**/*.jsonl"):
        # Skip agent files unless requested
        if not include_agents and session_file.name.startswith("agent-"):
            continue

        # Skip projects that don't match filter
        if project_filter and not matches_project_filter(
            session_file.parent.name, project_filter
        ):
            continue

        # Get summary and skip boring sessions
        summary = get_session_summary(session_file)
        if summary.lower() == "warmup" or summary == "(no summary)":
            continue

        # Get project folder
        project_folder = session_file.parent
        project_key = project_folder.name

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
