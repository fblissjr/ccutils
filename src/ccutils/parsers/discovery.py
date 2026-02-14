"""Session discovery and project management utilities.

This module provides functions for finding and organizing Claude Code sessions
across project directories.
"""

import shutil
from datetime import datetime
from pathlib import Path

import questionary
from rich.console import Console
from rich.table import Table
from .metadata import (
    SessionMetadata,
    extract_rich_metadata,
    format_duration,
)
from .session import extract_session_metadata, extract_session_slug, get_session_summary


def get_terminal_width():
    """Get the current terminal width.

    Returns:
        Terminal width in columns, defaults to 80 if unable to determine.
    """
    try:
        return shutil.get_terminal_size().columns
    except (AttributeError, ValueError):
        return 80


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


def build_session_choices(
    sessions_by_project, expand_chains=False, agent_counts=None, flat=False
):
    """Build questionary choices from sessions, with chain grouping support.

    Uses inline project markers instead of separators for better space efficiency.

    Args:
        sessions_by_project: Dict mapping project_key to list of (filepath, summary, slug) tuples
        expand_chains: If False (default), group sessions with same slug into single choice.
                      If True, show individual sessions with chain headers.
        agent_counts: Optional dict mapping filepath to agent count for display
        flat: If True, merge all projects into a single flat list sorted by mtime.
              In flat mode, slug grouping is disabled since chains don't span projects.

    Returns:
        List of questionary.Choice objects with inline project prefixes.
    """
    agent_counts = agent_counts or {}
    choices = []
    terminal_width = get_terminal_width()

    # Calculate max project name width for consistent padding across all projects
    max_project_width = 20

    # In flat mode, merge all sessions into a single list
    if flat:
        all_sessions = []
        for project_key, sessions in sessions_by_project.items():
            project_name = get_project_display_name(project_key)
            for filepath, summary, slug in sessions:
                all_sessions.append((filepath, summary, project_name))
        # Sort by mtime, most recent first (preserves caller's sort if already sorted)
        all_sessions.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)

        # In flat mode, just add all sessions individually with their project markers
        for filepath, summary, project_name in all_sessions:
            display = _format_session_display(
                filepath,
                summary,
                agent_counts.get(filepath, 0),
                project_name=project_name,
                terminal_width=terminal_width,
            )
            choices.append(questionary.Choice(title=display, value=filepath))
        return choices

    # Non-flat mode: process by project with slug grouping
    for project_key, sessions in sessions_by_project.items():
        project_name = get_project_display_name(project_key)

        # Group sessions by slug
        slug_groups = {}  # slug -> list of (filepath, summary, slug)
        standalone = []  # sessions without slug

        for filepath, summary, slug in sessions:
            if slug:
                if slug not in slug_groups:
                    slug_groups[slug] = []
                slug_groups[slug].append((filepath, summary, slug))
            else:
                standalone.append((filepath, summary, slug))

        if expand_chains:
            # Expanded mode: show individual sessions with chain info in display
            for slug, chain_sessions in slug_groups.items():
                # Add individual sessions with project prefix
                for filepath, summary, _ in chain_sessions:
                    # Add chain indicator for multi-session chains
                    chain_suffix = (
                        f" [{len(chain_sessions)}]" if len(chain_sessions) > 1 else ""
                    )
                    display = _format_session_display(
                        filepath,
                        summary + chain_suffix,
                        agent_counts.get(filepath, 0),
                        project_name=project_name,
                        terminal_width=terminal_width,
                    )
                    choices.append(questionary.Choice(title=display, value=filepath))

            # Add standalone sessions
            for filepath, summary, _ in standalone:
                display = _format_session_display(
                    filepath,
                    summary,
                    agent_counts.get(filepath, 0),
                    project_name=project_name,
                    terminal_width=terminal_width,
                )
                choices.append(questionary.Choice(title=display, value=filepath))

        else:
            # Collapsed mode: group chains into single choice
            for slug, chain_sessions in slug_groups.items():
                if len(chain_sessions) > 1:
                    # Create a single choice for the entire chain
                    paths = [s[0] for s in chain_sessions]
                    total_size = sum(p.stat().st_size for p in paths) / 1024

                    # Get date range with times
                    session_stats = [(p, p.stat().st_mtime) for p in paths]
                    session_stats.sort(key=lambda x: x[1])  # Sort by mtime
                    oldest_time = datetime.fromtimestamp(session_stats[0][1])
                    newest_time = datetime.fromtimestamp(session_stats[-1][1])

                    # Find summary from most recent session
                    newest_path = session_stats[-1][0]
                    latest_summary = None
                    for filepath, summary, _ in chain_sessions:
                        if filepath == newest_path:
                            latest_summary = summary
                            break

                    # Format date range
                    if oldest_time.date() == newest_time.date():
                        date_range = f"{oldest_time.strftime('%b %d %H:%M')} - {newest_time.strftime('%H:%M')}"
                    else:
                        date_range = f"{oldest_time.strftime('%b %d %H:%M')} - {newest_time.strftime('%b %d %H:%M')}"

                    # Calculate dynamic truncation for chain summary
                    # Project prefix + chain info takes about 60 chars on line 1
                    available_summary = max(30, terminal_width - 80)
                    if latest_summary and len(latest_summary) > available_summary:
                        latest_summary = latest_summary[: available_summary - 3] + "..."

                    # Format project prefix for consistent width
                    proj_display = project_name
                    if len(proj_display) > max_project_width:
                        proj_display = proj_display[: max_project_width - 2] + ".."
                    proj_prefix = f"[{proj_display}]".ljust(max_project_width + 2)

                    # Multi-line display for better readability
                    line1 = f"{proj_prefix} [{len(chain_sessions)} sessions] {slug}"
                    line2 = f"{''.ljust(max_project_width + 3)}{total_size:,.0f} KB | {date_range}"
                    if latest_summary:
                        line2 += f' | "{latest_summary}"'

                    display = f"{line1}\n{line2}"
                    choices.append(questionary.Choice(title=display, value=paths))
                else:
                    # Single session with slug - treat as standalone
                    filepath, summary, _ = chain_sessions[0]
                    display = _format_session_display(
                        filepath,
                        summary,
                        agent_counts.get(filepath, 0),
                        project_name=project_name,
                        terminal_width=terminal_width,
                    )
                    choices.append(questionary.Choice(title=display, value=filepath))

            # Add standalone sessions
            for filepath, summary, _ in standalone:
                display = _format_session_display(
                    filepath,
                    summary,
                    agent_counts.get(filepath, 0),
                    project_name=project_name,
                    terminal_width=terminal_width,
                )
                choices.append(questionary.Choice(title=display, value=filepath))

    return choices


def _format_session_display(
    filepath, summary, agent_count=0, project_name=None, terminal_width=None
):
    """Format a single session for display in the selection list.

    Args:
        filepath: Path to the session file
        summary: Session summary text
        agent_count: Number of related agent sessions
        project_name: Project name to show as inline prefix (optional)
        terminal_width: Terminal width for dynamic truncation (optional)

    Returns:
        Formatted display string.
    """
    if terminal_width is None:
        terminal_width = get_terminal_width()

    stat = filepath.stat()
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    size_kb = stat.st_size / 1024
    date_str = mod_time.strftime("%Y-%m-%d %H:%M")

    # Build suffix for agents
    suffix = ""
    if agent_count > 0:
        suffix = f" (+{agent_count} agents)"

    # Calculate fixed-width portions
    # Format: [project] date  size KB  summary(suffix)
    # Project prefix: [name] + space = max 22 chars (20 for name + brackets + space)
    # Date: 16 chars (YYYY-MM-DD HH:MM)
    # Size: 9 chars (5 digits + " KB" + 2 spaces)
    # Padding: ~4 chars for spacing

    max_project_width = 20
    project_prefix = ""
    if project_name:
        # Truncate or pad project name for consistent width
        if len(project_name) > max_project_width:
            project_name = project_name[: max_project_width - 2] + ".."
        project_prefix = f"[{project_name}]".ljust(max_project_width + 2) + " "

    fixed_width = len(project_prefix) + 16 + 9 + 4 + len(suffix)

    # Calculate available space for summary
    available = terminal_width - fixed_width
    min_summary_width = 20  # Always show at least this much

    max_summary = max(min_summary_width, available)

    # Truncate summary if needed
    if len(summary) > max_summary:
        summary = summary[: max_summary - 3] + "..."

    return f"{project_prefix}{date_str}  {size_kb:5.0f} KB  {summary}{suffix}"


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


def print_project_table(grouped_sessions, console=None):
    """Print a rich table summarizing available projects.

    Args:
        grouped_sessions: Dict from group_by_project()
        console: Optional Rich Console instance
    """
    if console is None:
        console = Console()

    table = Table(
        title="Projects",
        show_header=True,
        header_style="bold",
        border_style="dim",
        pad_edge=False,
        show_edge=False,
    )
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Sessions", justify="right", style="green")
    table.add_column("Last Active", style="yellow", no_wrap=True)
    table.add_column("Models", style="magenta")
    table.add_column("Branches", style="blue")

    for project_path, sessions in grouped_sessions.items():
        if not sessions:
            continue

        project_name = sessions[0].project_name

        # Collect unique models and branches
        models = set()
        branches = set()
        for s in sessions:
            if s.model_short:
                models.add(s.model_short)
            if s.git_branch:
                branches.add(s.git_branch)

        # Format last active date
        last_active = datetime.fromtimestamp(sessions[0].mtime)
        now = datetime.now()
        if last_active.date() == now.date():
            date_str = "Today"
        elif (now - last_active).days == 1:
            date_str = "Yesterday"
        elif (now - last_active).days < 7:
            date_str = last_active.strftime("%A")  # Day name
        else:
            date_str = last_active.strftime("%b %d")

        table.add_row(
            project_name,
            str(len(sessions)),
            date_str,
            ", ".join(sorted(models)) if models else "-",
            ", ".join(sorted(branches)) if branches else "-",
        )

    console.print(table)
    console.print()


def print_session_table(project_name, sessions, console=None):
    """Print a rich table of sessions for a single project.

    Args:
        project_name: Display name of the project
        sessions: List of SessionMetadata for this project
        console: Optional Rich Console instance
    """
    if console is None:
        console = Console()

    table = Table(
        title=f"{project_name} - {len(sessions)} session(s)",
        show_header=True,
        header_style="bold",
        border_style="dim",
        pad_edge=False,
        show_edge=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Date", style="yellow", no_wrap=True, width=14)
    table.add_column("Model", style="magenta", no_wrap=True, width=12)
    table.add_column("Branch", style="blue", no_wrap=True, width=10)
    table.add_column("Dur", justify="right", style="green", width=6)
    table.add_column("Msgs", justify="right", style="dim", width=4)
    table.add_column("Summary", style="white", no_wrap=False)

    now = datetime.now()
    for idx, s in enumerate(sessions, 1):
        mod_time = datetime.fromtimestamp(s.mtime)

        # Format date relative to now
        if mod_time.date() == now.date():
            date_str = f"Today {mod_time.strftime('%H:%M')}"
        elif (now - mod_time).days == 1:
            date_str = f"Yest {mod_time.strftime('%H:%M')}"
        elif (now - mod_time).days < 7:
            date_str = mod_time.strftime("%a %H:%M")
        else:
            date_str = mod_time.strftime("%b %d %H:%M")

        # Dim old sessions
        style = "dim" if (now - mod_time).days > 7 else ""

        table.add_row(
            str(idx),
            date_str,
            s.model_short or "-",
            s.git_branch or "-",
            format_duration(s.duration_minutes),
            str(s.user_msg_count) if s.user_msg_count > 0 else "-",
            s.summary,
            style=style,
        )

    console.print(table)
    console.print()


def build_project_choices(grouped_sessions):
    """Build questionary checkbox choices for project selection.

    Args:
        grouped_sessions: Dict from group_by_project()

    Returns:
        List of questionary.Choice objects, one per project.
    """
    choices = []
    for project_path, sessions in grouped_sessions.items():
        if not sessions:
            continue
        project_name = sessions[0].project_name
        count = len(sessions)
        label = f"{project_name} ({count} session{'s' if count != 1 else ''})"
        choices.append(questionary.Choice(title=label, value=project_path))
    return choices


def build_session_choices_for_projects(
    sessions, selected_project_paths, expand_chains=False
):
    """Build questionary checkbox choices for sessions within selected projects.

    Args:
        sessions: Full list of SessionMetadata
        selected_project_paths: List of project_path values selected in phase 1
        expand_chains: If True, show individual sessions in chains

    Returns:
        List of questionary.Choice objects for session selection.
    """
    selected_set = set(selected_project_paths)
    filtered = [s for s in sessions if s.project_path in selected_set]
    # Re-sort by mtime (most recent first)
    filtered.sort(key=lambda s: s.mtime, reverse=True)

    # Multiple projects selected -> show project prefix
    multi_project = len(selected_set) > 1

    if not expand_chains:
        # Group by slug for chain collapsing
        slug_groups = {}
        standalone = []
        for s in filtered:
            if s.slug:
                if s.slug not in slug_groups:
                    slug_groups[s.slug] = []
                slug_groups[s.slug].append(s)
            else:
                standalone.append(s)

        choices = []
        seen_slugs = set()

        # Build choices maintaining overall mtime order
        for s in filtered:
            if s.slug and s.slug not in seen_slugs:
                seen_slugs.add(s.slug)
                chain = slug_groups[s.slug]
                if len(chain) > 1:
                    # Collapsed chain
                    label = _format_rich_chain_label(chain, multi_project)
                    paths = [cs.path for cs in chain]
                    choices.append(questionary.Choice(title=label, value=paths))
                else:
                    label = _format_rich_session_label(chain[0], multi_project)
                    choices.append(questionary.Choice(title=label, value=chain[0].path))
            elif not s.slug:
                label = _format_rich_session_label(s, multi_project)
                choices.append(questionary.Choice(title=label, value=s.path))

        return choices
    else:
        # Expanded mode - show each session individually
        choices = []
        for s in filtered:
            label = _format_rich_session_label(s, multi_project)
            choices.append(questionary.Choice(title=label, value=s.path))
        return choices


def _format_rich_session_label(meta, show_project=False):
    """Format a single SessionMetadata for questionary display.

    Args:
        meta: SessionMetadata instance
        show_project: Whether to prefix with project name

    Returns:
        Formatted string for questionary choice title.
    """
    mod_time = datetime.fromtimestamp(meta.mtime)
    now = datetime.now()

    # Compact date
    if mod_time.date() == now.date():
        date_str = f"Today {mod_time.strftime('%H:%M')}"
    elif (now - mod_time).days == 1:
        date_str = f"Yest {mod_time.strftime('%H:%M')}"
    elif (now - mod_time).days < 7:
        date_str = mod_time.strftime("%a %H:%M")
    else:
        date_str = mod_time.strftime("%b %d")

    parts = []
    if show_project:
        proj = meta.project_name
        if len(proj) > 16:
            proj = proj[:14] + ".."
        parts.append(f"[{proj}]")

    parts.append(f"{date_str:>14s}")

    if meta.model_short:
        parts.append(f"{meta.model_short:>10s}")

    if meta.git_branch:
        branch = meta.git_branch
        if len(branch) > 12:
            branch = branch[:10] + ".."
        parts.append(branch)

    dur = format_duration(meta.duration_minutes)
    if dur:
        parts.append(f"{dur:>5s}")

    # Truncate summary for questionary line
    terminal_width = get_terminal_width()
    used = sum(len(p) for p in parts) + len(parts) * 2 + 6  # spacing + checkbox
    available = max(20, terminal_width - used)
    summary = meta.summary
    if len(summary) > available:
        summary = summary[: available - 3] + "..."
    parts.append(summary)

    return "  ".join(parts)


def _format_rich_chain_label(chain, show_project=False):
    """Format a collapsed chain of sessions for questionary display.

    Args:
        chain: List of SessionMetadata with the same slug
        show_project: Whether to prefix with project name

    Returns:
        Formatted string for questionary choice title.
    """
    # Use most recent session for display
    chain.sort(key=lambda s: s.mtime, reverse=True)
    newest = chain[0]

    mod_time = datetime.fromtimestamp(newest.mtime)
    now = datetime.now()

    if mod_time.date() == now.date():
        date_str = f"Today {mod_time.strftime('%H:%M')}"
    elif (now - mod_time).days == 1:
        date_str = f"Yest {mod_time.strftime('%H:%M')}"
    elif (now - mod_time).days < 7:
        date_str = mod_time.strftime("%a %H:%M")
    else:
        date_str = mod_time.strftime("%b %d")

    parts = []
    if show_project:
        proj = newest.project_name
        if len(proj) > 16:
            proj = proj[:14] + ".."
        parts.append(f"[{proj}]")

    parts.append(f"{date_str:>14s}")

    chain_tag = f"[{len(chain)} resumed]"
    parts.append(chain_tag)

    if newest.model_short:
        parts.append(f"{newest.model_short:>10s}")

    # Total duration across chain
    total_dur = sum(s.duration_minutes or 0 for s in chain)
    if total_dur > 0:
        parts.append(f"{format_duration(total_dur):>5s}")

    # Summary from newest session
    terminal_width = get_terminal_width()
    used = sum(len(p) for p in parts) + len(parts) * 2 + 6
    available = max(20, terminal_width - used)
    summary = newest.summary
    if len(summary) > available:
        summary = summary[: available - 3] + "..."
    parts.append(summary)

    return "  ".join(parts)


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
