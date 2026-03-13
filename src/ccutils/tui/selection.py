"""Questionary choice builders with styled FormattedText labels.

Each builder returns a list of questionary.Choice objects whose `title`
is a list of (style_string, text) tuples. When questionary renders these,
it calls `tokens.extend(choice.title)` directly, preserving per-segment
colors even when highlighted or selected.
"""

import questionary

from ..parsers.metadata import format_duration
from .formatters import (
    format_branch,
    format_project_name,
    format_relative_date,
    format_summary,
)
from .layout import ColumnSpec, calculate_column_widths, get_terminal_width
from .theme import STYLES, model_style_key

# ---------------------------------------------------------------------------
# Internal label builders (return list[tuple[str, str]])
# ---------------------------------------------------------------------------


def _styled(role, text):
    """Return a single (style, text) tuple for the given semantic role."""
    return (STYLES.get(role, ""), text)


def _project_label(
    meta_or_name, session_count, last_mtime, models, branches, terminal_width=None
):
    """Build a styled label for project selection.

    Column widths are computed proportionally based on terminal width.
    Project name and models/branches flex to fill available space.

    Args:
        meta_or_name: Project display name string.
        session_count: Number of sessions in this project.
        last_mtime: mtime of most recent session.
        models: Set of model short names.
        branches: Set of branch names.
        terminal_width: Override terminal width (for testing).

    Returns:
        List of (style, text) tuples for FormattedText.
    """
    if terminal_width is None:
        terminal_width = get_terminal_width()

    # Compute column widths proportionally.
    # checkbox chrome eats ~6 chars, subtract before calculating.
    usable = terminal_width - 6
    cols = [
        ColumnSpec(name="name", min_width=15, ratio=2.0),
        ColumnSpec(name="count", min_width=12, fixed=True),
        ColumnSpec(name="date", min_width=10, fixed=True),
        ColumnSpec(name="models", min_width=10, ratio=1.5),
        ColumnSpec(name="branches", min_width=4, ratio=1.0),
    ]
    widths = calculate_column_widths(cols, usable, padding=2)

    tokens = []

    # Project name (bold blue)
    w_name = widths["name"]
    name = format_project_name(meta_or_name, max_width=w_name)
    tokens.append(_styled("identity.bold", f"{name:<{w_name}s}"))
    tokens.append(_styled("primary", "  "))

    # Session count (green)
    count_str = f"{session_count} session{'s' if session_count != 1 else ''}"
    w_count = widths["count"]
    tokens.append(_styled("metric", f"{count_str:<{w_count}s}"))

    # Last active date (yellow)
    from .formatters import format_relative_date_short

    date_str = format_relative_date_short(last_mtime)
    w_date = widths["date"]
    tokens.append(_styled("temporal", f"{date_str:<{w_date}s}"))

    # Models (magenta)
    w_models = widths["models"]
    if models:
        model_str = ", ".join(sorted(models))
        if len(model_str) > w_models:
            model_str = model_str[: w_models - 2] + ".."
        first_model = sorted(models)[0]
        style_key = model_style_key(first_model)
        tokens.append(
            (STYLES.get(style_key, STYLES["model"]), f"{model_str:<{w_models}s}")
        )
    else:
        tokens.append(_styled("secondary", f"{'-':<{w_models}s}"))

    # Branches (blue)
    w_branches = widths["branches"]
    if branches:
        branch_str = ", ".join(sorted(branches))
        if len(branch_str) > w_branches:
            branch_str = branch_str[: w_branches - 2] + ".."
        tokens.append(_styled("identity", branch_str))
    else:
        tokens.append(_styled("secondary", "-"))

    return tokens


def _session_label(meta, show_project=False, terminal_width=None):
    """Build a styled label for a single session.

    Args:
        meta: SessionMetadata instance.
        show_project: Whether to prefix with project name.
        terminal_width: Terminal width for summary truncation.

    Returns:
        List of (style, text) tuples for FormattedText.
    """
    if terminal_width is None:
        terminal_width = get_terminal_width()

    tokens = []
    used = 6  # checkbox chrome

    # Compute project prefix width proportionally when in flat mode
    if show_project:
        # Scale project name width: ~15% of terminal, min 12, max 28
        proj_width = max(12, min(28, terminal_width // 7))
        proj = format_project_name(meta.project_name, max_width=proj_width)
        tokens.append(_styled("identity", f"[{proj}]"))
        tokens.append(_styled("primary", " "))
        used += len(proj) + 3

    # Date (yellow, right-aligned in 14 chars)
    date_str = format_relative_date(meta.mtime)
    tokens.append(_styled("temporal", f"{date_str:>14s}"))
    tokens.append(_styled("primary", "  "))
    used += 16

    # Model (magenta)
    if meta.model_short:
        style_key = model_style_key(meta.model_short)
        tokens.append(
            (STYLES.get(style_key, STYLES["model"]), f"{meta.model_short:>10s}")
        )
        tokens.append(_styled("primary", "  "))
        used += 12
    else:
        tokens.append(_styled("secondary", f"{'':>10s}"))
        tokens.append(_styled("primary", "  "))
        used += 12

    # Branch (blue) - scale width: ~10% of terminal, min 6, max 20
    if meta.git_branch:
        branch_width = max(6, min(20, terminal_width // 10))
        branch = format_branch(meta.git_branch, max_width=branch_width)
        tokens.append(_styled("identity", f"{branch:<{branch_width}s}"))
        tokens.append(_styled("primary", " "))
        used += branch_width + 1
    else:
        used += 0  # skip branch if missing

    # Duration (green)
    dur = format_duration(meta.duration_minutes)
    if dur:
        tokens.append(_styled("metric", f"{dur:>6s}"))
        tokens.append(_styled("primary", "  "))
        used += 8

    # Summary (default, fills remaining width)
    available = max(20, terminal_width - used)
    summary = format_summary(meta.summary, available)
    tokens.append(_styled("primary", summary))

    return tokens


def _chain_label(chain, show_project=False, terminal_width=None):
    """Build a styled label for a collapsed chain of resumed sessions.

    Args:
        chain: List of SessionMetadata with the same slug, sorted by mtime desc.
        show_project: Whether to prefix with project name.
        terminal_width: Terminal width for summary truncation.

    Returns:
        List of (style, text) tuples for FormattedText.
    """
    if terminal_width is None:
        terminal_width = get_terminal_width()

    chain.sort(key=lambda s: s.mtime, reverse=True)
    newest = chain[0]

    tokens = []
    used = 6  # checkbox chrome

    # Optional project prefix (same scaling as _session_label)
    if show_project:
        proj_width = max(12, min(28, terminal_width // 7))
        proj = format_project_name(newest.project_name, max_width=proj_width)
        tokens.append(_styled("identity", f"[{proj}]"))
        tokens.append(_styled("primary", " "))
        used += len(proj) + 3

    # Date
    date_str = format_relative_date(newest.mtime)
    tokens.append(_styled("temporal", f"{date_str:>14s}"))
    tokens.append(_styled("primary", "  "))
    used += 16

    # Chain indicator (italic yellow)
    chain_tag = f"[{len(chain)} resumed]"
    tokens.append(_styled("chain", chain_tag))
    tokens.append(_styled("primary", "  "))
    used += len(chain_tag) + 2

    # Model
    if newest.model_short:
        style_key = model_style_key(newest.model_short)
        tokens.append(
            (STYLES.get(style_key, STYLES["model"]), f"{newest.model_short:>10s}")
        )
        tokens.append(_styled("primary", "  "))
        used += 12

    # Total duration across chain
    total_dur = sum(s.duration_minutes or 0 for s in chain)
    if total_dur > 0:
        dur_str = format_duration(total_dur)
        tokens.append(_styled("metric", f"{dur_str:>6s}"))
        tokens.append(_styled("primary", "  "))
        used += 8

    # Summary from newest
    available = max(20, terminal_width - used)
    summary = format_summary(newest.summary, available)
    tokens.append(_styled("primary", summary))

    return tokens


# ---------------------------------------------------------------------------
# Public choice builders
# ---------------------------------------------------------------------------


def build_project_choices(grouped_sessions):
    """Build questionary checkbox choices for project selection.

    Each choice title is a FormattedText list with semantic coloring.
    Column widths are computed once from terminal width and shared
    across all choices for consistent alignment.

    Args:
        grouped_sessions: Dict from group_by_project(), mapping project_path
            to list of SessionMetadata objects.

    Returns:
        List of questionary.Choice objects, one per project.
    """
    terminal_width = get_terminal_width()
    choices = []
    for project_path, sessions in grouped_sessions.items():
        if not sessions:
            continue

        project_name = sessions[0].project_name
        count = len(sessions)

        models = set()
        branches = set()
        for s in sessions:
            if s.model_short:
                models.add(s.model_short)
            if s.git_branch:
                branches.add(s.git_branch)

        label = _project_label(
            project_name, count, sessions[0].mtime, models, branches, terminal_width
        )
        choices.append(questionary.Choice(title=label, value=project_path))

    return choices


def build_session_choices(sessions, selected_projects, expand_chains=False):
    """Build questionary checkbox choices for sessions within selected projects.

    Replaces discovery.build_session_choices_for_projects().

    Args:
        sessions: Full list of SessionMetadata.
        selected_projects: List of project_path values selected in phase 1.
        expand_chains: If True, show individual sessions in chains.

    Returns:
        List of questionary.Choice objects for session selection.
    """
    selected_set = set(selected_projects)
    filtered = [s for s in sessions if s.project_path in selected_set]
    filtered.sort(key=lambda s: s.mtime, reverse=True)

    multi_project = len(selected_set) > 1
    terminal_width = get_terminal_width()

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

        for s in filtered:
            if s.slug and s.slug not in seen_slugs:
                seen_slugs.add(s.slug)
                chain = slug_groups[s.slug]
                if len(chain) > 1:
                    label = _chain_label(chain, multi_project, terminal_width)
                    paths = [cs.path for cs in chain]
                    choices.append(questionary.Choice(title=label, value=paths))
                else:
                    label = _session_label(chain[0], multi_project, terminal_width)
                    choices.append(questionary.Choice(title=label, value=chain[0].path))
            elif not s.slug:
                label = _session_label(s, multi_project, terminal_width)
                choices.append(questionary.Choice(title=label, value=s.path))

        return choices
    else:
        choices = []
        for s in filtered:
            label = _session_label(s, multi_project, terminal_width)
            choices.append(questionary.Choice(title=label, value=s.path))
        return choices


def build_flat_choices(grouped_sessions, expand_chains=False):
    """Build questionary choices for flat mode (all projects merged).

    All sessions across projects are merged into a single mtime-sorted list.
    Each label shows [project_name] prefix via _session_label(show_project=True).
    Chain collapsing is supported when expand_chains=False.

    Args:
        grouped_sessions: Dict from group_by_project(), mapping project_path
            to list of SessionMetadata objects.
        expand_chains: If True, show individual sessions in chains.

    Returns:
        List of questionary.Choice objects for flat session selection.
    """
    terminal_width = get_terminal_width()

    # Merge all sessions across projects into one list
    all_sessions = []
    for sessions in grouped_sessions.values():
        all_sessions.extend(sessions)
    all_sessions.sort(key=lambda s: s.mtime, reverse=True)

    if expand_chains:
        choices = []
        for s in all_sessions:
            label = _session_label(s, show_project=True, terminal_width=terminal_width)
            choices.append(questionary.Choice(title=label, value=s.path))
        return choices

    # Chain collapsing: group by slug, then emit in mtime order
    slug_groups = {}
    for s in all_sessions:
        if s.slug:
            if s.slug not in slug_groups:
                slug_groups[s.slug] = []
            slug_groups[s.slug].append(s)

    choices = []
    seen_slugs = set()

    for s in all_sessions:
        if s.slug and s.slug not in seen_slugs:
            seen_slugs.add(s.slug)
            chain = slug_groups[s.slug]
            if len(chain) > 1:
                label = _chain_label(
                    chain, show_project=True, terminal_width=terminal_width
                )
                paths = [cs.path for cs in chain]
                choices.append(questionary.Choice(title=label, value=paths))
            else:
                label = _session_label(
                    chain[0], show_project=True, terminal_width=terminal_width
                )
                choices.append(questionary.Choice(title=label, value=chain[0].path))
        elif not s.slug:
            label = _session_label(s, show_project=True, terminal_width=terminal_width)
            choices.append(questionary.Choice(title=label, value=s.path))

    return choices


def build_web_session_choices(sessions):
    """Build styled questionary choices for web API session picker.

    Args:
        sessions: List of session dicts from the Claude API.

    Returns:
        List of questionary.Choice objects with styled labels.
    """
    choices = []
    for s in sessions:
        sid = s.get("id", "unknown")
        title = s.get("title", "Untitled")
        created_at = s.get("created_at", "")
        repo = s.get("repo")

        tokens = []

        # Repo (blue) or placeholder
        repo_display = repo if repo else "(no repo)"
        tokens.append(_styled("identity", f"{repo_display:<30s}"))
        tokens.append(_styled("primary", "  "))

        # Date (yellow)
        date_display = created_at[:19] if created_at else "N/A"
        tokens.append(_styled("temporal", f"{date_display:<19s}"))
        tokens.append(_styled("primary", "  "))

        # Title (default)
        if len(title) > 50:
            title = title[:47] + "..."
        tokens.append(_styled("primary", title))

        choices.append(questionary.Choice(title=tokens, value=sid))

    return choices


def build_import_choices(conversations):
    """Build styled questionary choices for Claude.ai export import.

    Args:
        conversations: List of conversation dicts from parsed export.

    Returns:
        List of questionary.Choice objects with styled labels.
    """
    choices = []
    sorted_convs = sorted(
        conversations, key=lambda c: c.get("updated_at", ""), reverse=True
    )

    for conv in sorted_convs:
        name = conv.get("name", "(untitled)")
        uuid = conv.get("uuid", "")
        msg_count = len(conv.get("chat_messages", []))
        updated = conv.get("updated_at", "")[:10]

        tokens = []

        # Date (yellow)
        tokens.append(_styled("temporal", f"{updated:<10s}"))
        tokens.append(_styled("primary", " "))

        # Message count (green)
        tokens.append(_styled("metric", f"({msg_count:3d} msgs)"))
        tokens.append(_styled("primary", " "))

        # Name (default)
        if len(name) > 50:
            name = name[:47] + "..."
        tokens.append(_styled("primary", name))

        choices.append(questionary.Choice(title=tokens, value=uuid))

    return choices
