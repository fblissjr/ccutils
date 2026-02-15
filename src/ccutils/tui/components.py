"""Rich console display components for project and session tables.

These functions render Rich tables with semantic coloring and
responsive column widths.
"""

from rich.console import Console
from rich.table import Table

from ..parsers.metadata import format_duration
from .formatters import (
    format_branch,
    format_msg_count,
    format_relative_date,
    format_relative_date_short,
)
from .theme import RICH_STYLES


def render_project_table(grouped_sessions, console=None):
    """Print a rich table summarizing available projects.

    Args:
        grouped_sessions: Dict from group_by_project(), mapping project_path
            to list of SessionMetadata objects.
        console: Optional Rich Console instance.
    """
    if console is None:
        console = Console()

    total_sessions = sum(len(v) for v in grouped_sessions.values())
    total_projects = len(grouped_sessions)

    table = Table(
        title=f"Projects ({total_projects} found, {total_sessions} sessions)",
        show_header=True,
        header_style="bold",
        border_style="dim",
        pad_edge=False,
        show_edge=False,
        expand=True,
    )
    table.add_column("Name", style=RICH_STYLES["identity"], no_wrap=True, ratio=3)
    table.add_column("Sessions", justify="right", style=RICH_STYLES["metric"], ratio=1)
    table.add_column(
        "Last Active", style=RICH_STYLES["temporal"], no_wrap=True, ratio=1
    )
    table.add_column("Models", style=RICH_STYLES["model"], ratio=2)
    table.add_column("Branches", style=RICH_STYLES["identity"], ratio=2)

    for project_path, sessions in grouped_sessions.items():
        if not sessions:
            continue

        project_name = sessions[0].project_name

        models = set()
        branches = set()
        for s in sessions:
            if s.model_short:
                models.add(s.model_short)
            if s.git_branch:
                branches.add(s.git_branch)

        date_str = format_relative_date_short(sessions[0].mtime)

        table.add_row(
            project_name,
            str(len(sessions)),
            date_str,
            ", ".join(sorted(models)) if models else "-",
            ", ".join(sorted(branches)) if branches else "-",
        )

    console.print(table)
    console.print()


def render_session_table(project_name, sessions, console=None):
    """Print a rich table of sessions for a single project.

    Uses ratio-based columns so that summary gets remaining space
    on wide terminals.

    Args:
        project_name: Display name of the project.
        sessions: List of SessionMetadata for this project.
        console: Optional Rich Console instance.
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
        expand=True,
    )
    table.add_column("#", justify="right", style=RICH_STYLES["secondary"], width=3)
    table.add_column("Date", style=RICH_STYLES["temporal"], no_wrap=True, width=14)
    table.add_column("Model", style=RICH_STYLES["model"], no_wrap=True, width=12)
    table.add_column("Branch", style=RICH_STYLES["identity"], no_wrap=True, width=12)
    table.add_column("Dur", justify="right", style=RICH_STYLES["metric"], width=7)
    table.add_column("Msgs", justify="right", style=RICH_STYLES["metric"], width=5)
    table.add_column("Summary", style=RICH_STYLES["primary"], no_wrap=False, ratio=1)

    from datetime import datetime

    now = datetime.now()
    for idx, s in enumerate(sessions, 1):
        mod_time = datetime.fromtimestamp(s.mtime)
        date_str = format_relative_date(s.mtime)

        # Dim old sessions
        style = RICH_STYLES["secondary"] if (now - mod_time).days > 7 else ""

        table.add_row(
            str(idx),
            date_str,
            s.model_short or "-",
            format_branch(s.git_branch),
            format_duration(s.duration_minutes),
            format_msg_count(s.user_msg_count, s.assistant_msg_count),
            s.summary,
            style=style,
        )

    console.print(table)
    console.print()


def render_status_header(total_sessions, total_projects, console=None):
    """Print a summary status bar above selection UI.

    Args:
        total_sessions: Total number of sessions found.
        total_projects: Total number of projects found.
        console: Optional Rich Console instance.
    """
    if console is None:
        console = Console()

    console.print(
        f"[{RICH_STYLES['metric']}]{total_sessions}[/] sessions across "
        f"[{RICH_STYLES['metric']}]{total_projects}[/] projects",
        style="dim",
    )
