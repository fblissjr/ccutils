"""Pure formatting functions for TUI display. No I/O, no side effects.

All functions accept data and return strings. Terminal width is passed
explicitly where needed rather than detected.
"""

from datetime import datetime

from ..parsers.metadata import format_duration as _format_duration


def format_relative_date(mtime: float) -> str:
    """Format a modification timestamp as a relative date string.

    Examples:
        Same day:      "Today 14:30"
        Yesterday:     "Yest 09:15"
        This week:     "Mon 16:00"
        Older:         "Feb 10"

    Args:
        mtime: Unix timestamp (seconds since epoch).

    Returns:
        Human-readable relative date string.
    """
    mod_time = datetime.fromtimestamp(mtime)
    now = datetime.now()

    if mod_time.date() == now.date():
        return f"Today {mod_time.strftime('%H:%M')}"
    elif (now - mod_time).days == 1:
        return f"Yest {mod_time.strftime('%H:%M')}"
    elif (now - mod_time).days < 7:
        return mod_time.strftime("%a %H:%M")
    else:
        return mod_time.strftime("%b %d")


def format_relative_date_short(mtime: float) -> str:
    """Format a modification timestamp as a short relative date (no time).

    Used for project-level "last active" display.

    Examples:
        Same day:      "Today"
        Yesterday:     "Yesterday"
        This week:     "Monday"
        Older:         "Feb 10"
    """
    mod_time = datetime.fromtimestamp(mtime)
    now = datetime.now()

    if mod_time.date() == now.date():
        return "Today"
    elif (now - mod_time).days == 1:
        return "Yesterday"
    elif (now - mod_time).days < 7:
        return mod_time.strftime("%A")
    else:
        return mod_time.strftime("%b %d")


def format_duration(minutes: int | None) -> str:
    """Format duration in minutes to human-readable string.

    Delegates to metadata.format_duration().
    """
    return _format_duration(minutes)


def format_project_name(name: str, max_width: int = 20) -> str:
    """Truncate project name to fit max_width.

    Args:
        name: Project display name.
        max_width: Maximum allowed width.

    Returns:
        Name truncated with ".." if necessary, otherwise unchanged.
    """
    if len(name) <= max_width:
        return name
    return name[: max_width - 2] + ".."


def format_summary(summary: str, available_width: int) -> str:
    """Truncate summary text to fit available terminal width.

    Args:
        summary: Session summary text.
        available_width: Maximum allowed width.

    Returns:
        Summary truncated with "..." if necessary, otherwise unchanged.
    """
    if available_width < 10:
        available_width = 10
    if len(summary) <= available_width:
        return summary
    return summary[: available_width - 3] + "..."


def format_branch(branch: str | None, max_width: int = 12) -> str:
    """Truncate branch name to fit max_width.

    Args:
        branch: Git branch name, or None.
        max_width: Maximum allowed width.

    Returns:
        Branch truncated with ".." if necessary, "-" if None.
    """
    if not branch:
        return "-"
    if len(branch) <= max_width:
        return branch
    return branch[: max_width - 2] + ".."


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string like "1.2 MB", "450 KB", "800 B".
    """
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KB"
    else:
        return f"{size_bytes} B"


def format_msg_count(user: int, assistant: int) -> str:
    """Format message counts as user/assistant ratio.

    Args:
        user: Number of user messages.
        assistant: Number of assistant messages.

    Returns:
        Formatted string like "12/8", or "-" if no messages.
    """
    if user == 0 and assistant == 0:
        return "-"
    return f"{user}/{assistant}"
