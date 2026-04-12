"""Shared utilities for star schema operations."""

import hashlib
from datetime import datetime


def generate_dimension_key(*natural_keys):
    """Generate a dimension key from natural key(s) using MD5 hash.

    This creates a consistent surrogate key for dimension tables based on
    the natural business key(s). Using a hash allows for:
    - Deterministic key generation (same input = same key)
    - No sequence coordination needed
    - Natural deduplication in dimensions

    Args:
        *natural_keys: One or more values that form the natural key.
                       Multiple values are joined with '|' separator.

    Returns:
        32-character lowercase hex string (MD5 hash)
    """
    key_parts = [str(k) if k is not None else "NULL" for k in natural_keys]
    combined = "|".join(key_parts)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()


# Tool category mapping for dim_tool
TOOL_CATEGORIES = {
    # File operations
    "Read": "file_operations",
    "Write": "file_operations",
    "Edit": "file_operations",
    "MultiEdit": "file_operations",
    "NotebookEdit": "file_operations",
    "Glob": "file_operations",
    # Search tools
    "Grep": "search",
    "WebSearch": "search",
    # Execution tools
    "Bash": "execution",
    "BashOutput": "execution",
    "KillShell": "execution",
    # Web tools
    "WebFetch": "web",
    # Task management
    "Task": "task_management",
    "TodoWrite": "task_management",
    # Planning tools
    "EnterPlanMode": "planning",
    "ExitPlanMode": "planning",
    # Other
    "Skill": "other",
    "SlashCommand": "other",
    "AskUserQuestion": "interaction",
}


def get_tool_category(tool_name):
    """Get the category for a tool name."""
    return TOOL_CATEGORIES.get(tool_name, "other")


def get_model_family(model_name):
    """Extract the model family from a model name.

    Args:
        model_name: Full model name like 'claude-opus-4-5-20251101'

    Returns:
        Model family: 'opus', 'sonnet', 'haiku', or 'unknown'
    """
    if model_name is None:
        return "unknown"
    model_lower = model_name.lower()
    if "opus" in model_lower:
        return "opus"
    elif "sonnet" in model_lower:
        return "sonnet"
    elif "haiku" in model_lower:
        return "haiku"
    return "unknown"


def ts_to_date_key(ts):
    """Convert a datetime to an integer date key (YYYYMMDD)."""
    return int(ts.strftime("%Y%m%d"))


def ts_to_time_key(ts):
    """Convert a datetime to an integer time key (HHMM)."""
    return int(ts.strftime("%H%M"))


_DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def ensure_dim_date(conn, date_key):
    """Insert a dim_date row if it doesn't already exist.

    Args:
        conn: DuckDB connection
        date_key: Integer date key (YYYYMMDD format)
    """
    if conn.execute(
        "SELECT 1 FROM dim_date WHERE date_key = ?", [date_key]
    ).fetchone():
        return
    year = date_key // 10000
    month = (date_key // 100) % 100
    day = date_key % 100
    try:
        full_date = datetime(year, month, day)
        day_of_week = full_date.weekday()
        quarter = (month - 1) // 3 + 1
        is_weekend = day_of_week >= 5
        week_of_year = full_date.isocalendar()[1]
        conn.execute(
            "INSERT INTO dim_date VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                date_key, full_date.date(), year, month, day,
                day_of_week, _DAY_NAMES[day_of_week], _MONTH_NAMES[month - 1],
                quarter, is_weekend, week_of_year,
            ],
        )
    except ValueError:
        pass


def get_time_of_day(hour):
    """Get time of day label from hour.

    Args:
        hour: Hour of day (0-23)

    Returns:
        Time of day label: 'night', 'morning', 'afternoon', 'evening'
    """
    if hour < 6:
        return "night"
    elif hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"
