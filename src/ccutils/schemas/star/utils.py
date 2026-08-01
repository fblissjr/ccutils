"""Shared utilities for star schema operations."""

import hashlib


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


def get_model_base(model_name):
    """Strip a context-window suffix from a model id.

    Claude Code writes context variants as a bracket suffix --
    'claude-opus-5[1m]' is the same model as 'claude-opus-5' with a larger
    window. Without this, the two become separate dim_model rows and every
    per-model aggregate splits silently.

    `model_name` stays byte-faithful to the transcript; this is the grouping
    key.
    """
    if model_name is None:
        return None
    base, _, _ = model_name.partition("[")
    return base.strip() or model_name


def get_model_family(model_name):
    """Extract the model family from a model name.

    Args:
        model_name: Full model name like 'claude-opus-4-5-20251101'

    Returns:
        Model family: 'opus', 'sonnet', 'haiku', 'fable', or 'unknown'

    Parsed STRUCTURALLY from the `claude-<family>-<version...>` naming
    convention rather than matched against a list of known families. An
    enumerated list goes stale the moment a new model line ships: `fable`
    was missing from it, which silently bucketed the corpus's
    third-most-used model -- more output tokens than Opus 5 -- as 'unknown'
    in every GROUP BY model_family. Parsing the convention means the next
    family classifies itself.

    NOTE: mirrored in SQL in etl/orchestrator.py's dim_model insert. Change
    both together -- `tests/test_dim_model_v15.py` asserts they agree.
    """
    if model_name is None:
        return "unknown"
    # Context suffix first: 'claude-opus-5[1m]' -> 'claude-opus-5'.
    base = get_model_base(model_name).lower()
    parts = base.split("-")
    # Anything not shaped like claude-<family>-... is not a Claude model id
    # (e.g. the '<synthetic>' placeholder) and has no family to report.
    if len(parts) < 2 or parts[0] != "claude" or not parts[1]:
        return "unknown"
    return parts[1]


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
