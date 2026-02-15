"""Terminal-responsive layout utilities.

Handles terminal width detection and proportional column width calculation.
"""

import shutil
from dataclasses import dataclass


def get_terminal_width() -> int:
    """Get the current terminal width.

    Returns:
        Terminal width in columns, defaults to 80 if unable to determine.
    """
    try:
        return shutil.get_terminal_size().columns
    except (AttributeError, ValueError):
        return 80


@dataclass
class ColumnSpec:
    """Specification for a proportional table column.

    Attributes:
        name: Column identifier.
        min_width: Minimum column width in characters.
        max_width: Maximum column width (0 = unlimited).
        ratio: Proportional share of remaining space.
        fixed: If True, width is exactly min_width (ignores ratio).
    """

    name: str
    min_width: int
    max_width: int = 0
    ratio: float = 1.0
    fixed: bool = False


def calculate_column_widths(
    columns: list[ColumnSpec], total_width: int, padding: int = 2
) -> dict[str, int]:
    """Calculate column widths proportionally within total_width.

    Fixed columns get exactly their min_width. Remaining space is
    distributed among non-fixed columns according to their ratio,
    respecting min/max constraints.

    Args:
        columns: List of column specifications.
        total_width: Total available width (e.g., terminal width).
        padding: Padding between columns (applied between each pair).

    Returns:
        Dict mapping column name to calculated width.
    """
    total_padding = padding * max(0, len(columns) - 1)
    available = total_width - total_padding

    # First pass: allocate fixed columns
    result = {}
    remaining = available
    flex_columns = []

    for col in columns:
        if col.fixed:
            result[col.name] = col.min_width
            remaining -= col.min_width
        else:
            flex_columns.append(col)

    if not flex_columns or remaining <= 0:
        # No flex columns or no remaining space: give minimums
        for col in flex_columns:
            result[col.name] = col.min_width
        return result

    # Second pass: distribute remaining space by ratio
    total_ratio = sum(c.ratio for c in flex_columns)
    if total_ratio == 0:
        total_ratio = len(flex_columns)

    for col in flex_columns:
        share = int(remaining * (col.ratio / total_ratio))
        width = max(col.min_width, share)
        if col.max_width > 0:
            width = min(width, col.max_width)
        result[col.name] = width

    return result
