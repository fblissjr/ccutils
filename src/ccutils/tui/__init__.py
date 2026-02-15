"""TUI package for styled terminal UI components.

Provides semantic color themes, formatted choice builders for questionary,
Rich table components, and responsive layout utilities.
"""

from .theme import (
    STYLES,
    RICH_STYLES,
    MODEL_FAMILIES,
    model_style_key,
    questionary_style,
)

from .formatters import (
    format_relative_date,
    format_relative_date_short,
    format_duration,
    format_project_name,
    format_summary,
    format_branch,
    format_size,
    format_msg_count,
)

from .layout import (
    get_terminal_width,
    ColumnSpec,
    calculate_column_widths,
)

from .components import (
    render_project_table,
    render_session_table,
    render_status_header,
)

from .selection import (
    build_project_choices,
    build_session_choices,
    build_flat_choices,
    build_web_session_choices,
    build_import_choices,
)

__all__ = [
    # Theme
    "STYLES",
    "RICH_STYLES",
    "MODEL_FAMILIES",
    "model_style_key",
    "questionary_style",
    # Formatters
    "format_relative_date",
    "format_relative_date_short",
    "format_duration",
    "format_project_name",
    "format_summary",
    "format_branch",
    "format_size",
    "format_msg_count",
    # Layout
    "get_terminal_width",
    "ColumnSpec",
    "calculate_column_widths",
    # Components
    "render_project_table",
    "render_session_table",
    "render_status_header",
    # Selection
    "build_project_choices",
    "build_session_choices",
    "build_flat_choices",
    "build_web_session_choices",
    "build_import_choices",
]
