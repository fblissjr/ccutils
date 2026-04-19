"""Schema definitions for Claude Code transcripts.

This package provides two schemas:
- simple: 4 tables (sessions, messages, tool_calls, thinking)
- star: 28 tables + 14 views for dimensional analytics

Use resolve_schema_format() to handle CLI schema/format combinations.
"""

from .simple import (
    create_duckdb_schema,
    export_session_to_duckdb,
    export_sessions_to_json,
    _extract_session_data,
)

from .star import (
    create_star_schema,
    run_star_schema_etl,
    export_star_schema_to_json,
    generate_dimension_key,
    get_tool_category,
    get_model_family,
    get_time_of_day,
    TOOL_CATEGORIES,
)


def resolve_schema_format(output_format):
    """Infer schema type from compound format names.

    'duckdb-star' and 'json-star' -> star schema.
    'duckdb', 'json', 'html' -> simple schema.

    Returns:
        Tuple of (schema, base_format) e.g. ("star", "duckdb") or ("simple", "json")
    """
    if output_format.endswith("-star"):
        return "star", output_format.replace("-star", "")
    return "simple", output_format


__all__ = [
    # Simple schema
    "create_duckdb_schema",
    "export_session_to_duckdb",
    "export_sessions_to_json",
    "_extract_session_data",
    # Star schema
    "create_star_schema",
    "run_star_schema_etl",
    "export_star_schema_to_json",
    "generate_dimension_key",
    "get_tool_category",
    "get_model_family",
    "get_time_of_day",
    "TOOL_CATEGORIES",
    # Utilities
    "resolve_schema_format",
]
