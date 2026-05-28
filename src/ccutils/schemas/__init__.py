"""Schema definitions for Claude Code transcripts.

Single schema: v0.15 dimensional model. Per-session ETL lives in
``ccutils.etl.orchestrator.run_v15_etl``. The legacy "simple" 4-table
schema was removed when v0.15 stabilized -- the CLI's `--format duckdb`
and `--format json` now write the star schema unconditionally.
"""

from .star import (
    create_star_schema,
    export_star_schema_to_json,
    generate_dimension_key,
    get_tool_category,
    get_model_family,
    get_time_of_day,
    TOOL_CATEGORIES,
)

__all__ = [
    "create_star_schema",
    "export_star_schema_to_json",
    "generate_dimension_key",
    "get_tool_category",
    "get_model_family",
    "get_time_of_day",
    "TOOL_CATEGORIES",
]
