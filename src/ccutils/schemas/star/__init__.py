"""Star Schema package for Claude Code Transcripts.

This package provides:
- DuckDB star schema creation for transcript analytics
- Semantic model generation for data exploration
- ETL pipeline for loading session data
- Heuristic classification for session categorization
- JSON export for star schema data
"""

from .embeddings import EmbeddingPipeline
from .etl import run_star_schema_etl
from .history_etl import load_history
from .heuristics import (
    classify_complexity,
    classify_domain,
    classify_error_type,
    classify_intent,
    classify_outcome,
)
from .json_export import export_star_schema_to_json
from .schema import create_star_schema
from .semantic import create_semantic_model
from .utils import (
    TOOL_CATEGORIES,
    generate_dimension_key,
    get_model_family,
    get_time_of_day,
    get_tool_category,
)

__all__ = [
    # Schema creation
    "create_star_schema",
    # Semantic model
    "create_semantic_model",
    # ETL
    "run_star_schema_etl",
    # JSON export
    "export_star_schema_to_json",
    # Heuristic classification
    "classify_intent",
    "classify_complexity",
    "classify_outcome",
    "classify_domain",
    "classify_error_type",
    # Embedding pipeline
    "EmbeddingPipeline",
    # Utilities
    "generate_dimension_key",
    "get_tool_category",
    "get_model_family",
    "get_time_of_day",
    "TOOL_CATEGORIES",
]
