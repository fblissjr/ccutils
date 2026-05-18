"""Star schema package: DDL + utilities.

Per-session ETL lives in `ccutils.etl.orchestrator.run_v15_etl`; the
optional ColBERT embedding pipeline lives in `.embeddings`. Heuristic
classifiers, the legacy per-session ETL, and the history.jsonl loader
were removed when v0.15 landed.
"""

from .embeddings import EmbeddingPipeline
from .json_export import export_star_schema_to_json
from .schema import create_star_schema
from .utils import (
    TOOL_CATEGORIES,
    generate_dimension_key,
    get_model_family,
    get_time_of_day,
    get_tool_category,
)

__all__ = [
    "create_star_schema",
    "export_star_schema_to_json",
    "EmbeddingPipeline",
    "generate_dimension_key",
    "get_tool_category",
    "get_model_family",
    "get_time_of_day",
    "TOOL_CATEGORIES",
]
