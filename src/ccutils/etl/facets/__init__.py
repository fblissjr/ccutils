"""Facet & cluster pipeline -- LLM-extracted (Tier 2) facets.

See docs/FACET_CLUSTER_PIPELINE.md and
internal/plans/facet_extractor_protocol.md for the design.

Tier 1 facets (SQL-computed F01..F19) live in
ccutils.etl.fact_session_facets; this subpackage is dedicated to Tier 2
(LLM-extracted) and the protocol boundary that makes the backend
swappable.
"""

from ccutils.etl.facets.anthropic import (
    AnthropicFacetExtractor,
    FacetExtractionError,
)
from ccutils.etl.facets.catalog import FACET_SPECS
from ccutils.etl.facets.extractor import (
    CannedFacetExtractor,
    FacetExtractor,
    FacetOutput,
    FacetSpec,
    SessionInputs,
)
from ccutils.etl.facets.populator import populate_tier2_facets

__all__ = [
    "AnthropicFacetExtractor",
    "CannedFacetExtractor",
    "FACET_SPECS",
    "FacetExtractionError",
    "FacetExtractor",
    "FacetOutput",
    "FacetSpec",
    "SessionInputs",
    "populate_tier2_facets",
]
