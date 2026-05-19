"""FacetExtractor protocol + supporting dataclasses + Canned fake-backend.

This is the boundary between "the populator that needs structured facets
extracted from a session" and "an LLM (or any other backend) that knows
how to do that." See internal/plans/facet_extractor_protocol.md for the
full design.

The populator never imports a concrete extractor -- it accepts a
FacetExtractor by parameter (dependency injection from the orchestrator).
Tests pass CannedFacetExtractor; production passes AnthropicFacetExtractor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


OutputType = Literal["text", "enum", "json", "int", "float", "bool"]

# Maps a FacetSpec.output_type to the `fact_session_facets.value_*`
# column that stores it. Made explicit so the contract between extractor
# output and DDL storage is one lookup, not a 6-arm if/else at each
# call site.
OUTPUT_TYPE_TO_COL: dict[OutputType, str] = {
    "text": "value_text",
    "enum": "value_text",
    "json": "value_json",
    "int": "value_numeric",
    "float": "value_numeric",
    "bool": "value_bool",
}


@dataclass(frozen=True)
class FacetSpec:
    """One entry in the Tier 2 catalog. Identifies the facet, pins its
    output type, and carries the prompt version that produced it.

    `enum_values` is set for `output_type='enum'`; the extractor uses it
    both in the prompt (so the model sees the allowed list) and in
    output validation (to reject hallucinated values). For non-enum
    facets, `enum_values` stays None.
    """

    facet_id: str
    facet_name: str
    output_type: OutputType
    prompt_version: str
    description: str
    enum_values: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SessionInputs:
    """Privacy-sanitized inputs the extractor sees. The populator is the
    boundary that prepares these -- everything downstream must work
    without raw transcript content beyond what lands here.
    """

    session_id: str
    first_user_message: str
    last_assistant_message: str
    tool_mix_summary: str
    model_used: str | None
    duration_seconds: int | None


@dataclass(frozen=True)
class FacetOutput:
    """One extracted facet. `value` is `str | None`; the populator
    serializes / casts it to the right `value_*` column based on
    `output_type` from the matching FacetSpec.

    `metadata_json` is a JSON-encoded blob with the raw model response,
    retry_count, cache_hit, latency_ms. Populator stores it as
    `fact_session_facets.extraction_metadata_json` for Tier 2 rows; Tier 1
    rows leave it NULL.
    """

    facet_id: str
    prompt_version: str
    value: str | None
    is_fallback: bool = False
    metadata_json: str = "{}"


@runtime_checkable
class FacetExtractor(Protocol):
    """The boundary contract.

    Implementations MAY batch multiple facets into a single underlying
    backend call (the Anthropic implementation does -- one JSON object
    per session). The return shape is per-facet either way.

    Every spec MUST get a FacetOutput in the return dict, even on
    failure. A failed extraction emits `value=None, is_fallback=True`
    rather than going missing.
    """

    def extract(
        self,
        session_inputs: SessionInputs,
        enabled_facets: list[FacetSpec],
    ) -> dict[str, FacetOutput]:
        ...


class CannedFacetExtractor:
    """Fake-backend extractor for tests. Returns canned values keyed by
    `(session_id, facet_id)`. When no canned value is set for a spec,
    emits `is_fallback=True, value=None` for that facet (matching the
    real extractor's contract for genuine extraction failure).
    """

    def __init__(self, canned: dict[tuple[str, str], str]) -> None:
        self._canned = canned

    def extract(
        self,
        session_inputs: SessionInputs,
        enabled_facets: list[FacetSpec],
    ) -> dict[str, FacetOutput]:
        out: dict[str, FacetOutput] = {}
        for spec in enabled_facets:
            value = self._canned.get((session_inputs.session_id, spec.facet_id))
            out[spec.facet_id] = FacetOutput(
                facet_id=spec.facet_id,
                prompt_version=spec.prompt_version,
                value=value,
                is_fallback=(value is None),
            )
        return out
