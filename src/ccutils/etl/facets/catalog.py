"""Tier 2 (LLM-extracted) facet catalog.

The single source of truth for which Tier 2 facets exist, what prompts
they use, and what schema the model is asked to produce. Three places
read this list:

1. `create_star_schema()` -- seeds one `dim_facet_type` row per spec,
   using `INSERT ... ON CONFLICT DO NOTHING` so historical prompt
   versions survive re-runs.
2. `populate_tier2_facets()` -- builds the LLM prompt from these specs
   and writes one `fact_session_facets` row per (session × spec).
3. The extractor (`AnthropicFacetExtractor` / `CannedFacetExtractor`)
   -- receives the list as the `enabled_facets` argument and uses it
   to build the JSON response schema.

**Adding a new Tier 2 facet:** append a `FacetSpec` row. No protocol
changes needed; the system prompt regenerates from the list, the
seed adds the new dim_facet_type row, the populator emits a column
per row.

**Bumping a prompt:** add a NEW `FacetSpec` row with the same
`facet_id` but a bumped `prompt_version` (e.g. v1 -> v2). The old row
survives (the seed uses ON CONFLICT DO NOTHING). Existing
`fact_session_facets` rows referencing v1 stay queryable; new
extractions emit v2 rows. `TestFacetTypeKeyVersionProperty` pins this
contract.

**F20 description workshopping:** the description text is the
load-bearing input to cluster quality -- it's what the model sees
in the system prompt's facet schema. Bigger investment in this text
than in any other field on the spec.
"""

from __future__ import annotations

from ccutils.etl.facets.extractor import FacetSpec


F20_TASK_DESCRIPTION = FacetSpec(
    facet_id="F20",
    facet_name="task_description",
    output_type="text",
    prompt_version="v1",
    description=(
        "A one- or two-sentence summary of what this session was about, "
        "written the way a skilled engineer would phrase a PR title: "
        "'Debugging a failing test in a Python data pipeline', "
        "'Refactoring an authentication middleware', 'Investigating a "
        "production performance regression'. Describe the kind of work "
        "in general terms. Avoid project-specific nouns -- specific "
        "names, file paths, repository names, framework versions, "
        "table names. If a detail can't be generalized, omit it."
    ),
)


FACET_SPECS: list[FacetSpec] = [
    F20_TASK_DESCRIPTION,
]


def facet_tier_scope_sql(tier: int) -> str:
    """SQL fragment scoping a `fact_session_facets` soft-delete to one
    tier's `facet_type_key`s. Used by both Tier 1 and Tier 2 populators
    so neither soft-deletes the other's rows when they share the
    `session_id` scope. See `upsert.lineage_upsert(soft_delete_scope_sql=)`.

    Centralized so a third populator (or a tier renumbering) doesn't have
    to know the exact SQL shape -- wrong scope here silently wipes the
    cross-tier rows it shouldn't touch.
    """
    return (
        f"tgt.facet_type_key IN "
        f"(SELECT facet_type_key FROM dim_facet_type WHERE tier = {tier})"
    )
