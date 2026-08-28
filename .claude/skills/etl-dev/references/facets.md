# Facet development (F01+)

Facets are one structured attribute per session, stored EAV-style in
`fact_session_facets` (exactly one of `value_text` / `value_json` /
`value_numeric` / `value_bool` per row, routed by `dim_facet_type.output_type`
via `OUTPUT_TYPE_TO_COL` in `etl/facets/extractor.py`). Design doc:
`docs/FACET_CLUSTER_PIPELINE.md` (§3 catalog, §4 schema, §5 transforms).

## Where each tier lives

| Tier | IDs | Produced by | Code |
|---|---|---|---|
| 1 | F01–F19 | SQL over existing facts, always on | `etl/fact_session_facets.py::populate_tier1_facets` |
| 2 | F20–F30 | One Haiku call per session, opt-in (`--llm-facets`) | `etl/facets/` (catalog, extractor, populator) |
| 3 | F40+ | Corpus-wide clustering (not yet built) | — |

## Hard rules

- **`etl/facets/catalog.py::FACET_SPECS` is the single source of truth for
  Tier 2.** `schemas/star` imports FROM the catalog at DDL seed time — never
  the reverse (import cycle at CLI startup).
- `dim_facet_type` seeds via `CREATE TABLE IF NOT EXISTS` +
  `INSERT ... ON CONFLICT DO NOTHING` so historical prompt_version rows
  survive re-runs.
- Both Tier 1 and Tier 2 populators write `fact_session_facets` — each MUST
  pass `soft_delete_scope_sql` to `lineage_upsert` (scoped by facet tier) or
  one soft-deletes the other's rows.
- Tier 2 credentials resolve ONLY via
  `cli/utils.py::build_facet_extractor_or_exit` (env `ANTHROPIC_API_KEY` or
  the `ccutils-anthropic` keychain entry) — no new credential paths.

## Adding a Tier 2 facet

Append a `FacetSpec` to `FACET_SPECS` (facet_id, facet_name, output_type,
prompt_version, description). No protocol changes: the system prompt
regenerates from the list, the seed adds the `dim_facet_type` row, the
populator emits the new facet. The `description` text is the load-bearing
input to extraction quality — invest there.

## Bumping a prompt

Add a NEW `FacetSpec` row with the same `facet_id` and a bumped
`prompt_version` (v1 → v2). The old dim row and old fact rows survive and stay
queryable; new extractions emit v2 rows. `TestFacetTypeKeyVersionProperty`
pins this contract — never mutate an existing spec's prompt in place.

## Adding a Tier 1 facet

Extend `populate_tier1_facets` (`etl/fact_session_facets.py`) with another
`_insert_facet` / SQL block, and register the facet in the Tier 1 registry it
seeds from. Ordering note: F01–F04 read the heuristic columns on
`dim_session`, so Tier 1 facets run after heuristic enrichment.

## Testing

- Fixture ids for hypothetical facet types use **F90+**
  (`tests/test_fact_session_facets_v15.py::two_f90_versions`).
- Tier 2 unit tests use `CannedFacetExtractor` — no API calls.
- After any `AnthropicFacetExtractor` change, run the live smoke (pennies):
  `ANTHROPIC_API_KEY=$(security find-generic-password -s ccutils-anthropic -a $USER -w) uv run pytest tests/test_populate_tier2_facets.py::TestLiveApiSmoke -v`
- QA columns on `fact_session_facets`: `is_fallback` (extractor couldn't
  produce a real value) and `extraction_metadata_json` (raw response +
  retry/cache bookkeeping) — assert on these, not just `value_*`.

## Cost note

Tier 2 is one Haiku call per session — pennies on a handful, dollars across
hundreds. Never make it default-on, and keep `--format json` +
`--llm-facets` semantics (extraction into a discarded temp DB) loudly
documented in help text.
