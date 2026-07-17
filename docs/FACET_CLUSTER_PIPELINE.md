<!-- path-privacy: skip-file -- references universal Claude Code data paths (not personal) -->
# Facet & Cluster Pipeline — Data Architecture

*Companion to `STAR_SCHEMA.md`. Defines the data, transforms, and pipeline layers that turn the v0.15 transcript archive into a queryable map of usage patterns. Use cases are derived from the data we capture, not the other way around.*

**Status:** Steps 1-4.5 landed. DDL + Tier 1 registry, Tier 1 SQL populator (F01-F19, `src/ccutils/etl/fact_session_facets.py`), Tier 2 extractor protocol (`AnthropicFacetExtractor` + `CannedFacetExtractor`, `src/ccutils/etl/facets/`), F20 `task_description` populator end-to-end, CLI flags `--with-llm-facets` (local) / `--batch-llm-facets` (all). `FACET_SPECS` in `etl/facets/catalog.py` still holds only F20 -- F21+ are cataloged below as design, not yet implemented. Step 5 (embedding + clustering) not yet started; the `fact_facet_embeddings` table exists as a DDL stub (unpopulated), and `fact_clustering_run` / `dim_cluster` / `bridge_cluster_session` / `fact_cluster_metrics` described in §4 don't exist in the DDL yet. Awaiting first F20 sample run on a real corpus to inform the embedding-model choice.
**Last updated:** 2026-07-17

---

## 1. Scope and framing

The v0.15 star schema captures *what happened* in every Claude Code session — every tool call, every message, every token. What it doesn't capture is *what the session was about* in any form that can be aggregated, searched, or compared across sessions. The regex-based heuristics in `etl/heuristics.py` give us a coarse first cut (`intent`, `complexity`, `outcome`, `domain`) but stop short of anything that supports cross-session pattern discovery.

This document defines the data layer that closes that gap. It does so without changing ccutils's source-data contract: **transcripts only, no API pulls.** All new data is derived from what's already in `~/.claude` plus what's already in the star schema.

Design principles, in priority order:

1. **Capture data once, derive many use cases.** Every new column or table earns its place by enabling at least three downstream analyses.
2. **Static facets before LLM facets.** Anything that can be computed from existing facts via SQL is computed that way. LLM extraction is reserved for what genuinely needs it.
3. **Extensibility through registries.** Adding a facet is a row in a registry table plus one populator. Not a schema migration.
4. **Privacy by default.** Facets are designed to be safe to publish at cluster-level aggregates. Private content stays in source transcripts.

---

## 2. What data already exists

Inventory of the v0.15 facts and dimensions, with the fields relevant to facet extraction. This is the substrate the new layer reads from.

| Existing table | Fields relevant to facets |
|---|---|
| `fact_messages` | First user message text, last assistant message text, message counts, stop_reason, prompt_id |
| `fact_tool_uses` | tool_name, timestamps, input_json |
| `fact_tool_results` | tool result content, is_error (tri-state), num_lines/total_lines (Read), exit_code/interrupted (Bash), structured_patch (Edit), Agent rollup |
| `fact_token_usage` | per-API-response token breakdown, cache hits, model name |
| `fact_session_summary` | aggregate tokens/cost/duration/tool counts per session |
| `fact_file_operations` | file paths, file extensions, operation counts (no LOC-delta column yet -- see F08 note below) |
| `fact_errors` | error messages, error types |
| `fact_attachments` | attachment subtypes (23 variants) |
| `fact_pr_links` | PR URLs referenced |
| `fact_plan_revisions` | plan content over time |
| `fact_agent_delegations` | subagent fan-out, agent types |
| `dim_session` | session id, cwd, git_branch, agent_id, parent_session_key, first/last timestamp, intent/complexity/outcome/domain (heuristic columns, `etl/heuristics.py`) |
| `dim_project` | project name, project path |
| `dim_model` | model name, family |
| `dim_tool` | tool name, tool_category |
| `dim_prompt` | prompts from Claude Code's prompt-history JSONL (cross-session) |
| `dim_file` | file path normalized, extension |
| `dim_date` / `dim_time` | calendar attributes (dow, hour, etc.) |

This is enough to derive every Tier 1 facet listed below without any LLM call.

---

## 3. Facet catalog (what new data we need to capture)

A facet is one structured attribute of a session, designed to be queryable, embeddable, and cluster-able. Facets are versioned — re-running with a new prompt or method produces a new facet row, not a destructive overwrite.

Facets are split into three tiers by *how* they're produced. The split matters because the cost, latency, and refresh cadence are different per tier.

### Tier 1 — Computed from existing facts (SQL only, no inference)

These are cheap, deterministic, and run inline with the existing `run_v15_etl()` orchestrator.

| ID | Facet | Type | Reads from | Notes / why we want it |
|---|---|---|---|---|
| F01 | `session_intent` | enum | `fact_messages` first user msg | Already exists; keep regex version as fallback |
| F02 | `session_complexity` | enum | `fact_session_summary` | Already exists |
| F03 | `session_outcome` | enum | `fact_messages` last asst msg + `fact_errors` rate | Already exists |
| F04 | `session_domain` | enum | `fact_file_operations` extensions | Already exists |
| F05 | `error_signature` | text[] | `fact_errors.error_type` | Ordered list — error progression within session |
| F06 | `tool_mix` | json | `fact_tool_uses` histogram | Tool name → count; basis for "session shape" |
| F07 | `tool_bigram_top3` | text[] | `fact_tool_chain_steps` | E.g. `["Read→Edit", "Edit→Bash", …]` — workflow signature |
| F08 | `loc_delta` | int | `fact_file_operations` | Added minus removed. **Implemented as a proxy**: current populator emits a count of write/edit operations, not a true added-minus-removed delta -- that needs unpacking `fact_tool_results.edit_structured_patch_json`, tracked as a follow-up |
| F09 | `file_extensions_touched` | text[] | `fact_file_operations` | Distinct extensions; finer than `session_domain` |
| F10 | `repo_slug` | text | `dim_project` | Stable across sessions in same repo |
| F11 | `model_mix` | json | `fact_token_usage` × `dim_model` | Tokens per model; catches model-switch sessions |
| F12 | `duration_seconds` | int | `dim_session` first/last | — |
| F13 | `agent_depth` | int | `dim_session` parent_session_key chain | 0 = primary; >0 = subagent |
| F14 | `human_message_count` | int | `fact_messages` | — |
| F15 | `tokens_in` / `tokens_out` / `cost_usd` | num | `fact_session_summary` | Already aggregated. **Implemented as a single value**: current populator emits only summed `input_tokens` from `fact_token_usage` (deliberately independent of `fact_session_summary`'s populator order); `tokens_out` and `cost_usd` are not computed anywhere in the codebase yet -- there is no USD pricing calculation at all |
| F16 | `local_hour` / `local_dow` | enum | `dim_time` | For temporal patterns |
| F17 | `had_subagents` | bool | `fact_agent_delegations` count | — |
| F18 | `pr_referenced` | bool | `fact_pr_links` | Was a PR opened/referenced in-session |
| F19 | `had_plan_revision` | bool | `fact_plan_revisions` | Did the session reshape its plan mid-flight |

### Tier 2 — LLM-extracted from session content (one inference call per facet × session)

These are the Clio-style facets. Each one is a short prompt to a small model (Haiku-class). One row per (session × facet), text-valued, embeddable. Cost target: ≤ $0.001 per session for the full Tier 2 set.

| ID | Facet | Output | Prompt input | Why we want it |
|---|---|---|---|---|
| F20 | `task_description` | short text (1-2 sentences) | first user msg + final assistant msg + tool-mix summary | The primary embeddable facet; basis for most clustering |
| F21 | `accomplishment` | short text | full message arc summary | What actually got done (may differ from task) |
| F22 | `blocker_type` | enum: `none / knowledge / environment / tool-limit / unclear-req / external-dep` | error log + final message | What stopped progress when outcome wasn't `success` |
| F23 | `resolution_pattern` | enum: `completed / pivoted / abandoned / handed-off / partial` | end-of-session signals | Finer than F03 outcome |
| F24 | `collaboration_style` | enum: `directive / exploratory / iterative / dialog / hand-off` | turn structure + message lengths | How the human worked with the model |
| F25 | `artifact_produced` | enum: `code / config / docs / decision-record / plan / analysis / none` | tool-mix + final assistant msg | What the session left behind |
| F26 | `codebase_subject` | short text | tool-paths + first msg | Finer than `repo_slug`; what part of the codebase |
| F27 | `knowledge_domain` | short text | first user msg + repo context | Finer than F04 domain (e.g. "rate limiting in HTTP middleware") |
| F28 | `prompt_register` | enum: `question / directive / debugging / planning / review` | first user msg | How the user opened the session |
| F29 | `tone_signal` | enum: `neutral / curious / frustrated / time-pressured / instructive` | message lengths + lexical signals | Affective context; feeds wellbeing analyses |
| F30 | `continuation_relationship` | enum: `fresh / continuation / split / parallel / unrelated` | `dim_session` chain + prior session task_description | Whether this is part of a larger arc |

### Tier 3 — Derived from clustering passes (corpus-wide)

These exist only after the clustering pipeline runs. They're the *output* of the cluster step, written back per-session for queryability.

| ID | Facet | Output | Method |
|---|---|---|---|
| F40 | `cluster_id_base` | text | k-means cluster id on F20 embedding |
| F41 | `cluster_id_l2` / `l3` / `l4` | text | hierarchical parent cluster ids |
| F42 | `cluster_neighbors` | text[] | top-N nearest sessions in F20 embedding space |
| F43 | `cluster_distance_to_centroid` | float | quality / outlier indicator |
| F44 | `task_taxonomy_label` | text | post-hoc mapping of cluster_id → human-named family |

---

## 4. Schema additions

The facet layer fits the existing star schema cleanly. Six new tables: one registry, one structured-fact table, one embedding fact, one clustering-run provenance, one cluster dim, one cluster-session bridge, plus a per-cluster metrics fact. Only the first two (`dim_facet_type`, `fact_session_facets`) plus the unpopulated `fact_facet_embeddings` stub exist in the DDL today; the pseudo-schemas below are the design as originally specified -- see `STAR_SCHEMA.md` for the exact, current column list on the tables that ship (implementation added a couple of Tier 2 QA columns not shown here, e.g. `is_fallback` / `extraction_metadata_json` on `fact_session_facets`).

**Schema-split note (locked in during build step 1):** the original design proposed an inline `embedding BLOB` column on `fact_session_facets`. That was changed before implementation. Embeddings live in their own table (`fact_facet_embeddings`) for three reasons:

1. **Native DuckDB ops on FLOAT[N].** `FLOAT[384]` enables `array_cosine_similarity` / `array_inner_product` directly in SQL. BLOB would force every consumer to bring its own deserializer.
2. **Lean structured-facet scans.** Putting kilobyte vectors next to enum/text values bloats the buffer pool on every plain SQL filter against the facet table.
3. **Multi-model evolution.** When a second embedding model is introduced (or a version bump), the embeddings table gets new rows keyed by `(embedding_model, embedding_model_version)` while the structured-value table stays untouched.

### `dim_facet_type` — registry of facet definitions

One row per facet, versioned. New facets = new rows. Seeded with F01–F19 (Tier 1, `method='computed'`) by `create_star_schema()`.

```
facet_type_key       VARCHAR PK   -- md5(facet_id || '|' || COALESCE(prompt_version, ''))
facet_id             VARCHAR      -- F01, F20, etc.
facet_name           VARCHAR      -- "task_description"
tier                 INTEGER      -- 1, 2, 3
method               VARCHAR      -- "computed" | "regex" | "llm" | "cluster"
output_type          VARCHAR      -- "enum" | "text" | "json" | "vector" | "int" | "float" | "bool"
prompt_text          VARCHAR NULL -- only for LLM facets
prompt_version       VARCHAR NULL -- bumped when prompt changes
embedding_model      VARCHAR NULL -- default embedder when this facet is embeddable
created_at           TIMESTAMP
```

### `fact_session_facets` — one row per session × facet × prompt_version

The structured-value facet table. Typed columns for filtering and clustering inputs; no embedding column (see split note above). Carries the standard v0.15 lineage envelope.

```
session_key              VARCHAR
session_id               VARCHAR
facet_type_key           VARCHAR
prompt_version           VARCHAR NULL  -- inherited from facet_type at extraction time
value_text               VARCHAR NULL  -- text / enum facets
value_json               JSON NULL     -- json facets (lists, histograms)
value_numeric            DOUBLE NULL
value_bool               BOOLEAN NULL
date_key                 INTEGER
time_key                 INTEGER
extracted_at             TIMESTAMP
+ lineage envelope (created_at, last_updated_at, version keys,
  etl_run_id, record_source, hash_diff, is_deleted, deleted_at)
Natural key: (session_key, facet_type_key, prompt_version)
```

### `fact_facet_embeddings` — one row per session × facet × model × model_version

Where the vectors live. `FLOAT[384]` locks in BGE-small-en-v1.5 as the default embedder and enables DuckDB-native cosine ops. `embedding_dim` is intentionally omitted — `(embedding_model, embedding_model_version)` uniquely determines the dim, so storing it would be redundant.

```
session_key              VARCHAR
session_id               VARCHAR
facet_type_key           VARCHAR
embedding_model          VARCHAR        -- e.g. 'bge-small-en-v1.5'
embedding_model_version  VARCHAR        -- e.g. 'v1.5'
embedding                FLOAT[384]
date_key                 INTEGER
time_key                 INTEGER
embedded_at              TIMESTAMP
+ lineage envelope
Natural key: (session_key, facet_type_key, embedding_model, embedding_model_version)
```

### `fact_clustering_run` — provenance

Every clustering pass logs itself.

```
clustering_run_id        TEXT PK
facet_basis_key          TEXT FK    -- which facet's embeddings were clustered
embedding_model          TEXT
algorithm                TEXT       -- "kmeans" | "hdbscan"
k_value                  INT NULL   -- if applicable
n_sessions               INT
n_clusters_base          INT
hierarchy_depth          INT
started_at               TIMESTAMP
completed_at             TIMESTAMP
```

### `dim_cluster` — one row per cluster across all runs and levels

```
cluster_key              TEXT PK
clustering_run_id        TEXT FK
level                    INT        -- 0 = base, 1+ = hierarchy levels
parent_cluster_key       TEXT NULL FK
title                    TEXT       -- LLM-generated
summary                  TEXT       -- LLM-generated
member_count             INT
privacy_audit_status     TEXT       -- "passed" | "flagged" | "skipped"
created_at               TIMESTAMP
```

### `bridge_cluster_session` — many-to-many

A session can belong to clusters at multiple hierarchy levels and across multiple clustering runs.

```
cluster_key              TEXT FK
session_key              TEXT FK
distance_to_centroid     DOUBLE
is_nearest_to_centroid   BOOLEAN    -- "exemplar" flag for cluster description
PRIMARY KEY (cluster_key, session_key)
```

### `fact_cluster_metrics` — per-cluster aggregates over existing facts

The "so what" join point. Lets you answer "what's the success rate of cluster X" without recomputing.

```
cluster_key              TEXT FK
n_sessions               INT
success_rate             DOUBLE
avg_duration_seconds     DOUBLE
avg_cost_usd             DOUBLE
top_tools_json           JSON
top_models_json          JSON
top_repos_json           JSON
top_extensions_json      JSON
date_range_start         DATE
date_range_end           DATE
computed_at              TIMESTAMP
```

---

## 5. ETL transforms

Each transform is a single populator function, idempotent on natural keys, following the existing `lineage_upsert` convention.

| ID | Transform | Input | Output | Method | Cost |
|---|---|---|---|---|---|
| T01 | Compute Tier 1 facets | existing facts | `fact_session_facets` (Tier 1 rows) | SQL aggregations | free |
| T02 | Extract Tier 2 facets per session | `fact_session_summary` + first/last messages from `fact_messages` | `fact_session_facets` (Tier 2 rows, text values) | Haiku prompt per facet × session, batched | ~$0.0005-0.001 per session per facet |
| T03 | Embed text facets | `fact_session_facets` where `embedding IS NULL` and facet is embeddable | `fact_session_facets.embedding` | sentence-transformers (local) or ColBERT (existing optional dep) | local CPU/GPU |
| T04 | Cluster (base level) | embeddings from T03 for chosen facet (typically F20) | `dim_cluster` (level=0) + `bridge_cluster_session` | k-means or HDBSCAN | local CPU |
| T05 | Describe each base cluster | `dim_cluster` + N exemplar sessions per cluster from `bridge_cluster_session` | `dim_cluster.title` + `summary` | Haiku prompt with privacy guardrails | ~$0.01 per cluster |
| T06 | Hierarchize | cluster titles+summaries from T05 | `dim_cluster` (level=1+) with `parent_cluster_key` | recursive k-means on description embeddings + label | ~$0.01 per parent cluster |
| T07 | Cluster metrics rollup | `bridge_cluster_session` joined to existing facts | `fact_cluster_metrics` | SQL aggregations | free |
| T08 | Privacy audit (optional) | `dim_cluster.title` + `summary` | `dim_cluster.privacy_audit_status` | LLM pass; flags any cluster mentioning PII or single-source identifiers | ~$0.005 per cluster |
| T09 | Write back cluster IDs as Tier 3 facets | `bridge_cluster_session` | `fact_session_facets` (Tier 3 rows) | SQL | free |

### Privacy guardrails embedded in the transforms

Even in single-user mode, the design assumes the output might one day be shared (with a teammate, in a write-up, on a blog). So:

- **T02 prompts** explicitly instruct: "Omit specific names, file paths, repo names, secrets, or personal information. Describe what was being done in general terms."
- **T04 minimum cluster size** = 5 sessions. Smaller groupings drop into an "other" bucket and aren't surfaced.
- **T05 prompts** repeat the privacy instructions and are given only the exemplar facet text (already privacy-sanitized at T02), not raw transcript content.
- **T08 audit** is a separate LLM pass that scans cluster titles/summaries for leaked specifics. Fail-closed: a cluster that flags can't be exported.

---

## 6. Pipeline structure and scheduling

The new pipeline splits into two cadences: **per-session** (extends `run_v15_etl`) and **corpus-wide** (new orchestrator that runs nightly or on-demand).

### Per-session additions to `run_v15_etl(conn, session_path, ...)`

```
[existing v0.15 pipeline through populate_fact_session_summary]
  ↓
populate_tier1_facets        (T01)   — runs always, free
  ↓
populate_tier2_facets        (T02)   — gated by flag (default off); LLM cost
  ↓
populate_facet_embeddings    (T03)   — only embeddable facets
```

Gating T02 behind a flag matters: most users will want to backfill in batch, not pay LLM cost per session at parse time. The orchestrator should accept `--with-llm-facets` (single-session) and `--batch-llm-facets` (corpus pass).

### Corpus-wide pipeline (new — `run_facet_clustering(conn, ...)`)

```
backfill_tier2_facets        (T02 over all sessions missing it)
  ↓
backfill_embeddings          (T03)
  ↓
cluster_base                 (T04)   — produces dim_cluster level=0 + bridge_cluster_session
  ↓
describe_clusters            (T05)   — fills dim_cluster.title + summary
  ↓
hierarchize                  (T06)   — recursive, until cluster count < threshold
  ↓
audit_privacy                (T08)   — optional, only if export planned
  ↓
write_cluster_ids_back       (T09)   — Tier 3 facets onto fact_session_facets
  ↓
compute_cluster_metrics      (T07)   — fact_cluster_metrics
```

### Scheduling

| Pipeline | Cadence | Trigger |
|---|---|---|
| Per-session Tier 1 facets | Inline with every `run_v15_etl` call | New session parsed |
| Per-session Tier 2 facets | Inline if `--with-llm-facets`, else deferred | Flag |
| Corpus-wide clustering | Nightly or on-demand | Cron / CLI |
| Re-clustering after schema/prompt change | On-demand | CLI; bumped `prompt_version` triggers re-extract |

---

## 7. Use cases — derived from the data we'd have

For each use case, what facets it requires and what queries it implies. Use cases that need facets not in the catalog above are flagged as "out of scope" so we don't pretend to enable them.

| # | Use case | Required facets | Required clustering? | Query shape |
|---|---|---|---|---|
| 1 | "What am I using Claude for, broken down by topic?" | F20 task_description | yes (base + hierarchy) | top-N clusters by member_count, with title + summary |
| 2 | "Where do I get stuck most often?" | F03 outcome + F22 blocker_type + F20 | yes | clusters filtered to `outcome != success`, ordered by member_count |
| 3 | "What's the success rate by task type?" | F20 + F03 | yes (for grouping) | join `bridge_cluster_session` × `fact_cluster_metrics`, sort by success_rate |
| 4 | "What's my best workflow?" | F06 tool_mix + F03 + F20 | yes | clusters with highest success_rate; inspect tool_mix patterns |
| 5 | "Semantic search across all past sessions" | F20 + embeddings | no | k-NN on embedding for ad-hoc query text |
| 6 | "Find every session where I worked on X" | F20 + F26 codebase_subject + embeddings | no | similarity query + repo_slug filter |
| 7 | "Weekly digest of what I worked on" | F20 + F21 accomplishment | yes (on the week's sessions) | re-cluster sessions where `last_timestamp` in date range; one bullet per cluster |
| 8 | "Auto-generate ADR drafts" | F25 = decision-record + F20 | no | filter F25, summarize each session via F21 |
| 9 | "Codebase priorities over time" | F10 repo_slug + F20 + F26 | yes (per repo) | per-repo clusters; track member_count over weekly windows |
| 10 | "Wellbeing / cognitive load signal" | F29 tone + F03 + F16 + F22 | optional | rate of frustrated-tone sessions, especially after-hours, with high-blocker incidence |
| 11 | "Which prompting register works best?" | F28 prompt_register + F03 | no | crosstab F28 × F03 success rate |
| 12 | "Model A vs Model B for task type X" | F11 model_mix + F20 + F03 | yes | cluster on F20, within-cluster compare F11 by F03 |
| 13 | "Skill / connector gaps" | F06 tool_mix + F20 + F22 | yes | clusters of recurring manual workflows (high session count, low custom-tool/skill usage) |
| 14 | "Personal Economic Index" | F20 + F25 + F03 + cluster taxonomy F44 | yes (hierarchy) | map base clusters to a task taxonomy (Anthropic's O*NET-derived taxonomy is one option); aggregate by augmentation/automation flag at session level |
| 15 | "Learning trajectory" | F27 knowledge_domain + F16 over time | yes | first-appearance dates for each domain cluster |
| 16 | "When did my usage shift?" | F40 cluster_id_base + F16 over time | yes | weekly cluster-distribution histograms; drift detection across windows |
| 17 | "Anomaly: session unlike anything I've done" | F20 embedding + F43 distance_to_centroid | yes | high-distance outliers |
| 18 | "Subagent effectiveness" | F13 agent_depth + F03 + F20 | optional | crosstab F13 × F03 within cluster |

### Use cases that are *not* enabled by this design (honest)

- **Real-time anomaly detection.** This is a batch design. Adding streaming would be a separate effort.
- **Cross-user / team analytics.** Single-user by default. Multi-user needs identity resolution and stronger privacy guardrails (the four-layer Clio model).
- **Content-aware investigation** (e.g., "did I ever paste a secret"). The facets are intentionally privacy-sanitized. The raw transcripts are still in the star schema for this if needed, but it's not what the cluster layer is for.
- **Cross-product analytics** (Claude.ai Chat, Cowork, Office Agent). ccutils is Claude Code transcripts only, by design.
- **Anything requiring API data** (per-API-key cost, organization-level usage, Compliance API events). Out of scope by user constraint; the Enterprise Analytics API would be the right place for those.

---

## 8. Open decisions

1. **LLM provider for T02 / T05.** Anthropic API direct, or stay credential-free and read from a configured local model? Cost favors Haiku via API. Privacy favors local.
2. **Embedding model.** ColBERT (already an optional dep, good for retrieval) vs. a general sentence-transformer (better for k-means clustering). Probably both, behind a config flag.
3. **Clustering algorithm.** k-means (need to pick k, simpler) vs. HDBSCAN (auto-k, handles outliers natively, more compute). Lean HDBSCAN given the typical corpus shape.
4. ~~**Where to store embeddings.** DuckDB BLOB column (simple, one source of truth) vs. separate Parquet shard (faster k-NN).~~ **Resolved during build step 1** (see the schema-split note in §4): neither BLOB nor Parquet -- `fact_facet_embeddings.embedding FLOAT[384]` in DuckDB, so `array_cosine_similarity` / `array_inner_product` work natively in SQL. The table exists as a DDL stub; T03 (the populator) is still unbuilt.
5. **Incremental re-clustering.** Full recompute nightly is simplest. Incremental (assign new sessions to existing clusters, rebuild quarterly) is cheaper. Start simple.
6. **Privacy audit cadence.** Always-on, or only when exporting? Always-on adds cost; export-time is sufficient for personal use.

---

## 9. Order to build

Minimum viable path that yields "so what" at every step:

1. **T01 + Tier 1 facets only.** Done -- `populate_tier1_facets` (`src/ccutils/etl/fact_session_facets.py`) writes all 19 Tier 1 facets into `fact_session_facets` against `dim_facet_type`, reusing the heuristic columns `populate_dim_session_heuristics` already writes onto `dim_session`. No new use cases, but lays the foundation. (½ day)
2. **T02 with one Tier 2 facet: F20 task_description.** Single Haiku call per session. Unlocks use cases 1, 5, 6, 7 once embeddings + clustering are added. (1 day)
3. **T03 + T04 + T05.** Embedding, clustering, description. The "first map" appears. Unlocks use cases 1, 2, 7. (2 days)
4. **T07 cluster metrics.** Joins clusters to existing facts. Unlocks use cases 3, 4, 12. (½ day)
5. **T06 hierarchization.** Better navigation. Unlocks deeper drill-down. (½ day)
6. **Remaining Tier 2 facets (F22, F25, F28, etc.)** as use cases demand. Each new facet is a registry row + populator.

That ordering gets to a working "personal map of usage" in roughly a week of build, with the schema extension small enough that none of it is throwaway.

---

*Cross-references: `STAR_SCHEMA.md` for the v0.15 base schema; `CLAUDE.md` for ETL conventions (`lineage_upsert`, populator scoping, late-arrival handling).*
