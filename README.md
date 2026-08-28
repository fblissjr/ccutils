<!-- path-privacy: skip-file -- documents ~/.claude/projects as a CLI default inside fenced help output, where a per-line marker would render literally -->
# ccutils

Claude Code transcript analytics: browsable HTML/markdown transcripts and a DuckDB star-schema warehouse with full ETL lineage, built from the session JSONL Claude Code already writes.

> **Origin:** This project began as a fork of Simon Willison's [claude-code-transcripts](https://github.com/simonw/claude-code-transcripts). It has since diverged significantly with star schema analytics, modular architecture, and as a broader Claude utility.

## Installation

```bash
uv tool install -e .
```

Or run without installing:

```bash
uv run ccutils --help
```

## Quick Start

```bash
# Interactive two-phase picker - select projects, then sessions
ccutils

# Convert a single session file (opens in browser)
ccutils session.jsonl

# Export to DuckDB (v0.15 star schema + four-tier ETL + Parquet lake)
ccutils --format duckdb -o ./analytics

# Everything on the machine, in one warehouse
ccutils --source --format duckdb -o ./analytics

# Open what you built
ccutils open -o ./analytics
```

## Commands

One command converts; two do the things that are not converting.

| Invocation | What it does |
|---|---|
| `ccutils` | Interactive picker -- **the default**, no subcommand needed |
| `ccutils PATHS...` | Convert exactly the session files you name |
| `ccutils --source` | Convert every session under a directory (the Claude projects dir by default) |
| `ccutils import` | Import a Claude.ai account export (Settings > Privacy > Export) |
| `ccutils open` | Open a built warehouse in the DuckDB SQL UI |

Removed in 0.20.0, with no aliases: `local` and `all` were one operation
split by scope, and the split is what let their behaviour drift apart --
global post-loop sources ran on one path and not the other. `convert` was a
hidden duplicate of `local` and is now the name of the single conversion
command. `web` and `schema` were deleted outright: one read a different data
source through an undocumented API, the other was a generic JSON
introspector, and neither had a path to the warehouse. The old names fail
with a pointer rather than silently redirecting.

### Converting sessions (default)

The default command. With no arguments, launches a two-phase interactive picker (projects then sessions). Pass a session file to convert it directly.

Thinking blocks and subagent sessions are included by default.

```bash
ccutils                                          # Interactive picker (writes to <archive>/picked/)
ccutils session.jsonl                            # Convert one file, open in browser
ccutils a.jsonl b.jsonl -o ./out                 # Convert several
ccutils session.jsonl --format duckdb -o ./out   # v0.15 star schema DuckDB
ccutils session.jsonl --format markdown -o ./out # One .md per session (quick sharing)
ccutils --format duckdb -o ./analytics           # Pick sessions, star schema
ccutils -p myproject                             # Filter by project name
ccutils --flat                                   # Single-list picker mode
ccutils --no-thinking --no-subagents             # Exclude thinking blocks / related agent sessions
ccutils --format duckdb --embed -o ./out         # With ColBERT embeddings
ccutils --format duckdb --llm-facets -o ./out    # + Tier 2 LLM facets (F20 via Haiku)
```

`--llm-facets` requires `ANTHROPIC_API_KEY` in the environment OR a `ccutils-anthropic` keychain entry (`security add-generic-password -s ccutils-anthropic -a $USER -w`). Star schema only.

Note: pairing `--llm-facets` with `--format json` runs the full LLM extraction against a temporary DuckDB that's discarded after the JSON export — you pay the API cost but don't get a queryable database to inspect the F20 outputs. If you want to see the extracted facets, use `--format duckdb` instead.

### Converting everything

Batch convert every session. Agents and thinking blocks included by default.

```bash
ccutils --source -o ./archive                    # Flat HTML archive: index + one file per session
ccutils --source --format markdown -o ./md-archive   # One .md per session, per-project tree
ccutils --source --format duckdb -o ./analytics  # v0.15 star schema for all sessions
ccutils --source --format duckdb --embed -o ./out    # With ColBERT embeddings
ccutils --source --format duckdb --llm-facets -o ./out   # + Tier 2 LLM facets
ccutils --source -j 4 --batch-size 20 -o ./archive   # Parallel processing
ccutils --source --no-subagents --no-thinking    # Exclude agents and thinking (any format)
ccutils --source --dry-run                       # Preview without converting
```

### import

Import Claude.ai web conversation exports (the ZIP/directory from Settings > Privacy). HTML output only.

```bash
ccutils import ./my-claude-export --open         # HTML, opens in browser
ccutils import ./export --interactive            # Pick conversations
ccutils import ./export --list                   # List without converting
```

### Exploring the warehouse

DuckDB ships its own local UI, which is a better SQL notebook than anything
worth wrapping, so `ccutils open` launches it rather than reimplementing it:

```bash
ccutils open                      # the default archive
ccutils open -o ./analytics       # a warehouse you built elsewhere
duckdb -ui ./analytics/archive.duckdb   # or drive it yourself
```

It launches; it does not build. With no warehouse there it says so and tells
you the command that would create one.

The first run fetches the `ui` extension (one-time, needs network). For a
terminal UI instead, any DuckDB client works — point it at the same file.

## Export Formats

Four formats: `html`, `markdown`, `duckdb`, `json` (plus `both` = html+duckdb on `all`). `html` and `markdown` are render-only transcripts; `duckdb` and `json` both write the v0.15 star schema. Coverage differs on purpose: the warehouse formats ingest **every** session on disk (including ones with no summary line), while html/markdown skip warmup/no-summary sessions for a curated browsing archive.

### HTML Transcripts

One self-contained `.html` file per session -- styling and script inlined and
pinned by sha256 in a strict CSP, so a transcript is a single file you can
open from disk or send to someone. Batch output is flat: one filterable
`index.html` plus one file per session; projects are a filter over the one
list, not a page tree. In-page navigation is a collapsible prompt list plus a
client-side filter box (no network calls -- the old fetch-based search
silently returned zero results when opened from `file://`).

```bash
ccutils -o ./transcript --open
ccutils --source -o ./archive               # Flat archive: index + one file per session
```

### DuckDB Analytics (v0.15 star schema)

```bash
ccutils --format duckdb -o ./analytics
```

v0.15 rebuilds the ETL as a four-tier pipeline:

1. **Tier 0** -- raw JSONL on disk (Claude Code writes).
2. **Tier 1** -- Parquet lake under `<output>/parquet_lake/projects/<project>/<session>/`. Typed-columnar cache of every parsed JSONL entry. Persistent and re-derivable; the DuckDB warehouse can be torn down and rebuilt from Parquet without re-parsing JSONL.
3. **Tier 2** -- DuckDB staging table (`stg_log_entries`) loaded from Parquet via `read_parquet()`, cleared at the end of every run.
4. **Tier 3** -- DuckDB warehouse: dimensions, facts, and semantic views consumers query. The warehouse is persistent and incremental: re-running the CLI against the same database file accumulates new sessions instead of wiping and rebuilding.

Every fact carries the v0.15 lineage convention: `created_at`, `last_updated_at`, `created_by_version_key`, `last_updated_by_version_key`, `etl_run_id`, `record_source`, `hash_diff`, plus soft-delete (`is_deleted`, `deleted_at`). Re-running ETL on unchanged source is a no-op (the `hash_diff` gate prevents spurious UPDATEs). DDL migrations are tracked via `meta_schema_version`.

**Run metadata (three grains):** every CLI invocation writes one `fact_etl_batch_runs` row; every session ETL writes one `fact_etl_runs` row (linked via `batch_run_id`, carrying a CDC data window of min/max entry timestamps); every pipeline node writes one `fact_etl_steps` row with real affected-row counts. Counts on the parent rows are derived from the children at completion -- the audit trail cannot drift from what the DML actually did. Query `semantic_etl_runs` for the joined picture.

Subagent transcripts are first-class sessions: agent files carry their parent's `sessionId` internally, so ccutils keys them by file identity (`agent-<id>`), links them to their parent session, and computes delegation depth -- your agents don't silently merge into their parents.

**Tables populated by `run_v15_etl`:**

- **Lineage / Meta:** `dim_etl_version`, `fact_etl_batch_runs`, `fact_etl_runs`, `fact_etl_steps`, `meta_schema_version`.
- **Dimensions:** `dim_session` (with intent/complexity/outcome/domain enrichment + subagent linkage), `dim_project`, `dim_tool`, `dim_model`, `dim_file`, `dim_session_chain`, `dim_facet_type` (facet registry).
- **Core facts:** `fact_messages`, `fact_tool_uses`, `fact_tool_results` (R1 structured `toolUseResult` payloads: Edit `structured_patch`, Bash `exit_code` / `interrupted`, Read `num_lines`, Agent rollups), `fact_token_usage` (R11 cache split: `cache_creation_5m_tokens` + `cache_creation_1h_tokens`), `fact_session_summary`.
- **Entry-type facts:** `fact_attachments`, `fact_progress_events`, `fact_system_events`, `fact_meta_events` (permission-mode time series), `fact_file_history_snapshots`, `fact_queue_operations`, `fact_pr_links`.
- **Derived:** `fact_file_operations` + `bridge_session_file`, `fact_diagnostics`, `fact_plan_revisions` (structural outcome from `fact_tool_results.is_error`), `fact_agent_delegations` (cross-session linkage via `dim_session.agent_id`), `fact_errors`, `fact_tool_chain_steps`.
- **Facets:** `fact_session_facets` Tier 1 (F01-F19, SQL-computed; always on). Tier 2 (F20+, LLM-extracted via Haiku) is opt-in via `--llm-facets`.

**Populated after the per-session loop** (global sources, not per-session, so they are outside `run_v15_etl` and only the batch archive path reaches them):

- `dim_prompt` -- user prompts from the Claude Code prompt-history JSONL, linked to sessions by `sessionId`.
- `dim_memory` + `bridge_memory_link` -- Claude Code **auto memory**: the markdown Claude writes to itself per project, plus subagent memory in all three scopes. A **Type 2 SCD**, because Claude Code overwrites memory files in place and keeps no history of its own -- one row per (memory file, content version), so re-ingesting does not destroy what the memory said before. Deleted memories are closed, not erased. `bridge_memory_link` carries the link graph in both forms Claude actually writes -- `[[wiki-links]]` between topic files and the `[Title](file.md)` entries by which `MEMORY.md` indexes them -- kept apart by `link_syntax`, dangling links included. Query `semantic_memory` / `semantic_memory_links` for the current state, `dim_memory` for the history.

> The JSON export mirrors the warehouse, so it includes `dim_memory` — auto-memory bodies and all. That is the same treatment `fact_messages` already gives full transcript text, but worth knowing before you move an export somewhere else.

**Not yet populated (DDL stubs only):** `fact_content_blocks`, `fact_code_blocks`, `fact_entity_mentions`, `fact_tool_input_params`. `fact_session_embeddings` is populated only by `--embed` runs (empty otherwise). `fact_turn_durations` / `fact_stop_events` are subsumed by `fact_system_events`; `fact_tool_calls` by `fact_tool_uses` + `fact_tool_results`.

```sql
-- Sessions ranked by uncached-equivalent token cost
SELECT
  ds.session_id,
  fss.total_input_tokens,
  fss.total_output_tokens,
  fss.total_cache_creation_5m_tokens,
  fss.total_cache_creation_1h_tokens,
  fss.total_cache_read_tokens,
  fss.total_uncached_equivalent_tokens
FROM fact_session_summary fss
JOIN dim_session ds USING (session_key)
ORDER BY total_uncached_equivalent_tokens DESC
LIMIT 20;

-- Tool usage by category
SELECT dt.tool_category, COUNT(*) AS uses
FROM fact_tool_uses ftu
JOIN dim_tool dt USING (tool_key)
GROUP BY dt.tool_category
ORDER BY uses DESC;

-- Bash invocations that exited non-zero or were interrupted (R1 toolUseResult capture)
SELECT
  ftu.session_id,
  json_extract_string(ftu.input_json, '$.command') AS command,
  ftr.bash_exit_code,
  ftr.bash_interrupted,
  ftr.timestamp
FROM fact_tool_uses ftu
JOIN fact_tool_results ftr USING (tool_use_id)
WHERE ftu.tool_name = 'Bash'
  AND (ftr.bash_exit_code <> 0 OR ftr.bash_interrupted = TRUE)
ORDER BY ftr.timestamp DESC
LIMIT 20;

-- ETL observability: per-session run status, duration, batch context,
-- CDC window, and step rollups in one view
SELECT
  etl_run_id,
  batch_run_id,
  status,
  duration_ms,
  data_start_ts,
  data_end_ts,
  step_count,
  rows_inserted,
  rows_updated,
  ccutils_version
FROM semantic_etl_runs
ORDER BY started_at DESC
LIMIT 10;

-- Orchestration-level scorecard: what did each CLI invocation do
SELECT
  status, sessions_seen, sessions_succeeded, sessions_failed,
  rows_inserted, data_start_ts, data_end_ts, output_format
FROM fact_etl_batch_runs
ORDER BY started_at DESC
LIMIT 5;
```

### JSON Export

The v0.15 star schema as a JSON directory tree (`meta.json` + `dimensions/` + `facts/`).

```bash
ccutils --format json -o ./json-export/
```

## Archiving the full corpus (encrypted)

`ccutils --source` with no value walks every project under your Claude config dir (the `--source` default) and writes a single `archive.duckdb` plus a `parquet_lake/` cache covering every session on the machine. ccutils does not do encryption itself -- that's an ops concern, kept out of the tool's scope -- but the export lands as plain files, so any encrypted-volume or encrypt-the-tarball approach works. On macOS the least-friction option is an encrypted sparse disk image: create it once, then mount → export → detach each run.

```bash
# One-time: create a password-protected, AES-256 encrypted sparse image.
# Size it for your corpus -- a full-system DuckDB + parquet lake is
# typically tens to a few hundred MB; grow the image if needed.
IMG=cc-archive.sparseimage   # put this wherever you keep backups
hdiutil create -size 5g -type SPARSE -encryption AES-256 \
  -fs APFS -volname CCArchive "$IMG"

# Each run: mount (prompts for password), export everything, detach.
hdiutil attach "$IMG"
ccutils --source --format duckdb -o /Volumes/CCArchive/$(date +%Y-%m-%d)
hdiutil detach /Volumes/CCArchive
```

The encrypted volume covers both the DuckDB file and the parquet lake (they're just files on it). Two caveats for a full-system export:

- **`--private` is best-effort and not wired through the v0.15 ETL** (the flag is rejected on `duckdb`/`json`). On the render formats (html, markdown) it only masks cwd/home-prefixed paths in a subset of channels -- `tool_use` inputs (`file_path`/`command`/`content`/`path`) and string `tool_result` content. It does **not** sanitize message text, thinking blocks, non-message entries (e.g. `file-history-snapshot` paths), project directory names, or absolute paths from another machine/user pasted into content. When it can't resolve a working directory (agent transcripts, `.json`/claude.ai exports) it now prints a loud warning rather than silently no-opping. Treat `--private` as a convenience for local encrypted archives, not a guarantee for public sharing; review output before sharing. (Comprehensive channel-walking is a tracked follow-up.)
- **The archive is machine-wide transcript data -- keep it out of your checkouts.** With no `-o` the output goes to `~/.ccutils/claude-archive`, which is home-anchored precisely so a default run never writes unredacted transcripts for every project on the machine into whatever repo you happened to be standing in. If you pass `-o` yourself, point it somewhere outside your worktrees (an encrypted volume as above, or any path a `git add -A` cannot reach).
- **Tier 2 LLM facets cost real money at full-system scale.** `--llm-facets` runs one Haiku call per session -- pennies on a handful of sessions, but potentially dollars across hundreds. Omit it for a bulk archive run, or run it separately on a filtered subset.

Portable alternative (any OS), encrypting the export as a single artifact with [age](https://github.com/FiloSottile/age):

```bash
ccutils --source --format duckdb -o ./archive
tar czf - ./archive | age -r <recipient-key> > archive.tar.gz.age && rm -rf ./archive
```

## Common Options

```bash
# Output
-o, --output PATH          Output DIRECTORY, always. --format duckdb writes
                           DIR/archive.duckdb + DIR/parquet_lake/ inside it.
                           Default: ~/.ccutils/claude-archive -- deliberately
                           outside any checkout (single-file conversion with
                           no -o still uses a temp dir)
-s, --source [PATH]        Convert every session under PATH instead of
                           picking. With no value, the Claude projects dir.
                           Mutually exclusive with PATHS
--format FORMAT            html, markdown, duckdb, json (+ both with --source)
--open                     Open result in browser

# Content (included by default -- use flags to exclude)
--no-thinking              Exclude thinking from outputs. Drops thinking from
                           dim_session messages + Tier 2 facet inputs and
                           clears the staging artifact (fact_messages already
                           excludes thinking by projection). Parquet lake is
                           unaffected -- delete it post-run if needed.
--no-subagents             Exclude related agent sessions
--include-temp-sessions    Include sessions whose cwd is under the OS temp
                           directory (excluded by default -- typically
                           sandboxed/ephemeral tooling like eval harnesses,
                           not real projects)
--private                  Sanitize file paths for sharing (html/markdown only; not wired through the v0.15 ETL)

# Rejected rather than ignored
# --embed and --llm-facets write to the star schema, so they are a hard
# error on --format html/markdown rather than a silent no-op. --private is
# a hard error on duckdb/json for the same reason.

# Selection
--flat                     Flat single-list mode (picker)
--expand-chains            Show individual sessions in resumed chains (picker)
-p, --project TEXT         Filter by project name
--dry-run                  Preview without converting (--source)

# Enrichment (star schema only)
--embed [MODEL]            Run ColBERT embeddings (optionally specify model)
--llm-facets               Extract Tier 2 LLM facets (F20 via Haiku)

# Batch processing (--source)
-j, --jobs N               Parallel workers (default: 1)
--batch-size N             Sessions per batch (default: 10)
-q, --quiet                Suppress output except errors
--no-search-index          Deprecated no-op (there is no search index); accepted for compatibility
```

## Documentation

- [Star Schema Reference](docs/STAR_SCHEMA.md) -- table definitions, populator notes, example queries.
- [Facet & Cluster Pipeline](docs/FACET_CLUSTER_PIPELINE.md) -- Tier 1/2/3 facet design, status, and roadmap.

## Development

```bash
uv run pytest tests/ --confcutdir=tests   # Full suite (1 skipped test = live-API smoke, expected)
uv run ccutils --help                     # Run development version
uv run pytest --cov=ccutils               # Coverage
```

Project conventions and hard-won pitfalls live in `CLAUDE.md`.

## License

Apache-2.0
