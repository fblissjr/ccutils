# Changelog

All notable changes to this project will be documented in this file.

## 0.12.0

### Changed
- **CLI simplification**: Opinionated defaults, removed dead weight, grouped options
  - `convert` command absorbed into `local` -- pass a file as positional arg to convert directly: `ccutils session.jsonl`
  - No file arg = interactive picker (previous `local` behavior). File arg = convert it (previous `convert` behavior)
  - URL input support removed (`curl url > file.jsonl && ccutils file.jsonl` instead)
  - `convert` still works as a hidden alias for backwards compatibility
  - Thinking blocks and subagents/agents now **included by default** -- use `--no-thinking`, `--no-subagents` (local), `--no-agents` (all) to opt out
  - Removed `--schema` flag (auto-inferred from `--format`)
  - Removed `--json`, `--output-auto`, `--repo` (from local), `--limit` flags
  - Merged `--embed` + `--embed-model` into single `--embed [MODEL]` flag
  - Options grouped into sections (Output, Selection, Content, Processing, Embeddings) via `click-option-group`

### Added
- `click-option-group` dependency for grouped CLI help output

## 0.11.0

### Added
- **`convert` command**: Renamed from `json`, now supports all output formats via `--format` (html, duckdb, duckdb-star, json, json-star) and `--schema` (simple, star) -- single entry point for converting JSON/JSONL files or URLs
- **Token estimation breakdown**: `total_thinking_tokens` and `total_tool_io_tokens` columns on `fact_session_summary` -- thinking blocks and tool I/O were previously uncounted
- **`estimated_tokens` column** on simple schema `sessions` table
- **Inclusive agent metric rollup**: `fact_session_summary` now carries `_incl_agents` columns (`total_estimated_tokens_incl_agents`, `total_tool_calls_incl_agents`, `total_errors_incl_agents`, `total_duration_incl_agents`) that aggregate metrics from all descendant subagent sessions. Bottom-up rollup runs during `finalize_star_schema()` using `dim_session.depth_level`. `fact_agent_delegations` also carries denormalized `agent_estimated_tokens`
- **Semantic view updates**: `semantic_sessions` and `semantic_project_context` expose `_incl_agents` columns; `semantic_agent_delegations` exposes `agent_estimated_tokens`
- **CLI test coverage**: New test files for 5 previously untested commands -- `test_convert_cmd.py`, `test_schema_cmd.py`, `test_import_cmd.py`, `test_web_cmd.py`, `test_explore_cmd.py`

### Fixed
- **Orphan tool use preservation**: Tool calls interrupted before receiving a result (session killed mid-tool) are now stored in both simple and star schema DuckDB exports with NULL `output_text` and `result_message_id` -- previously silently dropped, creating asymmetry with JSON export which already included them
- **Token estimation accuracy**: Star schema ETL now counts thinking blocks and tool input/output in token estimates (previously only counted text blocks). Per-message `estimated_tokens` in `fact_messages` now includes all content types (thinking, tool I/O, text) for that message -- previously only counted text, making `SUM(estimated_tokens)` miss ~75% of tokens
- **URL project_name**: `convert` command now uses URL filename stem as `project_name` instead of temp directory name
- **CSS brace bug in import command**: Multi-session index used `.format()` which conflicted with CSS `{}` braces -- switched to f-string with doubled braces
- **Simple ETL duplication**: Extracted `_extract_session_core()` and `SimpleExtractionResult` dataclass to share ~200 lines of logic between DuckDB and JSON export paths

### Removed
- Dead `get_terminal_width` wrapper from `parsers/discovery.py`
- `json` CLI command (replaced by `convert`)

## 0.10.2

### Added
- **Project context views**: Two new semantic views for catching up on project state
  - `semantic_project_context`: sessions with first/last messages, intent, metrics -- ordered by recency
  - `semantic_project_files`: file activity aggregated by project with session count, read/write/edit totals
- **Session message columns**: `first_user_message` and `last_assistant_message` persisted on `dim_session` (truncated to 500 chars) -- previously extracted during ETL but discarded after heuristic classification
- **Date/time on all semantic views**: Every view now exposes a DATE field and time_of_day for filtering and sorting
  - `semantic_sessions`: `session_datetime`, `time_of_day`, `hour` from dim_time
  - `semantic_file_operations`: `full_date`, `time_of_day` from dim_date/dim_time
  - `semantic_session_chains`: `chain_start_date` derived from first_timestamp
  - `semantic_agent_delegations`: `delegation_date`, `time_of_day` from dim_date/dim_time
  - `semantic_file_evolution`: `first_seen_date`, `last_seen_date` derived from timestamps
  - `semantic_project_context`: `session_date`, `time_of_day` from dim_time
  - `semantic_project_files`: `last_touched_date` derived from timestamp
- **time_key on fact_session_summary**: Enables dim_time joins for session-level views

### Changed
- View count updated from 8 to 10 across all docs and docstrings
- Embedding pipeline docs updated to honestly describe current status (infrastructure for future semantic search, no built-in query consumer yet)

## 0.10.1

### Fixed
- **Master index HTML rendering**: `_generate_master_index()` now passes `total_projects`, `total_sessions`, `recent_date`, and `global_search_js` to the template -- previously rendered empty
- **Project index HTML rendering**: `_generate_project_index()` now passes `session_count` to the template
- **CSS class mismatch**: `.index-item-number` in 3 templates renamed to `.index-item-num` to match stylesheet
- **33 missing CSS definitions**: Added styles for `.file-tool-*`, `.edit-*`, `.tool-header`, `.tool-icon`, `.todo-header`, `.todo-items`, `.index-commit-*`, `.search-result-*`, `.search-modal`, `.disabled`, `.continuation`, `.commit-card-hash`, `.image-block`, `.date`
- **Docstring privacy**: Removed hardcoded username from `metadata.py` docstring
- **Star schema post-ETL wiring**: `local` command now runs post-ETL steps (session chains, agent delegations, file bridge, depth calculation) that were previously only called by the `all` command. New `finalize_star_schema()` public function.

### Changed
- **Score-based intent classification**: `classify_intent()` now counts keyword matches per intent and returns the one with the most hits (ties broken by priority order). Fixes compound messages like "implement new error handling" being misclassified as `bug_fix` instead of `feature`

### Removed
- Dead templates: `star_schema_dashboard.html`, `data_explorer.html` (never loaded by any Python code)
- Dead code: `entity_type_key` generation in `extractors.py` (unused since degenerate dimension switch)

### Internal
- Split `test_star_schema.py` (3382 lines, 38 classes) into 4 focused files: `_ddl`, `_etl`, `_analytics`, `_advanced`
- Shared fixtures extracted to `conftest.py`
- README.md rewritten for accuracy (22 tables, heuristic classification, all CLI options)
- Source docstrings updated from "25+ tables" to "22 tables + 8 views"
- STAR_SCHEMA.md intent section updated to document score-based matching

## 0.10.0

### Breaking Changes
- **Star schema rebuilt from 37 tables to 22 tables + 8 views**
  - 15 tables removed: stg_raw_messages, dim_message_type, dim_content_block_type, dim_error_type, dim_entity_type, dim_programming_language, dim_intent, dim_topic, dim_sentiment, dim_goal, dim_task, dim_attempt, fact_message_enrichment, fact_message_topics, fact_session_insights
  - LLM enrichment pipeline (`enrichment.py`) deleted -- required user-provided callbacks that nobody used
  - Removed exports: `run_llm_enrichment`, `run_session_insights_enrichment`, `run_goal_task_enrichment`
  - `_populate_reference_data()` removed -- pre-populated dimension rows were misleading
  - JSON export meta.json version changed from "1.0" to "2.0"

### Added
- **Heuristic classification** runs during ETL with zero external dependencies (no LLM, no API key)
  - `classify_intent()`: bug_fix, feature, refactor, debug, test, docs, review, explore (from first user message keywords)
  - `classify_complexity()`: trivial, simple, moderate, complex (from session metrics)
  - `classify_outcome()`: success, failure, unknown (from last assistant message + error rate)
  - `classify_domain()`: web, backend, data, devops, docs, mixed, unknown (from file extensions)
  - `classify_error_type()`: permission_denied, file_not_found, syntax_error, timeout, import_error, tool_error (from error text)
  - Results stored on `dim_session` (intent, complexity, outcome, domain) and `fact_errors` (error_type)
  - 39 tests for heuristic classifiers
- **Tool call duration tracking**: `duration_seconds` on `fact_tool_calls` (time between invoke and result)
- **Tool chain enhancements**: `next_tool_key` and `is_error` on `fact_tool_chain_steps`
- **Enhanced session summary**: `total_errors`, `unique_tools_used`, `unique_files_touched`, `max_conversation_depth`, `total_estimated_tokens` on `fact_session_summary`
- **Agent delegation metrics**: `agent_tool_calls`, `agent_errors`, `agent_duration_seconds` denormalized on `fact_agent_delegations`
- **File language detection**: `language` column on `dim_file` inferred from extension
- **Week of year**: `week_of_year` column on `dim_date`
- **New view**: `semantic_tool_patterns` -- common tool sequences with frequency and error rates

### Changed
- 6 low-cardinality dimension tables replaced with degenerate VARCHAR columns on fact tables (Kimball best practice)
- Embedding pipeline default changed from "summary" to "first_user_message" (summary depended on removed LLM enrichment)
- `dim_session` no longer has `goal_key`, `task_key`, `attempt_key` columns
- Star schema docs (`docs/STAR_SCHEMA.md`) fully rewritten

## 0.9.5

### Added
- **`--private` flag** for privacy-preserving exports: sanitizes absolute file paths in HTML, DuckDB, and JSON output
  - `PathSanitizer` class converts cwd-relative paths to relative, home-relative to `~/...`, leaves system paths unchanged
  - Applied at ETL time so all downstream consumers get clean data automatically
  - Available on all commands: `local`, `all`, `json`, `web`, `import`
  - 49 new tests (31 unit + 18 integration)

### Changed
- `_export_to_html` cleanup: eliminated temp-file round-trip by passing loglines directly to `generate_html(loglines=)`, reused `_group_loglines_by_session` helper, simplified `auto_open` logic, removed unused `metadata` binding and `json` import

## 0.9.4

### Removed
- **Gist upload feature**: Removed `--gist` option from `local`, `web`, and `json` commands; deleted `create_gist()`, `GistError`, `inject_gist_preview_js()`, gist preview JS, and ~310 lines of gist tests
- Backward-compat re-exports (`build_project_choices`, `build_session_choices`, etc.) from `parsers/__init__.py`

### Changed
- **Codebase cleanup round 2**: Eliminated ~845 additional lines across 5 phases
  - Static file loading uses `importlib.resources.files()` instead of `Path(__file__)` for wheel/zip compatibility
  - Star ETL `_extract_star_data()` decomposed: `BlockContext` dataclass + extracted `_handle_tool_use_block()` and `_handle_tool_result_block()` handlers
  - Import command DuckDB export now reuses `simple/etl.py:export_session_to_duckdb()` via new `iter_loglines()` adapter, replacing ~200 lines of duplicated insert logic

## 0.9.3

### Changed
- **Codebase cleanup**: Eliminated ~790 lines of duplicated/dead code across 7 phases
  - Deleted 170-line `generate_html_from_session_data` clone; `generate_html()` now accepts optional `loglines` param
  - Unified `_extract_text()` duplicate in `parsers/metadata.py` with `extract_text_from_content()` from `parsers/session.py`
  - Extracted ~232 lines of inline CSS/JS from `export/html.py` to `src/ccutils/static/` files
  - New shared JSONL parser (`parsers/jsonl_reader.py`) with `iter_session_entries()` generator replaces triple-parsed sessions in simple and star schema ETL
  - Decomposed `star/etl.py` with `StarExtractionResult` dataclass; `_load_dimensions`/`_load_facts` take structured result instead of 20+ positional args
  - Removed 6 deprecated wrapper functions from `parsers/discovery.py`; imports now go through `tui/` package
  - Extracted `handle_gist_upload()` and `maybe_open_browser()` helpers to `cli/utils.py`

## 0.9.2

### Added
- **Styled TUI package** (`src/ccutils/tui/`): New modular package for terminal UI with semantic coloring
  - `theme.py`: Color constants for prompt_toolkit (questionary) and Rich, with model-family sub-styles (opus=bold, sonnet=normal, haiku=italic magenta)
  - `formatters.py`: Pure formatting functions for relative dates, durations, project names, summaries, branch names, file sizes, message counts
  - `layout.py`: Terminal width detection and proportional column width calculation with `ColumnSpec` dataclass
  - `components.py`: Rich table renderers using ratio-based columns that expand to fill terminal width; summary column gets remaining space
  - `selection.py`: Questionary choice builders using `FormattedText` (list of `(style, text)` tuples) for per-segment coloring in checkboxes/selects
- Styled questionary chrome: blue pointer/highlight, green selected markers, dim instructions via `questionary_style()`
- Color-coded session labels: dates in yellow, project names in blue, models in magenta, counts in green, summaries in default

### Changed
- `local` command now uses styled choices and styled questionary chrome for both flat and two-phase selection modes
- `web` command session picker now uses styled choices with color-coded repo/date/title
- `import` command interactive picker now uses styled choices with color-coded date/count/name
- Project table title now shows total counts: "Projects (N found, M sessions)"
- Session table columns use `expand=True` with `ratio` so summaries fill remaining terminal width
- `discovery.py` refactored: display/selection functions replaced with thin wrappers delegating to `tui/` (~400 lines removed)
- All backward-compatible re-exports preserved in `parsers/__init__.py`

## 0.9.1

### Changed
- **Deterministic agent delegation linking**: `_link_agent_delegations` now uses `progress` records from JSONL for zero-ambiguity matching (confidence 1.0) instead of relying solely on timestamp proximity heuristics
  - New `stg_task_agent_map` staging table captures `tool_use_id` -> `agent_id` links from progress records during ETL
  - Falls back to timestamp-based heuristic matching (confidence 0.5-0.8) for older data without progress records
  - Multiple simultaneous Task delegations are now matched correctly

## 0.9

### Added
- **Session chains**: `dim_session_chain` groups sessions sharing the same `slug` into chains
  - `chain_key` added to `dim_session` for chain membership
  - `semantic_session_chains` view for chain-level analytics
  - Chains auto-built during batch export from shared slug values
- **Agent delegation tracking**: `fact_agent_delegations` links agent sessions to their parent's Task tool_use calls
  - Heuristic matching by timestamp proximity with confidence scoring
  - Captures task description, prompt, subagent_type from Task tool inputs
  - `semantic_agent_delegations` view joining parent/agent sessions with metrics
- **Session hierarchy**: Goal > Task > Attempt dimensional tables
  - `dim_goal`, `dim_task`, `dim_attempt` tables (populated via LLM enrichment)
  - `goal_key`, `task_key`, `attempt_key` soft FKs on `dim_session`
  - `run_goal_task_enrichment(conn, classify_func)` enrichment API
- **ColBERT embedding pipeline**: Semantic matching via PyLate (optional dependency)
  - `EmbeddingPipeline` class with lazy model loading (`mxbai-edge-colbert-v0-32m`)
  - `embed_sessions()`: Embed session summaries into `fact_session_embeddings`
  - `match_delegations()`: Re-score agent delegation matches using semantic similarity
  - `cluster_sessions()`: K-means clustering with auto task assignment
  - `--embed` and `--embed-model` CLI flags for batch export
- **Cross-session file bridge**: `bridge_session_file` aggregates file operations across sessions
  - Per-file operation counts (read/write/edit) by session
  - `semantic_file_evolution` view for files touched by multiple sessions
- **Session slug storage**: `slug` column in `dim_session` preserves chain resume identifiers
- **Agent depth calculation**: `depth_level` correctly calculated for nested agent hierarchies
  - Iterative batch calculation handles arbitrarily deep nesting
  - Single-session ETL attempts parent lookup during insert

### Dependencies
- Added `pylate` as optional dependency: `pip install ccutils[colbert]`

## 0.8

### Added
- **Two-phase session selection UI**: Redesigned `local` command with project-first navigation
  - Phase 1: Pick project(s) from a rich summary table showing session counts, models, branches, last active date
  - Phase 2: Pick session(s) within selected projects with detailed metadata (model, branch, duration, message count)
  - Automatic skip of phase 1 when only one project matches (or when using `-p` filter)
  - `--flat` flag preserves old single-list behavior
  - `--expand-chains` flag shows individual sessions in resumed chains
- **Rich metadata extraction**: New `SessionMetadata` dataclass and extraction pipeline
  - `extract_rich_metadata()`: Single-pass extraction of cwd, model, branch, slug, duration, message counts
  - `get_meaningful_summary()`: Smarter summary extraction that skips interrupted/error/XML messages
  - `shorten_model_name()`: Human-friendly model names (`claude-opus-4-6` -> `opus-4.6`)
  - `format_duration()`: Human-readable duration (`45m`, `1h 5m`)
  - `derive_project_name()`: Derives project name from `cwd` field (actual directory name, not encoded path)
- **Rich terminal tables**: `rich` library for colorized project and session tables
  - `print_project_table()`: Summarizes projects with session counts, models, branches
  - `print_session_table()`: Shows session details with relative dates, model, branch, duration
- **New discovery functions**: `find_local_sessions_rich()`, `group_by_project()`, `build_project_choices()`, `build_session_choices_for_projects()`

### Changed
- Default `local` command now uses two-phase selection (projects then sessions)
- Project names derived from `cwd` metadata field when available (more accurate than folder name parsing)
- Session summaries no longer show `[Request interrupted...]` or XML system prompts

### Dependencies
- Added `rich` for terminal formatting

## 0.7

### Added
- **Repo display and filtering in `web` command**: Shows which GitHub repo each session belongs to (adapted from upstream simonw/claude-code-transcripts v0.6)
  - `extract_repo_from_session()`: Extracts repo from API session metadata (outcomes or sources URL)
  - `enrich_sessions_with_repos()`: Adds `repo` key to session list data
  - `filter_sessions_by_repo()`: Client-side filtering by repo name
  - Session picker now shows `{repo}  {date}  {title}` instead of `{session_id}  {date}  {title}`
  - `--repo` flag now filters session list in addition to setting default for commit links
- **Un-nested tool parameters in star schema**: Extract common tool parameters from JSON blobs for easier querying
  - New columns in `fact_tool_calls`: `file_path`, `command`, `pattern`, `query_text`
  - New `fact_tool_input_params` table: Key-value pairs for all tool input parameters
  - Updated `semantic_tool_calls` view to include extracted columns
  - Supports queries like `SELECT * FROM fact_tool_calls WHERE file_path IS NOT NULL`
- **Star schema support in `all` command**: Full star schema support for batch exports
  - New format options: `--format duckdb-star`, `--format json-star`
  - Uses 25+ dimensional tables for richer analytics
  - Progress reporting shows row counts, DB size, and processing rate
- **Performance options for batch processing**:
  - `-j/--jobs N`: Parallel workers for processing (default: 1)
  - `--batch-size N`: Sessions per transaction batch (default: 10)
  - Progress callback now includes stats (rows_inserted, db_size_mb, rate)
- **Enhanced progress reporting**: Shows rows processed, storage size, and sessions/sec rate
- **Claude.ai account export import**: New `import` command to convert Claude.ai account exports (from Settings > Privacy)
  - Supports all existing output formats: HTML, DuckDB
  - Lists conversations: `ccutils import ./export --list`
  - Interactive selection: `ccutils import ./export --interactive`
  - Filter by conversation UUID: `ccutils import ./export -c <uuid>`
  - Preserves thinking blocks and tool calls
  - New parser functions: `parse_claude_ai_export()`, `convert_conversation_to_loglines()`
- **JSON export format**: Export sessions to JSON in addition to HTML and DuckDB
  - `--format json`: Simple schema (sessions, messages, tool_calls, thinking) in single JSON file
  - `--format json-star`: Star schema exported as directory structure (meta.json + dimensions/*.json + facts/*.json)
  - New `--schema` option to explicitly set schema type (`simple` or `star`)
  - Backwards-compatible: compound format names (`duckdb-star`, `json-star`) still work
  - New functions: `resolve_schema_format()`, `export_sessions_to_json()`, `export_star_schema_to_json()`
- **Multi-select for local command**: Select multiple sessions using SPACE, confirm with ENTER
- **DuckDB export from local command**: New `--format` option supports `html`, `duckdb`, `duckdb-star`, `json`, or `json-star`
- **Subagent support**: New `--include-subagents` flag auto-includes related agent sessions (recursive)
- **Agent metadata in DuckDB**: Sessions and messages now track agent relationships
  - Sessions table: `is_agent`, `agent_id`, `parent_session_id`, `depth_level` columns
  - Messages table: `is_sidechain` column
  - Star schema: Same columns in `dim_session` and `fact_messages`
- New functions: `extract_session_metadata()`, `find_agent_sessions()` for agent discovery

### Changed
- Expanded README documentation for star schema analytics with comparison table, quick start code, and overview of dimensions/facts
- `local` command now uses `questionary.checkbox()` for multi-select (was single-select)

### Added (continued)
- Full-text search across the entire HTML archive generated by the `all` command
  - Search index (`search-index.js`) is generated alongside HTML files
  - In-browser JavaScript search with snippet highlighting
  - Search works offline and on `file://` protocol (unlike existing per-session search)
  - Results show project, type, timestamp, and link directly to the matching content
  - Mobile-friendly responsive design
- New CLI option `--no-search-index` to skip search index generation for faster/smaller output
- New functions: `extract_searchable_content()`, `extract_snippet()` for search indexing
- DuckDB export for structured analytics on transcript data
  - Export sessions, messages, tool calls, and thinking blocks to a single DuckDB database
  - Query your transcripts with SQL for analytics and insights
  - New CLI option `--format` to choose output format: `html` (default), `duckdb`, or `both`
  - New CLI option `--include-thinking` to include thinking blocks in DuckDB export (opt-in, can be large)
  - New functions: `create_duckdb_schema()`, `export_session_to_duckdb()`, `generate_duckdb_archive()`
