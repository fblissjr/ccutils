# Changelog

All notable changes to this project will be documented in this file.

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
