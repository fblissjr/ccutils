<!-- path-privacy: skip-file -- references universal ~/.claude data paths (not personal) -->
# ccutils

Claude utilities for session transcripts, star schema analytics, and probably more as it comes up as a use case in my day to day.

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

# Export to DuckDB for SQL analytics
ccutils --format duckdb -o ./archive

# Export with star schema (28 tables + 14 views)
ccutils --format duckdb-star -o ./analytics
```

## Commands

| Command | Description |
|---------|-------------|
| `local` | Interactive picker + single-file conversion -- **default** (no subcommand needed) |
| `all` | Batch convert all sessions (HTML archive, DuckDB, or JSON) |
| `web` | Import from Claude API (auto-detects credentials from macOS keychain) |
| `explore` | Open DuckDB database in harlequin (requires `ccutils[explore]`) |
| `import` | Import Claude.ai account exports (Settings > Privacy > Export) |
| `schema` | Inspect JSON structure without exposing content (safe to share publicly) |

### local (default)

The default command. With no arguments, launches a two-phase interactive picker (projects then sessions). Pass a session file to convert it directly.

Thinking blocks and subagent sessions are included by default.

```bash
ccutils                                          # Interactive picker
ccutils session.jsonl                            # Convert file, open in browser
ccutils session.jsonl --format duckdb-star -o .  # Star schema from file
ccutils --format duckdb-star -o ./analytics      # Pick sessions, star schema
ccutils -p myproject                             # Filter by project name
ccutils --flat                                   # Legacy single-list mode
ccutils --no-thinking --no-subagents             # Exclude thinking/agents
ccutils --format duckdb-star --embed -o .        # With ColBERT embeddings
```

### all

Batch convert every session. Agents and thinking blocks included by default.

```bash
ccutils all -o ./archive                         # HTML archive with search index
ccutils all --format duckdb-star -o ./analytics   # Star schema for all sessions
ccutils all --format duckdb-star --embed -o ./out # With ColBERT embeddings
ccutils all -j 4 --batch-size 20 -o ./archive    # Parallel processing
ccutils all --no-agents --no-thinking             # Exclude agents and thinking
ccutils all --dry-run                            # Preview without converting
```

### import

Import Claude.ai web conversation exports (the ZIP/directory from Settings > Privacy).

```bash
ccutils import ./my-claude-export --open         # HTML, opens in browser
ccutils import ./export --format duckdb -o data.duckdb  # DuckDB
ccutils import ./export --interactive            # Pick conversations
ccutils import ./export --list                   # List without converting
```

### web

Import a session from the Claude API. Auto-detects credentials from macOS keychain.

```bash
ccutils web                                      # Interactive session picker
ccutils web SESSION_ID -o ./transcript --open     # Convert specific session
ccutils web --repo owner/name                    # Filter by GitHub repo
```

### explore

Open a star schema DuckDB database in harlequin for interactive SQL exploration.

```bash
uv pip install ccutils[explore]    # one-time setup
ccutils explore ./analytics/archive.duckdb
```

### schema

Inspect JSON file structure without exposing content. Output is safe to share publicly or paste into AI assistants.

```bash
ccutils schema conversations.json
ccutils schema ./my-claude-export/               # Inspect all files in directory
ccutils schema ./export --json > schema.json     # Machine-readable output
```

## Export Formats

Schema type is auto-inferred from `--format`: `duckdb-star` and `json-star` use star schema, plain `duckdb` and `json` use simple.

### HTML Transcripts

Clean, mobile-friendly HTML with pagination, commit timeline, tool stats, and full-text search.

```bash
ccutils -o ./transcript --open
ccutils all -o ./archive                    # Archive with master index and search
```

### DuckDB Analytics

#### Simple Schema (4 tables)

```bash
ccutils --format duckdb -o ./archive
```

Tables: `sessions`, `messages`, `tool_calls`, `thinking`

#### Star Schema (28 tables + 14 views)

```bash
ccutils --format duckdb-star -o ./analytics
```

Dimensional model designed for analytics:

- **7 dimensions:** sessions (with heuristic classifications + entrypoint/title/agent_type), projects, tools (with categories), models (with families), dates, times, prompts (from ~/.claude/history.jsonl)
- **6 core facts:** messages (with actual API token counts), tool calls (with duration tracking), session summaries (with inclusive agent metric rollup + actual tokens + turn durations), file operations, errors (with type classification), tool chain steps
- **4 telemetry facts:** per-API-response token usage (with cache breakdown), turn durations, LSP diagnostics, stop events
- **5 granular tables:** files (with language detection), session chains, content blocks, code blocks, entity mentions
- **3 agent/bridge tables:** agent delegations (with denormalized metrics), cross-session file tracking, task-agent mapping
- **2 optional:** ColBERT embeddings, tool input parameters
- **13 semantic views:** pre-joined views for common queries (includes project context, file tracking, token usage, cost analysis, prompt history)

#### Heuristic Classification

The star schema ETL runs heuristic classification during ingestion with zero external dependencies -- no LLM, no API key needed. Results are stored on `dim_session`:

| Classifier | Method | Values |
|------------|--------|--------|
| **Intent** | Score-based keyword matching on first user message | bug_fix, feature, refactor, debug, test, docs, review, explore |
| **Complexity** | Points-based scoring from session metrics | trivial, simple, moderate, complex |
| **Outcome** | Inferred from last assistant message + error rate | success, failure, unknown |
| **Domain** | Inferred from file extensions touched | web, backend, data, devops, docs, mixed, unknown |
| **Error type** | Classified from error message text (on `fact_errors`) | permission_denied, file_not_found, syntax_error, timeout, import_error, tool_error |

```sql
-- What kinds of sessions do I have?
SELECT intent, complexity, COUNT(*) as sessions
FROM dim_session GROUP BY intent, complexity ORDER BY sessions DESC;

-- Tool usage by category
SELECT dt.tool_category, COUNT(*) as uses
FROM fact_tool_calls ftc
JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
GROUP BY dt.tool_category ORDER BY uses DESC;

-- Am I more productive mornings or evenings?
SELECT dti.time_of_day, COUNT(*) as sessions, AVG(fss.total_messages) as avg_msgs
FROM fact_session_summary fss
JOIN dim_time dti ON fss.time_key = dti.time_key
GROUP BY dti.time_of_day;

-- Most-touched files across all sessions
SELECT df.file_path, SUM(bsf.write_count + bsf.edit_count) as modifications
FROM bridge_session_file bsf
JOIN dim_file df ON bsf.file_key = df.file_key
GROUP BY df.file_path ORDER BY modifications DESC LIMIT 20;

-- Catch up on a project (what was worked on recently)
SELECT first_user_message, last_assistant_message, intent, created_at
FROM semantic_project_context
WHERE project_name = 'my-project' LIMIT 5;
```

### JSON Export

```bash
# Simple schema - single file
ccutils --format json -o ./sessions.json

# Star schema - directory structure (dimensions/ + facts/ + meta.json)
ccutils --format json-star -o ./star-export/
```

## Common Options

```bash
# Output
-o, --output PATH          Output directory or file
--format FORMAT            html, duckdb, duckdb-star, json, json-star (+ both for all)
--open                     Open result in browser

# Content (included by default -- use flags to exclude)
--no-thinking              Exclude thinking blocks
--no-subagents             Exclude related agent sessions (local)
--no-agents                Exclude agent-* session files (all)
--private                  Sanitize file paths for sharing

# Selection
--flat                     Flat single-list mode (local)
--expand-chains            Show individual sessions in resumed chains (local)
-p, --project TEXT         Filter by project name (local, all)
--dry-run                  Preview without converting (all)

# Embeddings (local and all, star schema only)
--embed [MODEL]            Run ColBERT embeddings (optionally specify model)

# Batch processing (all command)
-s, --source PATH          Source directory (default: ~/.claude/projects)
-j, --jobs N               Parallel workers (default: 1)
--batch-size N             Sessions per batch (default: 10)
-q, --quiet                Suppress output except errors
--no-search-index          Skip search index generation
```

## Documentation

- [Star Schema Reference](docs/STAR_SCHEMA.md) -- table definitions, ETL capabilities, heuristic classification, example queries

## Development

```bash
uv run pytest              # Run tests (~803 passing)
uv run ccutils --help      # Run development version
uv run pytest --cov=ccutils  # Coverage
```

## License

Apache-2.0
