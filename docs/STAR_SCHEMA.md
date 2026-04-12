# Star Schema DuckDB Implementation

Last updated: 2026-04-11

A dimensional data model for Claude Code transcript analytics with 27 tables and 13 views. Designed for questions the simple 4-table schema cannot answer: intent classification, tool duration analysis, cross-session file tracking, time-of-day patterns, tool sequence analysis, project context recovery, token cost analysis, and semantic similarity clustering.

## Quick Start (CLI)

```bash
# Generate star schema DuckDB from local sessions
ccutils local --format duckdb-star -o ./analytics

# Or export to JSON directory structure
ccutils local --format json-star -o ./star-export/

# Or generate from all sessions
ccutils all --format duckdb-star -o ./analytics

# With ColBERT embeddings (requires pylate)
ccutils local --format duckdb-star --embed -o ./analytics

# Launch the visual Data Explorer
ccutils explore ./analytics/archive.duckdb

# Query directly with DuckDB CLI
duckdb ./analytics/archive.duckdb
```

Once in DuckDB:

```sql
-- What kinds of sessions do I have?
SELECT intent, complexity, COUNT(*) as sessions
FROM dim_session
GROUP BY intent, complexity ORDER BY sessions DESC;

-- Which tools are slowest?
SELECT dt.tool_name, AVG(ftc.duration_seconds) as avg_sec
FROM fact_tool_calls ftc
JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
WHERE ftc.duration_seconds IS NOT NULL
GROUP BY dt.tool_name ORDER BY avg_sec DESC;

-- Most-touched files across all sessions
SELECT df.file_path, SUM(bsf.write_count + bsf.edit_count) as modifications
FROM bridge_session_file bsf
JOIN dim_file df ON bsf.file_key = df.file_key
GROUP BY df.file_path ORDER BY modifications DESC LIMIT 20;

-- Am I more productive mornings or evenings?
SELECT dt.time_of_day, COUNT(DISTINCT fm.session_key) as sessions,
       COUNT(*) as messages
FROM fact_messages fm
JOIN dim_time dt ON fm.time_key = dt.time_key
GROUP BY dt.time_of_day;
```

See [DATA_EXPLORER.md](DATA_EXPLORER.md) for the visual interface.

---

## Overview

The star schema follows dimensional modeling best practices:

- **Dimension tables** contain descriptive attributes (who, what, when, where)
- **Fact tables** contain measurable events and metrics
- **Hash-based surrogate keys** link facts to dimensions (no hard PK/FK constraints)
- **Integer keys** for date and time dimensions (efficient joins)
- **Degenerate dimensions** for low-cardinality categoricals (message_type, block_type, error_type, entity_type, language) -- stored directly on fact tables instead of in separate lookup tables
- **Heuristic classification** runs during ETL with zero external dependencies (no LLM, no API key)

## Schema Summary

| Category | Count | Tables |
|----------|-------|--------|
| Core dimensions | 7 | dim_session, dim_project, dim_tool, dim_model, dim_date, dim_time, dim_prompt |
| Core facts | 6 | fact_messages, fact_tool_calls, fact_session_summary, fact_file_operations, fact_errors, fact_tool_chain_steps |
| Telemetry facts | 4 | fact_token_usage, fact_turn_durations, fact_diagnostics, fact_stop_events |
| Granular dimensions | 2 | dim_file, dim_session_chain |
| Granular facts | 3 | fact_content_blocks, fact_code_blocks, fact_entity_mentions |
| Agent/bridge/staging | 3 | fact_agent_delegations, bridge_session_file, stg_task_agent_map |
| Optional | 2 | fact_session_embeddings (requires pylate), fact_tool_input_params |
| **Total** | **27 tables** | + 13 semantic views |

## ETL Pipeline Order

```
create_star_schema(conn)                    # DDL: create all 27 tables + 13 views
run_star_schema_etl(conn, ...)              # Per-session ETL (call once per session)
finalize_star_schema(conn, history_path=..) # Post-ETL: chains, delegations, file bridge, depths, history
create_semantic_model(conn)                 # Semantic views metadata for Data Explorer
# Optional: EmbeddingPipeline(conn).embed_sessions(conn)
```

Key details:
- `run_star_schema_etl` reads `.meta.json` sidecar for agent_type/agent_description automatically
- `finalize_star_schema` accepts optional `history_path` to load `~/.claude/history.jsonl` into `dim_prompt`

`finalize_star_schema()` must be called after all sessions are loaded. It populates cross-session tables: `dim_session_chain`, `fact_agent_delegations`, `bridge_session_file`, and `dim_session.depth_level`. Both `local` and `all` commands call it automatically.

---

## Tables

### Core Dimensions

#### dim_session
One row per Claude Code session.

| Column | Type | Description |
|--------|------|-------------|
| session_key | VARCHAR | PK (hash of session_id) |
| session_id | VARCHAR | Original session UUID |
| project_key | VARCHAR | FK to dim_project |
| cwd | VARCHAR | Working directory |
| git_branch | VARCHAR | Git branch at session start |
| version | VARCHAR | Claude Code version |
| slug | VARCHAR | Session slug for chain grouping |
| first_timestamp | TIMESTAMP | First message time |
| last_timestamp | TIMESTAMP | Last message time |
| is_agent | BOOLEAN | Whether this is an agent session |
| agent_id | VARCHAR | Agent identifier |
| parent_session_key | VARCHAR | FK to dim_session (parent) |
| depth_level | INTEGER | Nesting depth (0=root, populated by finalize) |
| chain_key | VARCHAR | FK to dim_session_chain (populated by finalize) |
| intent | VARCHAR | Heuristic: bug_fix, feature, refactor, debug, test, docs, review, explore |
| complexity | VARCHAR | Heuristic: trivial, simple, moderate, complex |
| outcome | VARCHAR | Heuristic: success, failure, unknown |
| domain | VARCHAR | Heuristic: web, backend, data, devops, docs, mixed, unknown |
| first_user_message | VARCHAR | First user message text (truncated to 500 chars) |
| last_assistant_message | VARCHAR | Last assistant message text (truncated to 500 chars) |

#### dim_project
One row per project (working directory).

| Column | Type | Description |
|--------|------|-------------|
| project_key | VARCHAR | PK (hash of project_path) |
| project_path | VARCHAR | Full path |
| project_name | VARCHAR | Directory name |

#### dim_tool
One row per tool name.

| Column | Type | Description |
|--------|------|-------------|
| tool_key | VARCHAR | PK (hash of tool_name) |
| tool_name | VARCHAR | Tool name (e.g., "Bash", "Read") |
| tool_category | VARCHAR | Category from TOOL_CATEGORIES lookup |

#### dim_model
One row per model string.

| Column | Type | Description |
|--------|------|-------------|
| model_key | VARCHAR | PK (hash of model_name) |
| model_name | VARCHAR | Full model identifier |
| model_family | VARCHAR | Family: opus, sonnet, haiku |

#### dim_date
One row per calendar date.

| Column | Type | Description |
|--------|------|-------------|
| date_key | INTEGER | PK (YYYYMMDD integer) |
| full_date | DATE | Calendar date |
| year | INTEGER | Year |
| month | INTEGER | Month (1-12) |
| day | INTEGER | Day of month |
| day_of_week | INTEGER | Day of week (0-6) |
| day_name | VARCHAR | Day name (Monday, etc.) |
| month_name | VARCHAR | Month name (January, etc.) |
| quarter | INTEGER | Quarter (1-4) |
| is_weekend | BOOLEAN | Weekend flag |
| week_of_year | INTEGER | ISO week number |

#### dim_time
One row per hour:minute.

| Column | Type | Description |
|--------|------|-------------|
| time_key | INTEGER | PK (HHMM integer) |
| hour | INTEGER | Hour (0-23) |
| minute | INTEGER | Minute (0-59) |
| time_of_day | VARCHAR | morning, afternoon, evening, night |

### Core Facts

#### fact_messages
One row per message.

| Column | Type | Description |
|--------|------|-------------|
| message_id | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| project_key | VARCHAR | FK to dim_project |
| message_type | VARCHAR | Degenerate: user, assistant, system |
| model_key | VARCHAR | FK to dim_model |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| parent_message_id | VARCHAR | Parent message UUID |
| timestamp | TIMESTAMP | Message timestamp |
| content_length | INTEGER | Character count |
| content_block_count | INTEGER | Number of content blocks |
| has_tool_use | BOOLEAN | Contains tool calls |
| has_tool_result | BOOLEAN | Contains tool results |
| has_thinking | BOOLEAN | Contains thinking blocks |
| word_count | INTEGER | Word count |
| estimated_tokens | INTEGER | Estimated token count |
| response_time_seconds | FLOAT | Time since previous message |
| conversation_depth | INTEGER | Turn count at this point |
| content_text | TEXT | Message text content |
| content_json | JSON | Raw content block JSON |
| is_sidechain | BOOLEAN | Part of agent sidechain |

#### fact_tool_calls
One row per tool invocation.

| Column | Type | Description |
|--------|------|-------------|
| tool_call_id | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| tool_key | VARCHAR | FK to dim_tool |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| invoke_message_id | VARCHAR | Message containing the tool_use block |
| result_message_id | VARCHAR | Message containing the tool_result block |
| timestamp | TIMESTAMP | Invocation timestamp |
| input_char_count | INTEGER | Input character count |
| output_char_count | INTEGER | Output character count |
| is_error | BOOLEAN | Whether the call errored |
| duration_seconds | FLOAT | Time between invoke and result |
| input_json | JSON | Raw input JSON |
| input_summary | TEXT | Summarized input (file path, command, etc.) |
| output_text | TEXT | Result text (truncated) |
| file_path | VARCHAR | Extracted file path (if applicable) |
| command | VARCHAR | Extracted command (if Bash) |
| pattern | VARCHAR | Extracted pattern (if Grep/Glob) |
| query_text | VARCHAR | Extracted query (if applicable) |

#### fact_session_summary
One row per session with pre-aggregated metrics.

| Column | Type | Description |
|--------|------|-------------|
| session_key | VARCHAR | PK, FK to dim_session |
| project_key | VARCHAR | FK to dim_project |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| total_messages | INTEGER | Total message count |
| user_messages | INTEGER | User message count |
| assistant_messages | INTEGER | Assistant message count |
| total_tool_calls | INTEGER | Tool call count |
| total_thinking_blocks | INTEGER | Thinking block count |
| total_content_blocks | INTEGER | Content block count |
| total_errors | INTEGER | Error count |
| unique_tools_used | INTEGER | Distinct tools used |
| unique_files_touched | INTEGER | Distinct files touched |
| max_conversation_depth | INTEGER | Maximum conversation depth |
| total_estimated_tokens | INTEGER | Estimated total tokens (text + thinking + tool I/O) |
| total_thinking_tokens | INTEGER | Estimated tokens from thinking blocks |
| total_tool_io_tokens | INTEGER | Estimated tokens from tool input/output |
| session_duration_seconds | INTEGER | Session duration |
| first_timestamp | TIMESTAMP | First message time |
| last_timestamp | TIMESTAMP | Last message time |
| total_estimated_tokens_incl_agents | INTEGER | Estimated tokens including all descendant agents |
| total_tool_calls_incl_agents | INTEGER | Tool calls including all descendant agents |
| total_errors_incl_agents | INTEGER | Errors including all descendant agents |
| total_duration_incl_agents | INTEGER | Duration including all descendant agents |

The `_incl_agents` columns are populated during `finalize_star_schema()` via bottom-up rollup from deepest agents to root. For sessions with no agents, these equal the base columns.

#### fact_file_operations
One row per file touch.

| Column | Type | Description |
|--------|------|-------------|
| file_operation_id | VARCHAR | PK |
| tool_call_id | VARCHAR | FK to fact_tool_calls |
| session_key | VARCHAR | FK to dim_session |
| file_key | VARCHAR | FK to dim_file |
| tool_key | VARCHAR | FK to dim_tool |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| operation_type | VARCHAR | Degenerate: read, write, edit |
| file_size_chars | INTEGER | Characters in file content |
| timestamp | TIMESTAMP | Operation timestamp |

#### fact_errors
One row per error occurrence.

| Column | Type | Description |
|--------|------|-------------|
| error_id | VARCHAR | PK |
| tool_call_id | VARCHAR | FK to fact_tool_calls |
| session_key | VARCHAR | FK to dim_session |
| tool_key | VARCHAR | FK to dim_tool |
| error_type | VARCHAR | Heuristic: permission_denied, file_not_found, syntax_error, timeout, import_error, tool_error |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| error_message | TEXT | Error text |
| timestamp | TIMESTAMP | Error timestamp |

#### fact_tool_chain_steps
One row per step in a tool invocation chain.

| Column | Type | Description |
|--------|------|-------------|
| chain_step_id | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| chain_id | VARCHAR | Groups steps in the same chain |
| tool_call_id | VARCHAR | FK to fact_tool_calls |
| tool_key | VARCHAR | FK to dim_tool |
| step_position | INTEGER | Position in chain (0-based) |
| prev_tool_key | VARCHAR | FK to dim_tool for previous step |
| next_tool_key | VARCHAR | FK to dim_tool for next step |
| is_error | BOOLEAN | Whether this step errored |
| time_since_prev_seconds | FLOAT | Seconds since previous step |

### Granular Dimensions

#### dim_file
One row per file path.

| Column | Type | Description |
|--------|------|-------------|
| file_key | VARCHAR | PK (hash of file_path) |
| file_path | VARCHAR | Full file path |
| file_name | VARCHAR | File name only |
| file_extension | VARCHAR | Extension (e.g., ".py") |
| directory_path | VARCHAR | Parent directory |
| language | VARCHAR | Inferred programming language |

#### dim_session_chain
One row per chain (group of resumed sessions sharing a slug). Populated by `finalize_star_schema()`.

| Column | Type | Description |
|--------|------|-------------|
| chain_key | VARCHAR | PK (hash of slug) |
| slug | VARCHAR | Shared slug |
| project_key | VARCHAR | FK to dim_project |
| first_session_key | VARCHAR | Earliest session in chain |
| last_session_key | VARCHAR | Latest session in chain |
| session_count | INTEGER | Number of sessions in chain |
| first_timestamp | TIMESTAMP | Earliest session start |
| last_timestamp | TIMESTAMP | Latest session end |
| total_duration_seconds | INTEGER | Chain total duration |

### Granular Facts

#### fact_content_blocks
One row per content block within a message.

| Column | Type | Description |
|--------|------|-------------|
| content_block_id | VARCHAR | PK |
| message_id | VARCHAR | FK to fact_messages |
| session_key | VARCHAR | FK to dim_session |
| block_type | VARCHAR | Degenerate: text, tool_use, tool_result, thinking, image |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| block_index | INTEGER | Position within message |
| content_length | INTEGER | Character count |
| content_text | TEXT | Block text content |
| content_json | JSON | Raw block JSON |

#### fact_code_blocks
One row per code block extracted from messages.

| Column | Type | Description |
|--------|------|-------------|
| code_block_id | VARCHAR | PK |
| message_id | VARCHAR | FK to fact_messages |
| session_key | VARCHAR | FK to dim_session |
| language | VARCHAR | Degenerate: python, javascript, etc. |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| block_index | INTEGER | Position within message |
| line_count | INTEGER | Number of lines |
| char_count | INTEGER | Character count |
| code_text | TEXT | Code content |

#### fact_entity_mentions
One row per entity mention (file, function, class, etc.).

| Column | Type | Description |
|--------|------|-------------|
| mention_id | VARCHAR | PK |
| message_id | VARCHAR | FK to fact_messages |
| session_key | VARCHAR | FK to dim_session |
| entity_type | VARCHAR | Degenerate: file, function, class, variable, module, url, error, command |
| entity_text | VARCHAR | Raw entity text |
| entity_normalized | VARCHAR | Normalized form |
| context_snippet | TEXT | Surrounding text for context |
| position_start | INTEGER | Start position in message |
| position_end | INTEGER | End position in message |

### Agent/Bridge/Staging

#### fact_agent_delegations
One row per agent delegation (Task tool invocation). Populated by `finalize_star_schema()`.

| Column | Type | Description |
|--------|------|-------------|
| delegation_key | VARCHAR | PK |
| parent_session_key | VARCHAR | FK to dim_session (parent) |
| agent_session_key | VARCHAR | FK to dim_session (agent) |
| task_tool_call_id | VARCHAR | FK to fact_tool_calls |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| task_description | TEXT | Task description from tool input |
| task_prompt | TEXT | Task prompt text |
| subagent_type | VARCHAR | Agent type (e.g., "Explore", "Plan") |
| agent_output | TEXT | Agent result text |
| completion_status | VARCHAR | Status of delegation |
| delegation_timestamp | TIMESTAMP | When agent was invoked |
| completion_timestamp | TIMESTAMP | When agent completed |
| match_confidence | FLOAT | Link confidence (1.0=deterministic, 0.5-0.8=heuristic) |
| agent_tool_calls | INTEGER | Denormalized: tool calls in agent session |
| agent_errors | INTEGER | Denormalized: errors in agent session |
| agent_duration_seconds | INTEGER | Denormalized: agent session duration |
| agent_estimated_tokens | INTEGER | Denormalized: estimated tokens in agent session |

#### bridge_session_file
One row per (session, file) pair for cross-session file tracking. Populated by `finalize_star_schema()`.

| Column | Type | Description |
|--------|------|-------------|
| session_file_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| file_key | VARCHAR | FK to dim_file |
| first_operation_timestamp | TIMESTAMP | Earliest operation on this file |
| last_operation_timestamp | TIMESTAMP | Latest operation on this file |
| operation_count | INTEGER | Total operations |
| read_count | INTEGER | Read operations |
| write_count | INTEGER | Write operations |
| edit_count | INTEGER | Edit operations |
| total_chars_written | INTEGER | Total characters written |

#### stg_task_agent_map
Staging table for deterministic agent delegation linking from progress records.

| Column | Type | Description |
|--------|------|-------------|
| tool_use_id | VARCHAR | Task tool_use_id |
| agent_id | VARCHAR | Agent identifier |
| session_key | VARCHAR | FK to dim_session (parent) |

### Telemetry Facts

#### fact_token_usage
Per-API-response actual token counts from Claude's usage data. One row per assistant message with usage.

| Column | Type | Description |
|--------|------|-------------|
| usage_id | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| model_key | VARCHAR | FK to dim_model |
| input_tokens | INTEGER | API-reported input tokens |
| output_tokens | INTEGER | API-reported output tokens |
| cache_creation_input_tokens | INTEGER | Tokens written to prompt cache |
| cache_read_input_tokens | INTEGER | Tokens read from prompt cache |
| cache_ephemeral_1h_tokens | INTEGER | 1-hour ephemeral cache tokens |
| cache_ephemeral_5m_tokens | INTEGER | 5-minute ephemeral cache tokens |
| service_tier | VARCHAR | "standard", etc. |
| speed | VARCHAR | "standard", etc. |
| timestamp | TIMESTAMP | When the response was generated |

#### fact_turn_durations
Actual turn processing time from system.turn_duration entries.

| Column | Type | Description |
|--------|------|-------------|
| turn_id | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| duration_ms | INTEGER | Turn duration in milliseconds |
| message_count | INTEGER | Messages in the turn |
| timestamp | TIMESTAMP | When the turn ended |

#### fact_diagnostics
LSP diagnostics (code errors/warnings) reported during sessions.

| Column | Type | Description |
|--------|------|-------------|
| diagnostic_id | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| file_key | VARCHAR | FK to dim_file |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| severity | VARCHAR | Error, Warning, Info, Hint |
| source | VARCHAR | LSP source (Pyright, typescript, etc.) |
| code | VARCHAR | Diagnostic code |
| message | TEXT | Diagnostic message |
| range_start_line | INTEGER | Start line |
| range_start_col | INTEGER | Start column |
| range_end_line | INTEGER | End line |
| range_end_col | INTEGER | End column |
| timestamp | TIMESTAMP | When diagnostic was reported |

#### fact_stop_events
Stop/turn-end events with hook execution details.

| Column | Type | Description |
|--------|------|-------------|
| stop_event_id | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| stop_reason | VARCHAR | Why the turn stopped |
| hook_count | INTEGER | Number of hooks executed |
| has_output | BOOLEAN | Whether hooks produced output |
| prevented_continuation | BOOLEAN | Whether hooks blocked continuation |
| hook_total_duration_ms | INTEGER | Total hook execution time |
| hook_error_count | INTEGER | Number of hook errors |
| timestamp | TIMESTAMP | When the stop occurred |

#### dim_prompt
User prompts from `~/.claude/history.jsonl`. Links to sessions via sessionId.

| Column | Type | Description |
|--------|------|-------------|
| prompt_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session (NULL if session not imported) |
| project_path | VARCHAR | Project directory path |
| project_name | VARCHAR | Project name |
| display_text | TEXT | The user's prompt text |
| timestamp | TIMESTAMP | When the prompt was submitted |
| date_key | INTEGER | FK to dim_date |
| time_key | INTEGER | FK to dim_time |
| has_pasted_content | BOOLEAN | Whether prompt included pasted content |

### Optional

#### fact_session_embeddings
ColBERT embeddings for semantic similarity. Requires `uv add ccutils[colbert]`.

| Column | Type | Description |
|--------|------|-------------|
| embedding_key | VARCHAR | PK |
| session_key | VARCHAR | FK to dim_session |
| content_type | VARCHAR | What was embedded (default: first_user_message) |
| embedding_model | VARCHAR | Model used |
| embedding_dim | INTEGER | Embedding dimensions (64) |
| mean_embedding | FLOAT[64] | Mean-pooled embedding vector |
| embedded_at | TIMESTAMP | When embedding was computed |
| content_hash | VARCHAR | MD5 of embedded content |

#### fact_tool_input_params
Key-value extraction of tool input parameters.

| Column | Type | Description |
|--------|------|-------------|
| param_id | VARCHAR | PK |
| tool_call_id | VARCHAR | FK to fact_tool_calls |
| session_key | VARCHAR | FK to dim_session |
| param_key | VARCHAR | Parameter name |
| param_value_text | VARCHAR | Text value |
| param_value_number | FLOAT | Numeric value |
| param_value_bool | BOOLEAN | Boolean value |

---

## Semantic Views

All views use the `semantic_` prefix and join facts with dimensions for easy querying. These are the fastest way to get started -- no manual JOINs required.

### semantic_sessions
Sessions enriched with project info, summary metrics, heuristic classifications, and date/time.

```sql
-- Complex bug fixes in a specific project
SELECT session_id, intent, complexity, total_tool_calls, total_errors,
       session_duration_seconds
FROM semantic_sessions
WHERE intent = 'bug_fix' AND complexity = 'complex'
ORDER BY total_errors DESC;

-- Recent sessions sorted by datetime
SELECT session_id, project_name, full_date, time_of_day,
       session_datetime, total_messages
FROM semantic_sessions
ORDER BY session_datetime DESC LIMIT 20;

-- Weekend warrior: sessions by day of week
SELECT day_name, COUNT(*) as sessions, AVG(total_messages) as avg_msgs
FROM semantic_sessions
GROUP BY day_name ORDER BY sessions DESC;
```

### semantic_messages
Messages with session, model, and time context.

```sql
-- Late-night opus sessions
SELECT session_id, content_text, response_time_seconds
FROM semantic_messages
WHERE model_family = 'opus' AND time_of_day = 'night'
ORDER BY response_time_seconds DESC LIMIT 20;

-- Message volume by model family
SELECT model_family, COUNT(*) as messages, AVG(word_count) as avg_words
FROM semantic_messages
WHERE message_type = 'assistant'
GROUP BY model_family;
```

### semantic_tool_calls
Tool calls with tool info, session context, and duration.

```sql
-- Slowest tool calls
SELECT tool_name, tool_category, duration_seconds, file_path, command
FROM semantic_tool_calls
WHERE duration_seconds > 5
ORDER BY duration_seconds DESC LIMIT 20;

-- Error rate by tool category
SELECT tool_category,
       COUNT(*) as calls,
       SUM(CASE WHEN is_error THEN 1 ELSE 0 END) as errors,
       ROUND(100.0 * SUM(CASE WHEN is_error THEN 1 ELSE 0 END) / COUNT(*), 1) as error_pct
FROM semantic_tool_calls
GROUP BY tool_category ORDER BY error_pct DESC;
```

### semantic_file_operations
File operations with file info, tool, and session context.

```sql
-- Most-edited Python files
SELECT file_path, COUNT(*) as operations,
       SUM(file_size_chars) as total_chars
FROM semantic_file_operations
WHERE language = 'python' AND operation_type = 'edit'
GROUP BY file_path ORDER BY operations DESC LIMIT 20;
```

### semantic_session_chains
Session chains with aggregate metrics across all member sessions.

```sql
-- Long-running chains (3+ resumed sessions)
SELECT slug, session_count, total_duration_seconds,
       SUM(total_messages) as chain_messages, SUM(total_tool_calls) as chain_tools
FROM semantic_session_chains
GROUP BY slug, session_count, total_duration_seconds
HAVING session_count > 2
ORDER BY session_count DESC;
```

### semantic_agent_delegations
Agent delegations with parent/agent session details and denormalized metrics.

```sql
-- Agent sessions that had errors
SELECT parent_session_id, agent_session_id, subagent_type,
       task_description, agent_tool_calls, agent_errors
FROM semantic_agent_delegations
WHERE agent_errors > 0
ORDER BY agent_errors DESC;
```

### semantic_file_evolution
Cross-session file activity aggregation. Only includes files touched in 2+ sessions.

```sql
-- Files with the most cross-session activity
SELECT file_path, language, session_count,
       total_operations, total_writes, total_edits
FROM semantic_file_evolution
ORDER BY session_count DESC LIMIT 20;
```

### semantic_tool_patterns
Common tool sequences with frequency and error rates. Only includes patterns seen 2+ times.

```sql
-- Tool sequences most likely to produce errors
SELECT tool_name, next_tool_name, frequency,
       error_count, error_rate_pct, avg_time_between
FROM semantic_tool_patterns
WHERE error_rate_pct > 20
ORDER BY frequency DESC;

-- Most common tool pairs
SELECT tool_name || ' -> ' || next_tool_name as pattern,
       frequency, avg_time_between
FROM semantic_tool_patterns
ORDER BY frequency DESC LIMIT 15;
```

### semantic_project_context
Sessions enriched with project info, first/last messages, summary metrics, and date/time. Designed for catching up on a project -- what sessions happened, what was worked on, what the user asked for.

```sql
-- Get up to speed on a project (most recent sessions first)
SELECT session_id, project_name, session_date, time_of_day,
       intent, complexity, outcome,
       first_user_message, last_assistant_message,
       total_messages, total_tool_calls, total_errors
FROM semantic_project_context
WHERE project_name = 'my-project'
LIMIT 10;

-- What was the last thing worked on?
SELECT project_name, first_user_message, last_assistant_message,
       created_at, intent
FROM semantic_project_context
LIMIT 1;

-- Filter by date
SELECT session_id, first_user_message, total_messages
FROM semantic_project_context
WHERE session_date = '2025-01-20'
ORDER BY created_at DESC;
```

### semantic_project_files
File activity aggregated by project. Shows which files matter most and when they were last touched. Requires `finalize_star_schema()` to populate `bridge_session_file`.

```sql
-- Most important files in a project (by session count)
SELECT file_path, language, sessions_touching_file,
       total_reads, total_writes, total_edits, last_touched
FROM semantic_project_files
WHERE project_name = 'my-project'
ORDER BY sessions_touching_file DESC LIMIT 20;

-- Recently touched files across all projects
SELECT project_name, file_path, last_touched
FROM semantic_project_files
ORDER BY last_touched DESC LIMIT 20;
```

### semantic_token_usage

Per-API-response token data joined with model, session, and project context.

```sql
-- Token usage by model
SELECT model_name, SUM(input_tokens) as total_in, SUM(output_tokens) as total_out,
       SUM(cache_read_input_tokens) as total_cache_read
FROM semantic_token_usage
GROUP BY model_name;
```

### semantic_cost_analysis

Session-level cost aggregation with cache hit rate.

```sql
-- Sessions with highest token usage
SELECT session_id, project_name, custom_title,
       actual_input_tokens, actual_output_tokens,
       cache_hit_rate_pct, total_turn_duration_ms
FROM semantic_cost_analysis
ORDER BY actual_output_tokens DESC LIMIT 10;

-- Cache efficiency by project
SELECT project_name, COUNT(*) as sessions,
       AVG(cache_hit_rate_pct) as avg_cache_hit_pct
FROM semantic_cost_analysis
WHERE actual_input_tokens IS NOT NULL
GROUP BY project_name ORDER BY avg_cache_hit_pct DESC;
```

### semantic_prompt_history

User prompts from history.jsonl linked to session metadata.

```sql
-- Recent prompts with session outcomes
SELECT display_text, project_name, intent, complexity, full_date
FROM semantic_prompt_history
ORDER BY timestamp DESC LIMIT 20;

-- Most active projects by prompt count
SELECT project_name, COUNT(*) as prompts
FROM semantic_prompt_history
WHERE project_name IS NOT NULL
GROUP BY project_name ORDER BY prompts DESC;
```

---

## Example Queries (Raw Tables)

These use direct table joins rather than semantic views, useful when you need columns the views don't expose.

### Session Analysis

```sql
-- Sessions ranked by cost (estimated tokens)
SELECT ds.session_id, dp.project_name, ds.intent, ds.complexity,
       fss.total_estimated_tokens, fss.total_tool_calls,
       fss.session_duration_seconds
FROM fact_session_summary fss
JOIN dim_session ds ON fss.session_key = ds.session_key
JOIN dim_project dp ON fss.project_key = dp.project_key
ORDER BY fss.total_estimated_tokens DESC LIMIT 20;

-- Session complexity distribution by project
SELECT dp.project_name, ds.complexity, COUNT(*) as sessions
FROM dim_session ds
JOIN dim_project dp ON ds.project_key = dp.project_key
GROUP BY dp.project_name, ds.complexity
ORDER BY dp.project_name, sessions DESC;

-- Average session duration by intent
SELECT ds.intent, COUNT(*) as sessions,
       AVG(fss.session_duration_seconds) as avg_duration_sec,
       AVG(fss.total_tool_calls) as avg_tools,
       AVG(fss.total_errors) as avg_errors
FROM fact_session_summary fss
JOIN dim_session ds ON fss.session_key = ds.session_key
GROUP BY ds.intent ORDER BY avg_duration_sec DESC;
```

### Tool Performance

```sql
-- Tool duration percentiles
SELECT dt.tool_name,
       COUNT(*) as calls,
       ROUND(AVG(ftc.duration_seconds), 2) as avg_sec,
       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ftc.duration_seconds), 2) as p50,
       ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ftc.duration_seconds), 2) as p95
FROM fact_tool_calls ftc
JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
WHERE ftc.duration_seconds IS NOT NULL
GROUP BY dt.tool_name
HAVING COUNT(*) >= 5
ORDER BY avg_sec DESC;

-- What commands are run most via Bash?
SELECT ftc.command, COUNT(*) as uses,
       SUM(CASE WHEN ftc.is_error THEN 1 ELSE 0 END) as errors
FROM fact_tool_calls ftc
JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
WHERE dt.tool_name = 'Bash' AND ftc.command IS NOT NULL
GROUP BY ftc.command ORDER BY uses DESC LIMIT 20;
```

### Time Patterns

```sql
-- Activity heatmap: hour vs day of week
SELECT dd.day_name, dt.hour, COUNT(*) as tool_calls
FROM fact_tool_calls ftc
JOIN dim_date dd ON ftc.date_key = dd.date_key
JOIN dim_time dt ON ftc.time_key = dt.time_key
GROUP BY dd.day_name, dt.hour
ORDER BY dd.day_name, dt.hour;

-- Weekend vs weekday productivity
SELECT dd.is_weekend,
       COUNT(DISTINCT fss.session_key) as sessions,
       AVG(fss.total_messages) as avg_msgs,
       AVG(fss.total_tool_calls) as avg_tools
FROM fact_session_summary fss
JOIN dim_date dd ON fss.date_key = dd.date_key
GROUP BY dd.is_weekend;
```

### Error Analysis

```sql
-- Error types by tool
SELECT dt.tool_name, fe.error_type, COUNT(*) as occurrences
FROM fact_errors fe
JOIN dim_tool dt ON fe.tool_key = dt.tool_key
GROUP BY dt.tool_name, fe.error_type
ORDER BY occurrences DESC LIMIT 20;

-- Sessions with highest error rates
SELECT ds.session_id, dp.project_name,
       fss.total_errors, fss.total_tool_calls,
       ROUND(100.0 * fss.total_errors / NULLIF(fss.total_tool_calls, 0), 1) as error_pct
FROM fact_session_summary fss
JOIN dim_session ds ON fss.session_key = ds.session_key
JOIN dim_project dp ON fss.project_key = dp.project_key
WHERE fss.total_tool_calls > 5
ORDER BY error_pct DESC LIMIT 20;
```

### File Tracking

```sql
-- Files modified across the most sessions
SELECT df.file_path, df.language,
       COUNT(DISTINCT bsf.session_key) as sessions_touched,
       SUM(bsf.edit_count) as total_edits,
       SUM(bsf.total_chars_written) as total_chars
FROM bridge_session_file bsf
JOIN dim_file df ON bsf.file_key = df.file_key
GROUP BY df.file_path, df.language
ORDER BY sessions_touched DESC LIMIT 20;

-- Most active directories
SELECT df.directory_path, COUNT(DISTINCT df.file_key) as files,
       COUNT(DISTINCT bsf.session_key) as sessions
FROM bridge_session_file bsf
JOIN dim_file df ON bsf.file_key = df.file_key
GROUP BY df.directory_path
ORDER BY files DESC LIMIT 15;
```

### Code Blocks

```sql
-- Languages used in code blocks
SELECT language, COUNT(*) as blocks,
       SUM(line_count) as total_lines, AVG(line_count) as avg_lines
FROM fact_code_blocks
WHERE language IS NOT NULL
GROUP BY language ORDER BY blocks DESC;

-- Largest code blocks written
SELECT fcb.language, fcb.line_count, fcb.char_count,
       ds.session_id, LEFT(fcb.code_text, 100) as preview
FROM fact_code_blocks fcb
JOIN dim_session ds ON fcb.session_key = ds.session_key
ORDER BY fcb.char_count DESC LIMIT 10;
```

### Agent Delegations

```sql
-- Agent type usage and performance
SELECT fad.subagent_type,
       COUNT(*) as delegations,
       AVG(fad.agent_tool_calls) as avg_tools,
       AVG(fad.agent_errors) as avg_errors,
       AVG(fad.agent_duration_seconds) as avg_duration
FROM fact_agent_delegations fad
GROUP BY fad.subagent_type ORDER BY delegations DESC;

-- Deepest agent nesting
SELECT ds.session_id, ds.depth_level, ds.agent_id,
       dp.project_name
FROM dim_session ds
JOIN dim_project dp ON ds.project_key = dp.project_key
WHERE ds.depth_level > 0
ORDER BY ds.depth_level DESC;
```

---

## Heuristic Classification

The star schema runs heuristic classification during ETL with zero external dependencies. Results are stored on `dim_session`.

### Intent
Classified from the first user message using score-based keyword matching. Each intent's keywords are checked against the message; the intent with the most keyword hits wins. On ties, priority order is used as tiebreaker. This correctly handles compound messages like "implement new error handling" (feature, not bug_fix).

| Intent | Triggers |
|--------|----------|
| bug_fix | fix, bug, broken, error, crash, wrong, issue, failing |
| debug | debug, investigate, why, trace, diagnose |
| test | test, spec, coverage |
| docs | doc, readme, comment, explain, documentation |
| review | review, check, audit, look at |
| feature | add, new, feature, implement, create, build |
| refactor | refactor, clean, reorganize, restructure, simplify |
| explore | (fallback) |

### Complexity
Scored from session metrics:

| Factor | Condition | Points |
|--------|-----------|--------|
| Tool calls | > 20 | +2 |
| Tool calls | > 2 | +1 |
| Messages | > 30 | +2 |
| Messages | > 8 | +1 |
| Agent depth | > 0 | +2 |
| Errors | > 3 | +1 |

| Score | Classification |
|-------|----------------|
| >= 5 | complex |
| >= 3 | moderate |
| >= 1 | simple |
| 0 | trivial |

### Outcome
Inferred from the last assistant message and error rate:

- **success**: Last message contains "done", "completed", "fixed", "finished", etc.
- **failure**: Last message contains "error", "failed", "couldn't", or error rate > 50%
- **unknown**: (fallback)

### Domain
Inferred from file extensions touched during the session:

| Domain | Extensions |
|--------|-----------|
| web | .tsx, .jsx, .css, .scss, .html, .vue, .svelte |
| backend | .py, .rs, .go, .java, .rb |
| data | .sql, .parquet, .csv |
| devops | .yaml, .yml, .tf, .dockerfile, .sh |
| docs | .md, .rst, .txt |
| mixed | (tie between domains) |
| unknown | (no files or unrecognized extensions) |

### Error Type
Classified from error message text on `fact_errors`:

| Type | Triggers |
|------|----------|
| permission_denied | "permission denied", "EACCES" |
| file_not_found | "not found", "ENOENT", "no such file" |
| syntax_error | "syntax error", "SyntaxError" |
| timeout | "timeout", "ETIMEDOUT" |
| import_error | "ImportError", "ModuleNotFoundError" |
| tool_error | (fallback) |

---

## What This Enables (vs Simple Schema)

| Question | Star Schema | Simple Schema |
|----------|-------------|---------------|
| What kinds of sessions do I have? | `SELECT intent, COUNT(*) FROM dim_session GROUP BY intent` | Not possible |
| Which tools are slowest? | `AVG(duration_seconds)` on fact_tool_calls | No duration tracking |
| Most-touched file across sessions? | bridge_session_file aggregation | Per-session only |
| Productive times of day? | dim_time join on fact_messages | No time dimension |
| Tool sequences that lead to errors? | fact_tool_chain_steps with next_tool_key + is_error | No chain tracking |
| How deep is agent nesting? | dim_session.depth_level | No depth tracking |
| What code was generated? | fact_code_blocks with language + line_count | No code extraction |
| Find similar sessions? | fact_session_embeddings with pylate | No embeddings |
| Catch up on a project? | semantic_project_context with first/last messages | Not possible |
| Which files matter most? | semantic_project_files with session counts | Per-session only |

---

## JSON Export Structure

When using `--format json-star`, the output is a directory:

```
output_dir/
  meta.json              # Schema metadata, version "2.0", relationships
  dimensions/            # One JSON file per dimension table
    dim_session.json
    dim_project.json
    ...
  facts/                 # One JSON file per fact table
    fact_messages.json
    fact_tool_calls.json
    ...
```

---

## Embedding Pipeline (Optional)

Requires the `colbert` extra: `uv add ccutils[colbert]`

The `--embed` flag runs a ColBERT pipeline that stores 64-dimensional mean-pooled vectors in `fact_session_embeddings`:

- `embed_sessions(conn)` -- embeds each session's first user message using `mxbai-edge-colbert-v0-32m`
- `match_delegations(conn)` -- re-scores agent delegation confidence using semantic similarity
- `cluster_sessions(conn)` -- KMeans clustering of sessions by embedding similarity

```python
from ccutils import EmbeddingPipeline

pipeline = EmbeddingPipeline()  # lazy model loading
pipeline.embed_sessions(conn)
pipeline.match_delegations(conn)
pipeline.cluster_sessions(conn)
```

CLI integration:

```bash
ccutils local --format duckdb-star --embed -o ./analytics
ccutils all --format duckdb-star --embed -o ./analytics
```

**Current status:** The vectors are stored and queryable via raw SQL but there is no built-in search interface or downstream query consumer. This is infrastructure for future semantic search (similar session lookup, project clustering). To use the embeddings today, query `fact_session_embeddings` directly.
