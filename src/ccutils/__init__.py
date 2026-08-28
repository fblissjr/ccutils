"""Convert Claude Code session JSON to a clean mobile-friendly HTML page with pagination."""

# Star schema imports (from modular package)
from .schemas.star import (
    create_star_schema,
    generate_dimension_key,
    get_model_family,
    get_time_of_day,
    get_tool_category,
    export_star_schema_to_json,
    EmbeddingPipeline,
    TOOL_CATEGORIES,
)

# v0.15 ETL orchestrator (replaces the legacy run_star_schema_etl)
from .etl.orchestrator import run_v15_etl

# Session parsing imports (from modular package)
from .parsers import (
    extract_searchable_content,
    extract_session_metadata,
    extract_session_slug,
    extract_snippet,
    extract_text_from_content,
    get_session_summary,
    parse_session_file,
    PROMPTS_PER_PAGE,
    # Session discovery
    find_agent_sessions,
    find_all_sessions,
    find_local_sessions,
    find_local_sessions_rich,
    flatten_selected_sessions,
    get_project_display_name,
    group_by_project,
    matches_project_filter,
    # Metadata extraction
    SessionMetadata,
    derive_project_name,
    extract_rich_metadata,
    format_duration,
    get_meaningful_summary,
    shorten_model_name,
)

# API client imports (from modular package)
from .api import (
    API_BASE_URL,
    ANTHROPIC_VERSION,
    CredentialsError,
    get_access_token_from_keychain,
    get_org_uuid_from_config,
    get_api_headers,
    fetch_sessions,
    fetch_session,
)

# Export format imports (from modular package)
from .export import (
    generate_duckdb_archive,
    # HTML generation
    CSS,
    JS,
    COMMIT_PATTERN,
    GITHUB_REPO_PATTERN,
    LONG_TEXT_THRESHOLD,
    _macros,  # Template macros for CLI functions
    analyze_conversation,
    detect_github_repo,
    detect_github_repo_from_cwd,
    format_json,
    format_tool_stats,
    generate_batch_html,
    generate_html,
    generate_multi_session_index,
    get_github_repo,
    get_template,
    is_json_like,
    is_tool_result_message,
    make_msg_id,
    render_assistant_message,
    render_bash_tool,
    render_content_block,
    render_edit_tool,
    render_markdown_text,
    render_message,
    render_todo_write,
    render_user_message_content,
    render_write_tool,
    set_github_repo,
)

# CLI imports (from modular package)
from .cli import (
    cli,
    main,
    convert_cmd,
    open_cmd,
    is_url,
    fetch_url_to_tempfile,
    resolve_credentials,
    format_session_for_display,
)
