"""Pydantic models for Claude Code session JSONL entries.

Discriminated union over the 12 distinct top-level entry types Claude Code
emits (per `internal/research/claude_code_jsonl_metadata.md`). All models
use `extra="allow"` so unknown fields land in `model_extra` rather than
raising -- the JSONL format is not a public contract and changes between
Claude Code versions.

Field names use snake_case in Python; JSON aliases (camelCase) are generated
automatically. Both `entry.session_id` and `EntryClass(sessionId="...")` work.

Sub-models (content blocks, attachment subtypes, system subtypes, progress
data variants, toolUseResult per tool) live in separate chunks (A2, A3) and
are wired in via `dict | str | None` placeholders here.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


# ---------------------------------------------------------------------------
# Base envelope
# ---------------------------------------------------------------------------


class _CamelBase(BaseModel):
    """Shared Pydantic config for every model in this module.

    extra="allow" preserves unknown fields (forward-compat with future
    Claude Code releases). The alias_generator handles snake_case Python /
    camelCase JSON; populate_by_name + validate_by_name + validate_by_alias
    let either form work at construction and validation time.

    Fields whose JSON keys preserve all-caps abbreviations (ID, UUID) need
    explicit Field(alias="...") because to_camel doesn't preserve them.
    """

    model_config = ConfigDict(
        extra="allow",
        alias_generator=to_camel,
        populate_by_name=True,
        validate_by_name=True,
        validate_by_alias=True,
    )


class _Envelope(_CamelBase):
    """Shared session/env fields present on most entries.

    Per the metadata reference, almost every entry carries: sessionId, uuid,
    parentUuid, timestamp, type, version, cwd, gitBranch, slug, userType,
    entrypoint, isSidechain, isMeta, agentId. These are optional on the meta
    entry types (custom-title, agent-name, permission-mode, etc.) which carry
    only their type-specific fields.
    """

    session_id: str | None = None
    uuid: str | None = None
    parent_uuid: str | None = None
    timestamp: str | None = None
    version: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    slug: str | None = None
    user_type: str | None = None
    entrypoint: str | None = None
    is_sidechain: bool = False
    is_meta: bool = False
    agent_id: str | None = None


# ---------------------------------------------------------------------------
# Entry-type models
# ---------------------------------------------------------------------------


class UserEntry(_Envelope):
    """User turn. Content can be human text, image paste, or tool_result."""

    type: Literal["user"]
    message: dict[str, Any]
    # toolUseResult is polymorphic per tool. Observed shapes:
    # - dict (most tools: ExitPlanMode, Read, Edit, Bash, ...)
    # - str (errors: "Error: <message>" plain string)
    # - list of content blocks (MCP tools, server-tool results: [{"type":"text","text":"..."}])
    tool_use_result: dict[str, Any] | list[Any] | str | None = None
    # Explicit aliases below: Claude Code preserves ID/UUID as all-caps in
    # these field names, which the to_camel generator doesn't handle.
    source_tool_use_id: str | None = Field(default=None, alias="sourceToolUseID")
    source_tool_assistant_uuid: str | None = Field(default=None, alias="sourceToolAssistantUUID")
    is_compact_summary: bool = False
    is_visible_in_transcript_only: bool | None = None
    permission_mode: str | None = None
    prompt_id: str | None = None
    plan_content: str | None = None
    thinking_metadata: dict[str, Any] | None = None
    image_paste_ids: list[int] | None = None
    origin: dict[str, Any] | None = None


class AssistantEntry(_Envelope):
    """Anthropic API response for one model turn, embedded verbatim in `message`."""

    type: Literal["assistant"]
    message: dict[str, Any]
    request_id: str | None = None
    api_error: str | None = None
    error: str | None = None
    is_api_error_message: bool = False


class ProgressEntry(_Envelope):
    """Streaming progress from a running tool or hook. Discriminated by data.type."""

    type: Literal["progress"]
    tool_use_id: str | None = Field(default=None, alias="toolUseID")
    parent_tool_use_id: str | None = Field(default=None, alias="parentToolUseID")
    data: dict[str, Any]


class SystemEntry(_Envelope):
    """Out-of-band events. Discriminated by `subtype`."""

    type: Literal["system"]
    subtype: str
    level: str | None = None


class AttachmentEntry(_Envelope):
    """Attached content delivered alongside a user message. Discriminated by attachment.type."""

    type: Literal["attachment"]
    attachment: dict[str, Any]


class PermissionModeEntry(_Envelope):
    """Session-level permission-mode change marker."""

    type: Literal["permission-mode"]
    permission_mode: str


class CustomTitleEntry(_Envelope):
    """Session custom title string."""

    type: Literal["custom-title"]
    custom_title: str


class AgentNameEntry(_Envelope):
    """Agent label/title (value identical to custom-title for agents)."""

    type: Literal["agent-name"]
    agent_name: str


class LastPromptEntry(_Envelope):
    """Mirror of the most recent user prompt text."""

    type: Literal["last-prompt"]
    last_prompt: str | None = None


class QueueOperationEntry(_Envelope):
    """User prompt queued/dequeued while Claude is mid-turn."""

    type: Literal["queue-operation"]
    operation: str | None = None
    content: str | None = None


class FileHistorySnapshotEntry(_Envelope):
    """Pre-edit file backup snapshot (restore buffer)."""

    type: Literal["file-history-snapshot"]
    message_id: str | None = None
    is_snapshot_update: bool | None = None
    snapshot: dict[str, Any] | None = None


class PrLinkEntry(_Envelope):
    """Created/associated GitHub pull request metadata."""

    type: Literal["pr-link"]
    pr_number: int | str | None = None
    pr_url: str | None = None
    pr_repository: str | None = None


class SummaryEntry(BaseModel):
    """Auto-summary entry written at session start (not a "real" log entry)."""

    model_config = ConfigDict(extra="allow", alias_generator=to_camel, populate_by_name=True)

    type: Literal["summary"]
    summary: str | None = None
    leaf_uuid: str | None = None


class UnknownEntry(_Envelope):
    """Fallback for entry types we don't yet recognize.

    Forward-compat: when Claude Code adds a new entry type (e.g. `ai-title`
    in newer Claude Code versions), the parser routes it here instead of
    raising. Inherits from _Envelope so the envelope fields still extract
    (session_id, uuid, timestamp, etc.) -- otherwise downstream ETL would
    drop these rows from session aggregations.

    The full raw dict is preserved in `model_extra` (via extra="allow").
    `type` defaults to "unknown" so even malformed entries without a
    `type` field land here.
    """

    type: str = "unknown"


# Discriminated union of all known entry models. The dispatch is done by
# the `parse_log_entry` function below rather than via Pydantic's
# `discriminator=` field, because we want the UnknownEntry fallback for
# forward-compat (Pydantic discriminated unions raise on unknown discriminator).
SessionLogEntry = (
    UserEntry
    | AssistantEntry
    | ProgressEntry
    | SystemEntry
    | AttachmentEntry
    | PermissionModeEntry
    | CustomTitleEntry
    | AgentNameEntry
    | LastPromptEntry
    | QueueOperationEntry
    | FileHistorySnapshotEntry
    | PrLinkEntry
    | SummaryEntry
    | UnknownEntry
)


_ENTRY_MODEL_BY_TYPE: dict[str, type[BaseModel]] = {
    "user": UserEntry,
    "assistant": AssistantEntry,
    "progress": ProgressEntry,
    "system": SystemEntry,
    "attachment": AttachmentEntry,
    "permission-mode": PermissionModeEntry,
    "custom-title": CustomTitleEntry,
    "agent-name": AgentNameEntry,
    "last-prompt": LastPromptEntry,
    "queue-operation": QueueOperationEntry,
    "file-history-snapshot": FileHistorySnapshotEntry,
    "pr-link": PrLinkEntry,
    "summary": SummaryEntry,
}


def parse_log_entry(raw: dict[str, Any]) -> SessionLogEntry:
    """Parse one raw JSONL entry dict into a typed model.

    Dispatches on `raw["type"]`. Unknown types route to `UnknownEntry`,
    which preserves the full payload in `model_extra` for forward-compat.
    """
    entry_type = raw.get("type")
    model_cls = _ENTRY_MODEL_BY_TYPE.get(entry_type or "", UnknownEntry)
    return model_cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Content block sub-models (inside message.content)
# ---------------------------------------------------------------------------


class _BlockBase(_CamelBase):
    pass


class TextBlock(_BlockBase):
    type: Literal["text"]
    text: str = ""


class ThinkingBlock(_BlockBase):
    type: Literal["thinking"]
    thinking: str = ""
    # Required for multi-turn continuity per extended-thinking API docs.
    # Opaque base64 signature; preserve verbatim, do not strip.
    signature: str | None = None


class RedactedThinkingBlock(_BlockBase):
    """Safety-redacted thinking. Carries opaque `data` string instead of `thinking` text."""

    type: Literal["redacted_thinking"]
    data: str = ""


class ToolUseBlock(_BlockBase):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    # caller.type distinguishes model-initiated ("direct") from server-initiated
    # (code_execution_20250825, etc.). Field may be absent on older sessions.
    caller: dict[str, Any] | None = None


class ToolResultBlock(_BlockBase):
    type: Literal["tool_result"]
    tool_use_id: str
    # content can be a string OR a list of blocks (text, image, search_result,
    # document, tool_reference per Messages API spec). Preserve both shapes.
    content: str | list[dict[str, Any]] = ""
    # Tri-state per Messages API: True / False / missing are all valid.
    # None = "model didn't assert an error state" (different from False).
    is_error: bool | None = None


class ImageBlock(_BlockBase):
    type: Literal["image"]
    source: dict[str, Any] = Field(default_factory=dict)  # {type: "base64", media_type, data}


class UnknownContentBlock(_BlockBase):
    """Forward-compat fallback for content block types we don't yet model.

    Covers server-tool siblings that may appear in newer Claude Code versions:
    server_tool_use, web_search_tool_result, web_fetch_tool_result,
    code_execution_tool_result, bash_code_execution_tool_result,
    text_editor_code_execution_tool_result, tool_search_tool_result,
    document, search_result, tool_reference, container_upload.
    """

    type: str = "unknown"


ContentBlock = (
    TextBlock
    | ThinkingBlock
    | RedactedThinkingBlock
    | ToolUseBlock
    | ToolResultBlock
    | ImageBlock
    | UnknownContentBlock
)


_CONTENT_BLOCK_MODEL_BY_TYPE: dict[str, type[_BlockBase]] = {
    "text": TextBlock,
    "thinking": ThinkingBlock,
    "redacted_thinking": RedactedThinkingBlock,
    "tool_use": ToolUseBlock,
    "tool_result": ToolResultBlock,
    "image": ImageBlock,
}


def parse_content_block(raw: dict[str, Any]) -> ContentBlock:
    """Parse a single content block into its typed model.

    Falls back to UnknownContentBlock for unknown types (server-tool siblings,
    future block types). Raw payload preserved via extra="allow".
    """
    block_type = raw.get("type", "unknown")
    model_cls = _CONTENT_BLOCK_MODEL_BY_TYPE.get(block_type, UnknownContentBlock)
    return model_cls.model_validate(raw)


# ---------------------------------------------------------------------------
# System entry subtype sub-models
# ---------------------------------------------------------------------------


class _SystemSubtypeBase(_CamelBase):
    pass


class TurnDurationPayload(_SystemSubtypeBase):
    """system.subtype=turn_duration -- wall-clock time for one turn."""

    subtype: Literal["turn_duration"] = "turn_duration"
    duration_ms: int = 0
    message_count: int = 0


class StopHookSummaryPayload(_SystemSubtypeBase):
    """system.subtype=stop_hook_summary -- aggregated Stop/SubagentStop hook output."""

    subtype: Literal["stop_hook_summary"] = "stop_hook_summary"
    hook_count: int = 0
    hook_infos: list[dict[str, Any]] = Field(default_factory=list)
    # Observed in real data as plain error-message strings (a hook script's
    # stderr text), not structured dicts -- a full corpus scan found zero
    # dict-shaped elements. Widened to tolerate both rather than assume.
    hook_errors: list[dict[str, Any] | str] = Field(default_factory=list)
    prevented_continuation: bool = False
    stop_reason: str | None = None
    has_output: bool = False
    level: str | None = None
    tool_use_id: str | None = Field(default=None, alias="toolUseID")


class ApiErrorPayload(_SystemSubtypeBase):
    """system.subtype=api_error -- Anthropic API error (retryable)."""

    subtype: Literal["api_error"] = "api_error"
    error: dict[str, Any] = Field(default_factory=dict)  # {status, headers, requestID, type}
    # Observed as float in real data (e.g. 534.31...) -- API jitter values.
    retry_in_ms: float | None = None
    retry_attempt: int | None = None
    max_retries: int | None = None
    level: str | None = None


class CompactBoundaryPayload(_SystemSubtypeBase):
    """system.subtype=compact_boundary -- conversation compaction happened."""

    subtype: Literal["compact_boundary"] = "compact_boundary"
    content: str = ""
    # compactMetadata = {trigger: "auto" | "manual", preTokens: int}
    compact_metadata: dict[str, Any] = Field(default_factory=dict)
    logical_parent_uuid: str | None = None


class LocalCommandPayload(_SystemSubtypeBase):
    """system.subtype=local_command -- slash command output."""

    subtype: Literal["local_command"] = "local_command"
    content: str = ""
    level: str | None = None


class AwaySummaryPayload(_SystemSubtypeBase):
    """system.subtype=away_summary -- recap when user returns."""

    subtype: Literal["away_summary"] = "away_summary"
    content: str = ""
    level: str | None = None


class BridgeStatusPayload(_SystemSubtypeBase):
    """system.subtype=bridge_status -- remote-control bridge on/off."""

    subtype: Literal["bridge_status"] = "bridge_status"
    content: str = ""
    url: str | None = None
    # upgradeNudge is observed as a STRING (an upgrade message),
    # NOT a boolean as the empirical doc initially recorded.
    upgrade_nudge: str | bool | None = None


class UnknownSystemPayload(_SystemSubtypeBase):
    subtype: str = "unknown"


SystemSubtypePayload = (
    TurnDurationPayload
    | StopHookSummaryPayload
    | ApiErrorPayload
    | CompactBoundaryPayload
    | LocalCommandPayload
    | AwaySummaryPayload
    | BridgeStatusPayload
    | UnknownSystemPayload
)


_SYSTEM_SUBTYPE_MODEL_BY_NAME: dict[str, type[_SystemSubtypeBase]] = {
    "turn_duration": TurnDurationPayload,
    "stop_hook_summary": StopHookSummaryPayload,
    "api_error": ApiErrorPayload,
    "compact_boundary": CompactBoundaryPayload,
    "local_command": LocalCommandPayload,
    "away_summary": AwaySummaryPayload,
    "bridge_status": BridgeStatusPayload,
}


def parse_system_payload(raw: dict[str, Any]) -> SystemSubtypePayload:
    """Parse a system entry into its typed subtype payload.

    Takes the full system-entry dict (not just the subtype discriminator).
    Falls back to UnknownSystemPayload for unknown subtypes.
    """
    subtype = raw.get("subtype", "unknown")
    model_cls = _SYSTEM_SUBTYPE_MODEL_BY_NAME.get(subtype, UnknownSystemPayload)
    return model_cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Progress data variant sub-models (inside progress.data)
# ---------------------------------------------------------------------------


class _ProgressDataBase(_CamelBase):
    pass


class HookProgressData(_ProgressDataBase):
    """progress.data.type=hook_progress -- lifecycle event for a running hook."""

    type: Literal["hook_progress"] = "hook_progress"
    hook_event: str | None = None
    hook_name: str | None = None
    command: str | None = None


class BashProgressData(_ProgressDataBase):
    """progress.data.type=bash_progress -- streaming bash output."""

    type: Literal["bash_progress"] = "bash_progress"
    stdout: str | None = None
    stderr: str | None = None


class AgentProgressData(_ProgressDataBase):
    """progress.data.type=agent_progress -- subagent progress update."""

    type: Literal["agent_progress"] = "agent_progress"
    agent_id: str | None = None


class QueryUpdateData(_ProgressDataBase):
    type: Literal["query_update"] = "query_update"


class SearchResultsReceivedData(_ProgressDataBase):
    type: Literal["search_results_received"] = "search_results_received"


class McpProgressData(_ProgressDataBase):
    type: Literal["mcp_progress"] = "mcp_progress"


class UnknownProgressData(_ProgressDataBase):
    type: str = "unknown"


ProgressData = (
    HookProgressData
    | BashProgressData
    | AgentProgressData
    | QueryUpdateData
    | SearchResultsReceivedData
    | McpProgressData
    | UnknownProgressData
)


_PROGRESS_DATA_MODEL_BY_TYPE: dict[str, type[_ProgressDataBase]] = {
    "hook_progress": HookProgressData,
    "bash_progress": BashProgressData,
    "agent_progress": AgentProgressData,
    "query_update": QueryUpdateData,
    "search_results_received": SearchResultsReceivedData,
    "mcp_progress": McpProgressData,
}


def parse_progress_data(raw: dict[str, Any]) -> ProgressData:
    """Parse a progress.data payload into its typed variant.

    Falls back to UnknownProgressData for unknown data.type values.
    """
    data_type = raw.get("type", "unknown")
    model_cls = _PROGRESS_DATA_MODEL_BY_TYPE.get(data_type, UnknownProgressData)
    return model_cls.model_validate(raw)


# ---------------------------------------------------------------------------
# toolUseResult sub-models per tool
# ---------------------------------------------------------------------------
# Shape per the empirical reference + observed real-data variants. Tools that
# error out collapse to a plain `Error: ...` string regardless of normal shape;
# the parse_tool_use_result dispatcher returns ToolErrorString for those cases.


class _ToolResultBase(_CamelBase):
    pass


class ReadResult(_ToolResultBase):
    """Read tool result. type='text' for text files, 'image' for images."""

    type: str | None = None  # "text" or "image"
    file: dict[str, Any] = Field(default_factory=dict)  # {filePath, content/base64, numLines, ...}


class EditResult(_ToolResultBase):
    """Edit tool result. Includes the structuredPatch hunks."""

    file_path: str | None = None
    old_string: str | None = None
    new_string: str | None = None
    original_file: str | None = None
    structured_patch: list[dict[str, Any]] = Field(default_factory=list)
    user_modified: bool | None = None
    replace_all: bool | None = None


class WriteResult(_ToolResultBase):
    """Write tool result. type is 'create' or 'update'."""

    type: str | None = None
    file_path: str | None = None
    content: str | None = None
    structured_patch: list[dict[str, Any]] = Field(default_factory=list)
    user_modified: bool | None = None


class GlobResult(_ToolResultBase):
    filenames: list[str] = Field(default_factory=list)
    num_files: int = 0
    truncated: bool | None = None
    duration_ms: int | None = None


class GrepResult(_ToolResultBase):
    mode: str | None = None  # "content" | "files_with_matches" | "count"
    num_files: int = 0
    filenames: list[str] = Field(default_factory=list)
    content: str | None = None
    num_lines: int | None = None


class BashResult(_ToolResultBase):
    """Bash tool result. interrupted, exitCode, etc. preserved -- key signals
    for behavioral analysis (when did Bash get interrupted?)."""

    stdout: str | None = None
    stderr: str | None = None
    interrupted: bool | None = None
    is_image: bool | None = None
    no_output_expected: bool | None = None
    exit_code: int | None = None
    truncated: bool | None = None
    shell_id: str | None = None
    duration_ms: int | float | None = None


class WebFetchResult(_ToolResultBase):
    bytes: int | None = None
    code: int | None = None
    code_text: str | None = None
    result: str | None = None
    url: str | None = None


class WebSearchResult(_ToolResultBase):
    query: str | None = None
    # results is a heterogeneous list (server-tool metadata + summary string)
    results: list[Any] = Field(default_factory=list)


class ExitPlanModeResult(_ToolResultBase):
    """Captures the saved plan text + filePath when user accepts a plan."""

    plan: str | None = None
    is_agent: str | bool | None = None  # observed as string "false"/"true" in some versions
    file_path: str | None = None


class EnterPlanModeResult(_ToolResultBase):
    message: str | None = None


class TodoWriteResult(_ToolResultBase):
    old_todos: list[dict[str, Any]] = Field(default_factory=list)
    new_todos: list[dict[str, Any]] = Field(default_factory=list)
    verification_nudge_needed: bool | None = None


class AskUserQuestionResult(_ToolResultBase):
    """The QUESTION object, not the user's answer (which comes back as a
    text message in the next user entry)."""

    questions: list[dict[str, Any]] = Field(default_factory=list)


class AgentResult(_ToolResultBase):
    """Agent (formerly Task) tool result. Subagent delegation rollup with
    metrics computed from the subagent's own transcript."""

    status: str | None = None  # "completed", "interrupted", "error"
    prompt: str | None = None
    agent_id: str | None = None
    agent_type: str | None = None
    content: list[dict[str, Any]] | str | None = None
    total_duration_ms: int | float | None = None
    total_tokens: int | None = None
    total_tool_use_count: int | None = None
    usage: dict[str, Any] | None = None
    was_interrupted: bool | None = None
    # Optional fields per version drift:
    subagent_type: str | None = None
    subagent_description: str | None = None
    result_uuid: str | None = None
    child_session_id: str | None = None


class SkillResult(_ToolResultBase):
    success: bool | None = None
    command_name: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)


class ToolSearchResult(_ToolResultBase):
    matches: list[str] = Field(default_factory=list)
    query: str | None = None
    total_deferred_tools: int | None = None


class TaskCreateResult(_ToolResultBase):
    """TaskCreate tool result. Payload shape: {task: {id, subject}}."""

    task: dict[str, Any] | None = None


class TaskUpdateResult(_ToolResultBase):
    """TaskUpdate tool result."""

    success: bool | None = None
    task_id: str | None = None
    updated_fields: list[str] = Field(default_factory=list)
    status_change: dict[str, Any] | None = None
    verification_nudge_needed: bool | None = None


class TaskListResult(_ToolResultBase):
    """TaskList tool result. Returns the current task collection."""

    tasks: list[dict[str, Any]] = Field(default_factory=list)


class TaskGetResult(_ToolResultBase):
    """TaskGet tool result. Returns a single task plus its detail."""

    task: dict[str, Any] | None = None


class GenericToolResult(_ToolResultBase):
    """Catch-all for tools we don't have typed models for, including MCP
    tools (mcp__server__tool which return arbitrary server-defined shapes).
    Raw payload preserved via extra='allow'."""


# Errors collapse to a plain string regardless of tool. Wrapped so callers
# can distinguish from a parsed payload via type check.
class ToolErrorString(BaseModel):
    """Tool-level error: the toolUseResult is just a plain string."""

    model_config = ConfigDict(extra="allow")
    error_text: str


# Type alias for any toolUseResult shape.
ToolUseResult = (
    ReadResult
    | EditResult
    | WriteResult
    | GlobResult
    | GrepResult
    | BashResult
    | WebFetchResult
    | WebSearchResult
    | ExitPlanModeResult
    | EnterPlanModeResult
    | TodoWriteResult
    | AskUserQuestionResult
    | AgentResult
    | SkillResult
    | ToolSearchResult
    | TaskCreateResult
    | TaskUpdateResult
    | TaskListResult
    | TaskGetResult
    | GenericToolResult
    | ToolErrorString
    | None
)


_TOOL_RESULT_MODEL_BY_NAME: dict[str, type[_ToolResultBase]] = {
    "Read": ReadResult,
    "Edit": EditResult,
    "MultiEdit": EditResult,  # MultiEdit returns same shape as Edit (multiple structuredPatches)
    "Write": WriteResult,
    "Glob": GlobResult,
    "Grep": GrepResult,
    "Bash": BashResult,
    "BashOutput": BashResult,  # Same shape as Bash
    "WebFetch": WebFetchResult,
    "WebSearch": WebSearchResult,
    "ExitPlanMode": ExitPlanModeResult,
    "EnterPlanMode": EnterPlanModeResult,
    "TodoWrite": TodoWriteResult,
    "AskUserQuestion": AskUserQuestionResult,
    "Agent": AgentResult,
    "Task": AgentResult,  # pre-v2.1.63 alias for Agent
    "TaskCreate": TaskCreateResult,
    "TaskUpdate": TaskUpdateResult,
    "TaskList": TaskListResult,
    "TaskGet": TaskGetResult,
    "Skill": SkillResult,
    "ToolSearch": ToolSearchResult,
}


def parse_tool_use_result(tool_name: str | None, raw: Any) -> ToolUseResult:
    """Parse a toolUseResult payload into a typed model.

    Args:
        tool_name: The tool that produced the result (e.g. "Read", "Bash").
                   Used to dispatch to the right typed model. If None or
                   unknown, returns GenericToolResult.
        raw: The toolUseResult value (dict, list, str, or None).

    Returns:
        - None if raw is None
        - ToolErrorString if raw is a plain string (always indicates an error)
        - GenericToolResult for list-shape (MCP tools), unknown tool names,
          or when raw isn't a dict
        - The typed per-tool model otherwise
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return ToolErrorString(error_text=raw)
    if isinstance(raw, list):
        # MCP tools and server-tool results return list-of-content-blocks.
        # Wrap as GenericToolResult with the list under a known field for
        # forward-compat -- callers can inspect content from model_extra.
        return GenericToolResult.model_validate({"content": raw})
    if not isinstance(raw, dict):
        return GenericToolResult.model_validate({})

    if tool_name is None:
        return GenericToolResult.model_validate(raw)

    model_cls = _TOOL_RESULT_MODEL_BY_NAME.get(tool_name, GenericToolResult)
    return model_cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


def iter_typed_entries(jsonl_path) -> Iterator[SessionLogEntry]:
    """Iterate over a JSONL session file as typed log entries.

    Yields one SessionLogEntry per line. Malformed JSON lines are skipped
    silently (matches existing `iter_session_entries` behavior). Unknown
    entry types yield `UnknownEntry` rather than raising.
    """
    import json
    from pathlib import Path

    path = Path(jsonl_path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield parse_log_entry(raw)
