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

from typing import Any, Literal

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
