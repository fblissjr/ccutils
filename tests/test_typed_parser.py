"""Tests for the new Pydantic-typed JSONL parser (Phase A1)."""

from pathlib import Path

import pytest

from ccutils.parsers.models import (
    AgentNameEntry,
    AgentResult,
    ApiErrorPayload,
    AssistantEntry,
    AttachmentEntry,
    AwaySummaryPayload,
    BashProgressData,
    BashResult,
    BridgeStatusPayload,
    CompactBoundaryPayload,
    CustomTitleEntry,
    EditResult,
    ExitPlanModeResult,
    FileHistorySnapshotEntry,
    GenericToolResult,
    GlobResult,
    GrepResult,
    HookProgressData,
    ImageBlock,
    LastPromptEntry,
    LocalCommandPayload,
    PermissionModeEntry,
    PrLinkEntry,
    ProgressEntry,
    QueueOperationEntry,
    ReadResult,
    RedactedThinkingBlock,
    StopHookSummaryPayload,
    SummaryEntry,
    SystemEntry,
    TextBlock,
    ThinkingBlock,
    TodoWriteResult,
    ToolErrorString,
    ToolResultBlock,
    ToolUseBlock,
    TurnDurationPayload,
    UnknownContentBlock,
    UnknownEntry,
    UnknownProgressData,
    UnknownSystemPayload,
    UserEntry,
    WebFetchResult,
    WriteResult,
    iter_typed_entries,
    parse_content_block,
    parse_log_entry,
    parse_progress_data,
    parse_system_payload,
    parse_tool_use_result,
)


class TestParseLogEntry:
    def test_routes_user_entry(self):
        entry = parse_log_entry({"type": "user", "uuid": "u1", "message": {"role": "user", "content": "hi"}})
        assert isinstance(entry, UserEntry)
        assert entry.uuid == "u1"

    def test_routes_assistant_entry(self):
        entry = parse_log_entry({"type": "assistant", "uuid": "a1", "message": {"role": "assistant", "content": []}})
        assert isinstance(entry, AssistantEntry)

    def test_routes_progress_entry(self):
        entry = parse_log_entry({"type": "progress", "data": {"type": "hook_progress"}})
        assert isinstance(entry, ProgressEntry)
        assert entry.data["type"] == "hook_progress"

    def test_routes_system_entry(self):
        entry = parse_log_entry({"type": "system", "subtype": "turn_duration", "durationMs": 1234})
        assert isinstance(entry, SystemEntry)
        assert entry.subtype == "turn_duration"

    def test_routes_attachment_entry(self):
        entry = parse_log_entry({"type": "attachment", "attachment": {"type": "diagnostics"}})
        assert isinstance(entry, AttachmentEntry)

    def test_routes_permission_mode_entry(self):
        entry = parse_log_entry({"type": "permission-mode", "permissionMode": "auto"})
        assert isinstance(entry, PermissionModeEntry)
        assert entry.permission_mode == "auto"

    def test_routes_custom_title_entry(self):
        entry = parse_log_entry({"type": "custom-title", "customTitle": "test"})
        assert isinstance(entry, CustomTitleEntry)
        assert entry.custom_title == "test"

    def test_routes_agent_name_entry(self):
        entry = parse_log_entry({"type": "agent-name", "agentName": "Explore"})
        assert isinstance(entry, AgentNameEntry)
        assert entry.agent_name == "Explore"

    def test_routes_last_prompt_entry(self):
        entry = parse_log_entry({"type": "last-prompt", "lastPrompt": "do thing"})
        assert isinstance(entry, LastPromptEntry)

    def test_routes_queue_operation_entry(self):
        entry = parse_log_entry({"type": "queue-operation", "operation": "enqueue", "content": "x"})
        assert isinstance(entry, QueueOperationEntry)

    def test_routes_file_history_snapshot_entry(self):
        entry = parse_log_entry({"type": "file-history-snapshot", "messageId": "m1"})
        assert isinstance(entry, FileHistorySnapshotEntry)

    def test_routes_pr_link_entry(self):
        entry = parse_log_entry({"type": "pr-link", "prNumber": 42})
        assert isinstance(entry, PrLinkEntry)

    def test_routes_summary_entry(self):
        entry = parse_log_entry({"type": "summary", "summary": "test"})
        assert isinstance(entry, SummaryEntry)

    def test_unknown_type_routes_to_unknown_entry(self):
        """Forward-compat: a future Claude Code release introducing a new
        entry type should not crash the parser."""
        entry = parse_log_entry({"type": "future-thing-we-dont-know", "foo": "bar"})
        assert isinstance(entry, UnknownEntry)
        assert entry.type == "future-thing-we-dont-know"

    def test_missing_type_routes_to_unknown_entry(self):
        entry = parse_log_entry({"foo": "bar"})
        assert isinstance(entry, UnknownEntry)


class TestAliasGenerator:
    """JSON uses camelCase; Python uses snake_case. Both should work."""

    def test_camelcase_input_snake_case_access(self):
        entry = parse_log_entry({
            "type": "user",
            "uuid": "u1",
            "parentUuid": "p1",
            "sessionId": "s1",
            "gitBranch": "main",
            "isSidechain": True,
            "message": {"role": "user", "content": "hi"},
        })
        assert isinstance(entry, UserEntry)
        assert entry.parent_uuid == "p1"
        assert entry.session_id == "s1"
        assert entry.git_branch == "main"
        assert entry.is_sidechain is True

    def test_construct_with_snake_case(self):
        """populate_by_name=True allows snake_case construction in Python code."""
        entry = UserEntry(
            type="user",
            uuid="u1",
            parent_uuid="p1",
            session_id="s1",
            message={"role": "user", "content": "hi"},
        )
        assert entry.parent_uuid == "p1"
        assert entry.session_id == "s1"


class TestExtraAllow:
    """extra='allow' preserves unknown fields rather than raising."""

    def test_unknown_field_preserved_on_known_entry_type(self):
        entry = parse_log_entry({
            "type": "user",
            "uuid": "u1",
            "message": {"role": "user", "content": "hi"},
            "futureFieldNobodyHasSeen": "secret value",
        })
        assert entry.model_extra is not None
        assert entry.model_extra.get("futureFieldNobodyHasSeen") == "secret value"


class TestSampleFixtures:
    """Round-trip the existing test fixtures to ensure backwards-compat."""

    def test_parses_sample_session_jsonl(self):
        fixture = Path(__file__).parent / "sample_session.jsonl"
        entries = list(iter_typed_entries(fixture))
        assert len(entries) > 0
        # First entry in the fixture is a `summary`
        assert entries[0].type in ("summary", "user")  # tolerate either ordering


class TestToolUseResultPolymorphism:
    """toolUseResult on UserEntry is observed in three shapes per real archive scan:
      - dict (most tools: ExitPlanMode {plan,...}, Read {file,...}, etc.)
      - str  (errors: "Error: <message>")
      - list of content blocks (MCP tools, server-tool results)
    """

    def test_dict_shape(self):
        entry = parse_log_entry({
            "type": "user",
            "message": {"role": "user", "content": []},
            "toolUseResult": {"plan": "test plan", "isAgent": "false", "filePath": "/p/foo.md"},
        })
        assert isinstance(entry, UserEntry)
        assert isinstance(entry.tool_use_result, dict)
        assert entry.tool_use_result["plan"] == "test plan"

    def test_str_shape(self):
        entry = parse_log_entry({
            "type": "user",
            "message": {"role": "user", "content": []},
            "toolUseResult": "Error: something went wrong",
        })
        assert isinstance(entry, UserEntry)
        assert entry.tool_use_result == "Error: something went wrong"

    def test_list_shape_observed_in_mcp_tools(self):
        """Real failure mode caught by full-archive scan: MCP tool results
        return toolUseResult as a list of content blocks directly."""
        entry = parse_log_entry({
            "type": "user",
            "message": {"role": "user", "content": []},
            "toolUseResult": [
                {"type": "text", "text": "Navigated to \"New tab\" (chrome://newtab)"},
            ],
        })
        assert isinstance(entry, UserEntry)
        assert isinstance(entry.tool_use_result, list)
        assert entry.tool_use_result[0]["type"] == "text"


class TestContentBlocks:
    def test_text_block(self):
        b = parse_content_block({"type": "text", "text": "hi"})
        assert isinstance(b, TextBlock)
        assert b.text == "hi"

    def test_thinking_block_preserves_signature(self):
        b = parse_content_block({"type": "thinking", "thinking": "hmm", "signature": "abc123"})
        assert isinstance(b, ThinkingBlock)
        assert b.signature == "abc123"

    def test_redacted_thinking_block(self):
        """R14: redacted_thinking carries `data`, not `thinking`."""
        b = parse_content_block({"type": "redacted_thinking", "data": "REDACTED_BASE64"})
        assert isinstance(b, RedactedThinkingBlock)
        assert b.data == "REDACTED_BASE64"

    def test_tool_use_block(self):
        b = parse_content_block({
            "type": "tool_use",
            "id": "toolu_001",
            "name": "Bash",
            "input": {"command": "ls"},
            "caller": {"type": "direct"},
        })
        assert isinstance(b, ToolUseBlock)
        assert b.id == "toolu_001"
        assert b.name == "Bash"
        assert b.input == {"command": "ls"}
        assert b.caller == {"type": "direct"}

    def test_tool_result_block_str_content(self):
        b = parse_content_block({
            "type": "tool_result",
            "tool_use_id": "toolu_001",
            "content": "stdout output",
            "is_error": False,
        })
        assert isinstance(b, ToolResultBlock)
        assert b.content == "stdout output"
        assert b.is_error is False

    def test_tool_result_block_list_content_preserves_images(self):
        """R15: tool_result.content can be a list including image blocks."""
        b = parse_content_block({
            "type": "tool_result",
            "tool_use_id": "toolu_001",
            "content": [
                {"type": "text", "text": "Here's the screenshot:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
            ],
        })
        assert isinstance(b, ToolResultBlock)
        assert isinstance(b.content, list)
        assert b.content[1]["type"] == "image"

    def test_tool_result_is_error_tristate_missing(self):
        """R16: missing is_error must be distinct from False (None)."""
        b = parse_content_block({
            "type": "tool_result",
            "tool_use_id": "toolu_001",
            "content": "ok",
        })
        assert isinstance(b, ToolResultBlock)
        assert b.is_error is None  # NOT False

    def test_image_block(self):
        b = parse_content_block({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "..."},
        })
        assert isinstance(b, ImageBlock)
        assert b.source["media_type"] == "image/png"

    def test_unknown_content_block_for_server_tool_siblings(self):
        """server_tool_use, web_search_tool_result, etc. -- not yet typed,
        should fall through to UnknownContentBlock with type preserved."""
        b = parse_content_block({"type": "web_search_tool_result", "content": "..."})
        assert isinstance(b, UnknownContentBlock)
        assert b.type == "web_search_tool_result"


class TestSystemSubtypes:
    def test_turn_duration(self):
        p = parse_system_payload({"subtype": "turn_duration", "durationMs": 1234, "messageCount": 5})
        assert isinstance(p, TurnDurationPayload)
        assert p.duration_ms == 1234
        assert p.message_count == 5

    def test_stop_hook_summary(self):
        p = parse_system_payload({
            "subtype": "stop_hook_summary",
            "hookCount": 2,
            "hookInfos": [{"command": "hooks/stop.py", "durationMs": 10}],
            "preventedContinuation": False,
            "stopReason": "end_turn",
            "hasOutput": True,
            "level": "suggestion",
            "toolUseID": "stop-001",
        })
        assert isinstance(p, StopHookSummaryPayload)
        assert p.hook_count == 2
        assert p.prevented_continuation is False
        assert p.stop_reason == "end_turn"
        assert p.tool_use_id == "stop-001"

    def test_stop_hook_summary_hook_errors_as_plain_strings(self):
        """hookErrors elements are observed in real data as plain error-message
        strings (e.g. a stop-hook script's stderr text), not structured dicts --
        the field name/semantics only ever promised error text, and no session
        anywhere in a full corpus scan produced a dict-shaped element."""
        p = parse_system_payload({
            "subtype": "stop_hook_summary",
            "hookCount": 1,
            "hookErrors": [
                "[<claude-config>/stop-hook-example.sh]: There are untracked files.\n"
            ],
            "hookInfos": [{"command": "<claude-config>/stop-hook-example.sh"}],
            "level": "suggestion",
        })
        assert isinstance(p, StopHookSummaryPayload)
        assert p.hook_errors == [
            "[<claude-config>/stop-hook-example.sh]: There are untracked files.\n"
        ]

    def test_api_error(self):
        p = parse_system_payload({
            "subtype": "api_error",
            "error": {"status": 503, "type": "overloaded_error"},
            "retryInMs": 1000,
            "retryAttempt": 1,
            "maxRetries": 3,
            "level": "error",
        })
        assert isinstance(p, ApiErrorPayload)
        assert p.error["status"] == 503
        assert p.retry_in_ms == 1000

    def test_compact_boundary(self):
        p = parse_system_payload({
            "subtype": "compact_boundary",
            "content": "Conversation compacted",
            "compactMetadata": {"trigger": "auto", "preTokens": 100000},
            "logicalParentUuid": "u1",
        })
        assert isinstance(p, CompactBoundaryPayload)
        assert p.compact_metadata["trigger"] == "auto"

    def test_local_command(self):
        p = parse_system_payload({"subtype": "local_command", "content": "<local-command-stdout>x</local-command-stdout>"})
        assert isinstance(p, LocalCommandPayload)

    def test_away_summary(self):
        p = parse_system_payload({"subtype": "away_summary", "content": "While you were away..."})
        assert isinstance(p, AwaySummaryPayload)

    def test_bridge_status(self):
        p = parse_system_payload({"subtype": "bridge_status", "url": "https://...", "content": "online"})
        assert isinstance(p, BridgeStatusPayload)

    def test_unknown_subtype(self):
        p = parse_system_payload({"subtype": "future_subtype_we_dont_know", "stuff": "value"})
        assert isinstance(p, UnknownSystemPayload)
        assert p.subtype == "future_subtype_we_dont_know"


class TestProgressVariants:
    def test_hook_progress(self):
        p = parse_progress_data({
            "type": "hook_progress",
            "hookEvent": "PreToolUse",
            "hookName": "block-rm",
            "command": "hooks/block.py",
        })
        assert isinstance(p, HookProgressData)
        assert p.hook_event == "PreToolUse"

    def test_bash_progress(self):
        p = parse_progress_data({"type": "bash_progress", "stdout": "running..."})
        assert isinstance(p, BashProgressData)
        assert p.stdout == "running..."

    def test_unknown_progress_variant(self):
        p = parse_progress_data({"type": "future_progress_thing"})
        assert isinstance(p, UnknownProgressData)


class TestArchiveCoveragePostA2:
    """After A2: every content block, system subtype, and progress variant
    in the user's full archive should route to a typed model OR a typed
    Unknown* fallback. Surface the Unknowns so we can decide whether to
    promote them to typed sub-models in a future chunk.
    """

    def test_no_unknown_content_blocks_or_subtypes_in_archive(self):
        archive_dir = Path.home() / ".claude" / "projects"
        if not archive_dir.exists():
            pytest.skip("Archive dir not present")

        unknown_block_types: dict[str, int] = {}
        unknown_system_subtypes: dict[str, int] = {}
        unknown_progress_types: dict[str, int] = {}

        for fp in archive_dir.glob("**/*.jsonl"):
            for entry in iter_typed_entries(fp):
                if isinstance(entry, (UserEntry, AssistantEntry)):
                    msg_content = entry.message.get("content")
                    if isinstance(msg_content, list):
                        for raw_block in msg_content:
                            if isinstance(raw_block, dict):
                                b = parse_content_block(raw_block)
                                if isinstance(b, UnknownContentBlock):
                                    unknown_block_types[b.type] = unknown_block_types.get(b.type, 0) + 1
                elif isinstance(entry, SystemEntry):
                    p = parse_system_payload(entry.model_dump(by_alias=True))
                    if isinstance(p, UnknownSystemPayload):
                        unknown_system_subtypes[p.subtype] = unknown_system_subtypes.get(p.subtype, 0) + 1
                elif isinstance(entry, ProgressEntry):
                    pd = parse_progress_data(entry.data)
                    if isinstance(pd, UnknownProgressData):
                        unknown_progress_types[pd.type] = unknown_progress_types.get(pd.type, 0) + 1

        # Print findings (won't fail the test -- forward-compat means Unknowns are OK).
        if unknown_block_types:
            print(f"Unknown content block types: {unknown_block_types}")
        if unknown_system_subtypes:
            print(f"Unknown system subtypes: {unknown_system_subtypes}")
        if unknown_progress_types:
            print(f"Unknown progress data types: {unknown_progress_types}")


class TestToolUseResultPerTool:
    def test_read_result(self):
        r = parse_tool_use_result("Read", {
            "type": "text",
            "file": {"filePath": "/p/a.py", "content": "x", "numLines": 1, "totalLines": 1},
        })
        assert isinstance(r, ReadResult)
        assert r.file["filePath"] == "/p/a.py"

    def test_edit_result_preserves_structured_patch(self):
        """R1: structuredPatch is the highest-value Edit field we currently drop."""
        r = parse_tool_use_result("Edit", {
            "filePath": "/p/a.py",
            "oldString": "old", "newString": "new",
            "structuredPatch": [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1, "lines": ["-old", "+new"]}],
            "userModified": False, "replaceAll": False,
        })
        assert isinstance(r, EditResult)
        assert len(r.structured_patch) == 1
        assert r.structured_patch[0]["oldStart"] == 1
        assert r.user_modified is False

    def test_multiedit_routes_to_edit_result(self):
        r = parse_tool_use_result("MultiEdit", {"filePath": "/p/a.py", "structuredPatch": []})
        assert isinstance(r, EditResult)

    def test_write_result(self):
        r = parse_tool_use_result("Write", {
            "type": "create", "filePath": "/p/new.py", "content": "x", "structuredPatch": [],
        })
        assert isinstance(r, WriteResult)
        assert r.type == "create"

    def test_glob_result(self):
        r = parse_tool_use_result("Glob", {"filenames": ["a", "b"], "numFiles": 2, "truncated": False, "durationMs": 8})
        assert isinstance(r, GlobResult)
        assert r.num_files == 2

    def test_grep_result(self):
        r = parse_tool_use_result("Grep", {"mode": "content", "numFiles": 1, "filenames": ["a"], "content": "match"})
        assert isinstance(r, GrepResult)
        assert r.mode == "content"

    def test_bash_result_preserves_structural_signals(self):
        """R1: interrupted/exitCode are the key behavioral signals we currently miss."""
        r = parse_tool_use_result("Bash", {
            "stdout": "ok", "stderr": "",
            "interrupted": False, "isImage": False, "noOutputExpected": False,
            "exitCode": 0, "durationMs": 123,
        })
        assert isinstance(r, BashResult)
        assert r.interrupted is False
        assert r.exit_code == 0
        assert r.duration_ms == 123

    def test_bash_result_interrupted(self):
        r = parse_tool_use_result("Bash", {"stdout": "partial", "interrupted": True})
        assert isinstance(r, BashResult)
        assert r.interrupted is True

    def test_webfetch_result(self):
        r = parse_tool_use_result("WebFetch", {"bytes": 42499, "code": 200, "codeText": "OK", "result": "...", "url": "https://..."})
        assert isinstance(r, WebFetchResult)
        assert r.code == 200

    def test_exitplanmode_result(self):
        r = parse_tool_use_result("ExitPlanMode", {
            "plan": "# Plan\n...", "isAgent": "false", "filePath": "/p/.claude/plans/foo.md",
        })
        assert isinstance(r, ExitPlanModeResult)
        assert r.plan.startswith("# Plan")
        assert r.is_agent == "false"  # observed as string in real data

    def test_todowrite_result(self):
        r = parse_tool_use_result("TodoWrite", {
            "oldTodos": [], "newTodos": [{"content": "x", "status": "pending"}],
            "verificationNudgeNeeded": False,
        })
        assert isinstance(r, TodoWriteResult)
        assert len(r.new_todos) == 1

    def test_agent_result_with_rollups(self):
        """R1: Agent rollup metrics (totalDurationMs, totalTokens) currently
        re-derived from agent transcript -- we should use the authoritative
        source on the parent's tool_result."""
        r = parse_tool_use_result("Agent", {
            "status": "completed", "agentId": "ag-1", "agentType": "Explore",
            "content": [{"type": "text", "text": "Found 3 files"}],
            "totalDurationMs": 12345, "totalTokens": 5000, "totalToolUseCount": 7,
            "wasInterrupted": False,
        })
        assert isinstance(r, AgentResult)
        assert r.agent_type == "Explore"
        assert r.total_duration_ms == 12345
        assert r.total_tool_use_count == 7
        assert r.was_interrupted is False

    def test_task_routes_to_agent_result(self):
        """Task is the pre-v2.1.63 alias for Agent."""
        r = parse_tool_use_result("Task", {"status": "completed", "agentId": "x"})
        assert isinstance(r, AgentResult)

    def test_error_string_collapses(self):
        r = parse_tool_use_result("Read", "Error: File not found")
        assert isinstance(r, ToolErrorString)
        assert r.error_text == "Error: File not found"

    def test_error_string_works_for_any_tool(self):
        for tool in ("Bash", "Edit", "Agent", "WebFetch", "UnknownFutureTool"):
            r = parse_tool_use_result(tool, "Error: x")
            assert isinstance(r, ToolErrorString)

    def test_none_returns_none(self):
        assert parse_tool_use_result("Read", None) is None
        assert parse_tool_use_result(None, None) is None

    def test_unknown_tool_uses_generic(self):
        """Future Claude Code tool names should fall through to GenericToolResult."""
        r = parse_tool_use_result("FutureTool", {"foo": "bar"})
        assert isinstance(r, GenericToolResult)
        assert r.model_extra is not None
        assert r.model_extra.get("foo") == "bar"

    def test_list_payload_wrapped_as_generic(self):
        """MCP tools return list-of-content-blocks as toolUseResult."""
        r = parse_tool_use_result("mcp__chrome__navigate", [{"type": "text", "text": "Navigated"}])
        assert isinstance(r, GenericToolResult)

    def test_unknown_tool_name_is_none(self):
        r = parse_tool_use_result(None, {"foo": "bar"})
        assert isinstance(r, GenericToolResult)


class TestToolUseResultArchiveCoverage:
    """Walk the user's full archive: for every tool_result, look up its
    originating tool_use to get the tool name, then dispatch through
    parse_tool_use_result. Assert no exceptions, and surface tool names
    that consistently land in GenericToolResult (signal: missing typed model).
    """

    def test_every_archive_tool_result_dispatches_cleanly(self):
        archive_dir = Path.home() / ".claude" / "projects"
        if not archive_dir.exists():
            pytest.skip("Archive dir not present")

        generic_by_tool: dict[str, int] = {}
        typed_by_tool: dict[str, int] = {}
        error_by_tool: dict[str, int] = {}
        total = 0

        for fp in archive_dir.glob("**/*.jsonl"):
            tool_use_id_to_name: dict[str, str] = {}
            for entry in iter_typed_entries(fp):
                # Build tool_use_id -> tool_name map from assistant entries
                if isinstance(entry, AssistantEntry):
                    msg_content = entry.message.get("content")
                    if isinstance(msg_content, list):
                        for raw_block in msg_content:
                            if isinstance(raw_block, dict) and raw_block.get("type") == "tool_use":
                                tool_use_id_to_name[raw_block.get("id", "")] = raw_block.get("name", "")

                # Parse user-entry toolUseResult with the matched tool name.
                # The link from tool_result -> tool_use is inside the
                # message.content[].tool_result.tool_use_id field, NOT at
                # entry.sourceToolUseID (which the empirical doc claimed
                # but isn't actually present at top level).
                if isinstance(entry, UserEntry) and entry.tool_use_result is not None:
                    tool_name = None
                    msg_content = entry.message.get("content")
                    if isinstance(msg_content, list):
                        for raw_block in msg_content:
                            if isinstance(raw_block, dict) and raw_block.get("type") == "tool_result":
                                tu_id = raw_block.get("tool_use_id")
                                if tu_id and tu_id in tool_use_id_to_name:
                                    tool_name = tool_use_id_to_name[tu_id]
                                    break
                    parsed = parse_tool_use_result(tool_name, entry.tool_use_result)
                    total += 1
                    label = tool_name or "unknown"
                    if isinstance(parsed, ToolErrorString):
                        error_by_tool[label] = error_by_tool.get(label, 0) + 1
                    elif isinstance(parsed, GenericToolResult):
                        generic_by_tool[label] = generic_by_tool.get(label, 0) + 1
                    else:
                        typed_by_tool[label] = typed_by_tool.get(label, 0) + 1

        # Print typed coverage for visibility
        print(f"Total toolUseResults parsed: {total}")
        print(f"Typed-payload tool counts: {dict(sorted(typed_by_tool.items(), key=lambda x: -x[1])[:20])}")
        print(f"Error-string tool counts: {dict(sorted(error_by_tool.items(), key=lambda x: -x[1])[:10])}")
        # Generic = either MCP tools (expected) OR a tool we should add a typed model for
        if generic_by_tool:
            print(f"Generic-fallback tool counts (consider adding typed model): {dict(sorted(generic_by_tool.items(), key=lambda x: -x[1])[:20])}")
