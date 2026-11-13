"""Tests for the new Pydantic-typed JSONL parser (Phase A1)."""

import json
from pathlib import Path

import pytest

from ccutils.parsers.models import (
    UserEntry,
    parse_log_entry,
)


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


class TestIsErrorAbsenceContract:
    """The invariant that lets an omitted is_error be read as success.

    Claude Code writes `is_error: true` on a failed tool_result. Success is
    encoded either by `is_error: false` or by omitting the field, and those
    two are equivalent -- the API defines an absent is_error as false. The
    consumers rely on exactly that equivalence (see the accepted-by-absence
    branch in etl/fact_plan_revisions.py, and the
    `SUM(CASE WHEN is_error THEN 1 ELSE 0 END)` error counts, which send NULL
    to the non-error branch).

    NOT asserted: "only Bash writes false". That looked true on a 71,635-row
    corpus (Bash 1,034 true / 31,259 false / 0 null; every other tool
    true-or-absent) but it is incidental -- a scan turned up a single
    `Workflow` result writing false, which is harmless precisely because
    false and absent mean the same thing. Encoding that pattern as a test
    would break on correct data.

    Claim: delete this and a third encoding of is_error (a string, an object,
    an explicit null meaning "unknown") could appear and silently reroute
    every error count, because the tri-state CASE in fact_tool_calls.py would
    coerce it rather than fail.
    """

    def test_is_error_is_only_ever_true_false_or_absent(self):
        archive_dir = Path.home() / ".claude" / "projects"
        if not archive_dir.exists():
            pytest.skip("Archive dir not present")

        bad_values: dict[str, int] = {}
        true_seen = 0
        false_seen = 0
        for jsonl in archive_dir.glob("**/*.jsonl"):
            try:
                lines = jsonl.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                if '"is_error"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = entry.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    if "is_error" not in block:
                        continue
                    value = block["is_error"]
                    if value is True:
                        true_seen += 1
                    elif value is False:
                        false_seen += 1
                    else:
                        key = f"{type(value).__name__}:{value!r}"
                        bad_values[key] = bad_values.get(key, 0) + 1

        # Guard against a vacuous pass: both encodings must actually occur,
        # or this scan proves nothing about the contract.
        assert true_seen > 0 and false_seen > 0, (
            f"is_error scan found true={true_seen} false={false_seen}; "
            "the archive does not exercise both encodings, so the "
            "absent-means-false reading cannot be confirmed from it."
        )
        assert not bad_values, (
            "is_error carried a value that is neither true nor false. A third "
            "encoding (notably an explicit null meaning 'unknown') would break "
            "the absent-means-false reading that fact_plan_revisions.py and "
            f"every error-rate view depend on. Found: {bad_values}"
        )


