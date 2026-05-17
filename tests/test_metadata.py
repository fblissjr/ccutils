# path-privacy: skip-file -- generic /Users/fred and /Users/dev placeholders only
"""Tests for session metadata extraction."""

import json

import pytest

from ccutils.parsers.metadata import (
    derive_project_name,
    extract_rich_metadata,
    format_duration,
    get_meaningful_summary,
    shorten_model_name,
    _is_skip_summary,
)


class TestShortenModelName:
    """Tests for model name shortening."""

    def test_opus_4_6(self):
        assert shorten_model_name("claude-opus-4-6") == "opus-4.6"

    def test_sonnet_4_5_with_date(self):
        assert shorten_model_name("claude-sonnet-4-5-20250929") == "sonnet-4.5"

    def test_sonnet_4_no_minor(self):
        assert shorten_model_name("claude-sonnet-4-20250514") == "sonnet-4"

    def test_haiku_4_5_with_date(self):
        assert shorten_model_name("claude-haiku-4-5-20251001") == "haiku-4.5"

    def test_claude_3_5_sonnet(self):
        assert shorten_model_name("claude-3-5-sonnet-20241022") == "sonnet-3.5"

    def test_claude_3_opus(self):
        assert shorten_model_name("claude-3-opus-20240229") == "opus-3"

    def test_none_returns_empty(self):
        assert shorten_model_name(None) == ""

    def test_empty_string_returns_empty(self):
        assert shorten_model_name("") == ""

    def test_unknown_model_strips_prefix(self):
        assert shorten_model_name("claude-future-model") == "future-model"

    def test_no_claude_prefix(self):
        assert shorten_model_name("gpt-4o") == "gpt-4o"

    def test_case_insensitive(self):
        assert shorten_model_name("Claude-Opus-4-6") == "opus-4.6"


class TestDeriveProjectName:
    """Tests for project name derivation from cwd."""

    def test_cwd_extracts_last_component(self):
        assert derive_project_name("/Users/fred/workspace/ccutils", "") == "ccutils"

    def test_cwd_with_trailing_slash(self):
        assert derive_project_name("/Users/fred/workspace/ccutils/", "") == "ccutils"

    def test_none_cwd_falls_back_to_folder(self):
        # get_project_display_name strips -Users- prefix, "workspace" is not
        # in skip_dirs, so result is "fred-workspace-ccutils"
        assert (
            derive_project_name(None, "-Users-fred-workspace-ccutils")
            == "fred-workspace-ccutils"
        )

    def test_empty_cwd_falls_back_to_folder(self):
        assert (
            derive_project_name("", "-Users-fred-workspace-ccutils")
            == "fred-workspace-ccutils"
        )

    def test_none_cwd_projects_folder(self):
        # "projects" IS in skip_dirs
        assert derive_project_name(None, "-Users-fred-projects-myapp") == "myapp"

    def test_deep_path(self):
        assert derive_project_name("/home/user/code/apps/webapp", "") == "webapp"


class TestIsSkipSummary:
    """Tests for summary skip detection."""

    def test_empty_is_skipped(self):
        assert _is_skip_summary("") is True

    def test_whitespace_is_skipped(self):
        assert _is_skip_summary("   ") is True

    def test_xml_is_skipped(self):
        assert _is_skip_summary("<system-reminder>some stuff</system-reminder>") is True

    def test_request_interrupted_is_skipped(self):
        assert _is_skip_summary("[Request interrupted by user for tool use]") is True

    def test_error_is_skipped(self):
        assert _is_skip_summary("[Error: something went wrong]") is True

    def test_api_error_is_skipped(self):
        assert _is_skip_summary("API error: rate limited") is True

    def test_warmup_is_skipped(self):
        assert _is_skip_summary("warmup") is True

    def test_normal_text_is_not_skipped(self):
        assert _is_skip_summary("Fix the login bug") is False

    def test_case_insensitive(self):
        assert _is_skip_summary("[REQUEST INTERRUPTED by user]") is True


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_none(self):
        assert format_duration(None) == ""

    def test_zero(self):
        assert format_duration(0) == "<1m"

    def test_minutes_only(self):
        assert format_duration(5) == "5m"
        assert format_duration(45) == "45m"

    def test_hours_and_minutes(self):
        assert format_duration(65) == "1h 5m"
        assert format_duration(125) == "2h 5m"

    def test_exact_hours(self):
        assert format_duration(60) == "1h"
        assert format_duration(120) == "2h"


class TestGetMeaningfulSummary:
    """Tests for meaningful summary extraction."""

    @pytest.fixture
    def tmp_session(self, tmp_path):
        """Create a temp session file from lines."""

        def _create(lines):
            path = tmp_path / "test.jsonl"
            path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
            return path

        return _create

    def test_extracts_first_user_message(self, tmp_session):
        path = tmp_session(
            [
                {
                    "type": "user",
                    "message": {"content": "Fix the login bug"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
            ]
        )
        assert get_meaningful_summary(path) == "Fix the login bug"

    def test_skips_xml_messages(self, tmp_session):
        path = tmp_session(
            [
                {
                    "type": "user",
                    "message": {"content": "<system-reminder>ignore</system-reminder>"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
                {
                    "type": "user",
                    "message": {"content": "Real question here"},
                    "timestamp": "2026-01-01T10:01:00.000Z",
                },
            ]
        )
        assert get_meaningful_summary(path) == "Real question here"

    def test_skips_interrupted_messages(self, tmp_session):
        path = tmp_session(
            [
                {
                    "type": "user",
                    "message": {
                        "content": "[Request interrupted by user for tool use]"
                    },
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
                {
                    "type": "user",
                    "message": {"content": "Add unit tests"},
                    "timestamp": "2026-01-01T10:01:00.000Z",
                },
            ]
        )
        assert get_meaningful_summary(path) == "Add unit tests"

    def test_skips_meta_messages(self, tmp_session):
        path = tmp_session(
            [
                {
                    "type": "user",
                    "isMeta": True,
                    "message": {"content": "meta stuff"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
                {
                    "type": "user",
                    "message": {"content": "Implement feature X"},
                    "timestamp": "2026-01-01T10:01:00.000Z",
                },
            ]
        )
        assert get_meaningful_summary(path) == "Implement feature X"

    def test_truncates_long_summary(self, tmp_session):
        long_text = "x" * 200
        path = tmp_session(
            [
                {
                    "type": "user",
                    "message": {"content": long_text},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
            ]
        )
        result = get_meaningful_summary(path, max_length=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_handles_content_array(self, tmp_session):
        path = tmp_session(
            [
                {
                    "type": "user",
                    "message": {
                        "content": [{"type": "text", "text": "Array content here"}]
                    },
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
            ]
        )
        assert get_meaningful_summary(path) == "Array content here"

    def test_returns_no_summary_for_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert get_meaningful_summary(path) == "(no summary)"

    def test_prefers_summary_type_entry(self, tmp_session):
        path = tmp_session(
            [
                {"type": "summary", "summary": "Session about fixing bugs"},
                {
                    "type": "user",
                    "message": {"content": "Fix the login bug"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
            ]
        )
        assert get_meaningful_summary(path) == "Session about fixing bugs"


class TestExtractRichMetadata:
    """Tests for full metadata extraction."""

    def _make_session(self, tmp_path, lines, name="test.jsonl"):
        path = tmp_path / "project-folder" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
        return path

    def test_extracts_basic_fields(self, tmp_path):
        path = self._make_session(
            tmp_path,
            [
                {
                    "type": "user",
                    "cwd": "/Users/fred/workspace/myproject",
                    "sessionId": "abc-123",
                    "gitBranch": "main",
                    "version": "2.1.17",
                    "slug": "happy-dancing-fox",
                    "message": {"content": "Hello world"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
                {
                    "type": "assistant",
                    "message": {
                        "model": "claude-opus-4-6",
                        "content": [{"type": "text", "text": "Hi!"}],
                    },
                    "timestamp": "2026-01-01T10:05:00.000Z",
                },
            ],
        )

        meta = extract_rich_metadata(path, "project-folder")

        assert meta.session_id == "abc-123"
        assert meta.cwd == "/Users/fred/workspace/myproject"
        assert meta.project_name == "myproject"
        assert meta.project_path == "project-folder"
        assert meta.git_branch == "main"
        assert meta.model == "claude-opus-4-6"
        assert meta.model_short == "opus-4.6"
        assert meta.slug == "happy-dancing-fox"
        assert meta.summary == "Hello world"
        assert meta.version == "2.1.17"
        assert meta.user_msg_count == 1
        assert meta.assistant_msg_count == 1
        assert meta.duration_minutes == 5

    def test_handles_missing_fields(self, tmp_path):
        path = self._make_session(
            tmp_path,
            [
                {
                    "type": "user",
                    "message": {"content": "Test"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
            ],
        )

        meta = extract_rich_metadata(path, "-Users-fred-projects-myproject")

        assert meta.session_id is None
        assert meta.model is None
        assert meta.model_short == ""
        assert meta.git_branch is None
        assert meta.slug is None
        # Falls back to folder name parsing (get_project_display_name)
        assert meta.project_name == "myproject"

    def test_counts_messages(self, tmp_path):
        path = self._make_session(
            tmp_path,
            [
                {
                    "type": "user",
                    "message": {"content": "Q1"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "A1"}]},
                    "timestamp": "2026-01-01T10:01:00.000Z",
                },
                {
                    "type": "user",
                    "message": {"content": "Q2"},
                    "timestamp": "2026-01-01T10:02:00.000Z",
                },
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "A2"}]},
                    "timestamp": "2026-01-01T10:03:00.000Z",
                },
                {
                    "type": "user",
                    "message": {"content": "Q3"},
                    "timestamp": "2026-01-01T10:04:00.000Z",
                },
            ],
        )

        meta = extract_rich_metadata(path, "folder")
        assert meta.user_msg_count == 3
        assert meta.assistant_msg_count == 2

    def test_skips_meta_messages_in_count(self, tmp_path):
        path = self._make_session(
            tmp_path,
            [
                {
                    "type": "user",
                    "isMeta": True,
                    "message": {"content": "meta"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
                {
                    "type": "user",
                    "message": {"content": "Real message"},
                    "timestamp": "2026-01-01T10:01:00.000Z",
                },
            ],
        )

        meta = extract_rich_metadata(path, "folder")
        assert meta.user_msg_count == 1

    def test_estimates_duration(self, tmp_path):
        path = self._make_session(
            tmp_path,
            [
                {
                    "type": "user",
                    "message": {"content": "Start"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "End"}]},
                    "timestamp": "2026-01-01T10:45:00.000Z",
                },
            ],
        )

        meta = extract_rich_metadata(path, "folder")
        assert meta.duration_minutes == 45

    def test_handles_empty_file(self, tmp_path):
        path = self._make_session(tmp_path, [])
        # Write empty file
        path.write_text("")

        meta = extract_rich_metadata(path, "folder")
        assert meta.summary == "(no summary)"
        assert meta.user_msg_count == 0
        assert meta.duration_minutes is None

    def test_slug_from_later_lines(self, tmp_path):
        """Slug may not appear on the first line."""
        path = self._make_session(
            tmp_path,
            [
                {
                    "type": "progress",
                    "data": {"type": "hook_progress"},
                    "timestamp": "2026-01-01T10:00:00.000Z",
                },
                {
                    "type": "user",
                    "slug": "fancy-slug-here",
                    "message": {"content": "Hello"},
                    "timestamp": "2026-01-01T10:01:00.000Z",
                },
            ],
        )

        meta = extract_rich_metadata(path, "folder")
        assert meta.slug == "fancy-slug-here"
