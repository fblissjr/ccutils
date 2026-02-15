"""Tests for the tui package: theme, formatters, layout, components, selection."""

import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console


# ---------------------------------------------------------------------------
# theme.py tests
# ---------------------------------------------------------------------------


class TestTheme:
    def test_styles_dict_has_required_roles(self):
        from ccutils.tui.theme import STYLES

        for role in ("temporal", "identity", "model", "metric", "primary", "secondary"):
            assert role in STYLES

    def test_rich_styles_dict_has_required_roles(self):
        from ccutils.tui.theme import RICH_STYLES

        for role in ("temporal", "identity", "model", "metric", "primary", "secondary"):
            assert role in RICH_STYLES

    def test_model_style_key_opus(self):
        from ccutils.tui.theme import model_style_key

        assert model_style_key("opus-4.6") == "model.opus"

    def test_model_style_key_sonnet(self):
        from ccutils.tui.theme import model_style_key

        assert model_style_key("sonnet-4.5") == "model.sonnet"

    def test_model_style_key_haiku(self):
        from ccutils.tui.theme import model_style_key

        assert model_style_key("haiku-4.5") == "model.haiku"

    def test_model_style_key_unknown(self):
        from ccutils.tui.theme import model_style_key

        assert model_style_key("unknown-model") == "model"

    def test_model_style_key_case_insensitive(self):
        from ccutils.tui.theme import model_style_key

        assert model_style_key("Opus-4.6") == "model.opus"

    def test_questionary_style_returns_style_object(self):
        from ccutils.tui.theme import questionary_style
        from prompt_toolkit.styles import Style

        style = questionary_style()
        assert isinstance(style, Style)


# ---------------------------------------------------------------------------
# formatters.py tests
# ---------------------------------------------------------------------------


class TestFormatRelativeDate:
    def test_today(self):
        from ccutils.tui.formatters import format_relative_date

        now = datetime.now()
        mtime = now.timestamp()
        result = format_relative_date(mtime)
        assert result.startswith("Today ")
        assert now.strftime("%H:%M") in result

    def test_yesterday(self):
        from ccutils.tui.formatters import format_relative_date

        yesterday = datetime.now() - timedelta(days=1)
        result = format_relative_date(yesterday.timestamp())
        assert result.startswith("Yest ")

    def test_this_week(self):
        from ccutils.tui.formatters import format_relative_date

        three_days_ago = datetime.now() - timedelta(days=3)
        result = format_relative_date(three_days_ago.timestamp())
        # Should show day name like "Mon 16:00"
        day_abbrevs = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
        assert any(result.startswith(d) for d in day_abbrevs)

    def test_older(self):
        from ccutils.tui.formatters import format_relative_date

        old = datetime.now() - timedelta(days=30)
        result = format_relative_date(old.timestamp())
        # Should show "Feb 10" style
        assert len(result) <= 6  # "Feb 10"
        assert ":" not in result  # no time component


class TestFormatRelativeDateShort:
    def test_today(self):
        from ccutils.tui.formatters import format_relative_date_short

        assert format_relative_date_short(datetime.now().timestamp()) == "Today"

    def test_yesterday(self):
        from ccutils.tui.formatters import format_relative_date_short

        yesterday = datetime.now() - timedelta(days=1)
        assert format_relative_date_short(yesterday.timestamp()) == "Yesterday"


class TestFormatDuration:
    def test_none(self):
        from ccutils.tui.formatters import format_duration

        assert format_duration(None) == ""

    def test_zero(self):
        from ccutils.tui.formatters import format_duration

        assert format_duration(0) == "<1m"

    def test_minutes(self):
        from ccutils.tui.formatters import format_duration

        assert format_duration(45) == "45m"

    def test_hours_and_minutes(self):
        from ccutils.tui.formatters import format_duration

        assert format_duration(125) == "2h 5m"

    def test_exact_hours(self):
        from ccutils.tui.formatters import format_duration

        assert format_duration(120) == "2h"


class TestFormatProjectName:
    def test_short_name(self):
        from ccutils.tui.formatters import format_project_name

        assert format_project_name("ccutils") == "ccutils"

    def test_exact_width(self):
        from ccutils.tui.formatters import format_project_name

        name = "a" * 20
        assert format_project_name(name, max_width=20) == name

    def test_truncation(self):
        from ccutils.tui.formatters import format_project_name

        name = "a-very-long-project-name"
        result = format_project_name(name, max_width=15)
        assert len(result) == 15
        assert result.endswith("..")


class TestFormatSummary:
    def test_short_summary(self):
        from ccutils.tui.formatters import format_summary

        assert format_summary("hello world", 50) == "hello world"

    def test_truncation(self):
        from ccutils.tui.formatters import format_summary

        long_text = "a" * 100
        result = format_summary(long_text, 30)
        assert len(result) == 30
        assert result.endswith("...")

    def test_minimum_width(self):
        from ccutils.tui.formatters import format_summary

        result = format_summary("hello world test", 5)
        # Should clamp to 10 minimum
        assert len(result) == 10


class TestFormatBranch:
    def test_none(self):
        from ccutils.tui.formatters import format_branch

        assert format_branch(None) == "-"

    def test_short_branch(self):
        from ccutils.tui.formatters import format_branch

        assert format_branch("main") == "main"

    def test_truncation(self):
        from ccutils.tui.formatters import format_branch

        result = format_branch("feat/very-long-branch-name", max_width=12)
        assert len(result) == 12
        assert result.endswith("..")


class TestFormatSize:
    def test_megabytes(self):
        from ccutils.tui.formatters import format_size

        assert format_size(1_500_000) == "1.4 MB"

    def test_kilobytes(self):
        from ccutils.tui.formatters import format_size

        assert format_size(450_000) == "439 KB"

    def test_bytes(self):
        from ccutils.tui.formatters import format_size

        assert format_size(800) == "800 B"


class TestFormatMsgCount:
    def test_normal(self):
        from ccutils.tui.formatters import format_msg_count

        assert format_msg_count(12, 8) == "12/8"

    def test_no_messages(self):
        from ccutils.tui.formatters import format_msg_count

        assert format_msg_count(0, 0) == "-"


# ---------------------------------------------------------------------------
# layout.py tests
# ---------------------------------------------------------------------------


class TestGetTerminalWidth:
    def test_returns_int(self):
        from ccutils.tui.layout import get_terminal_width

        width = get_terminal_width()
        assert isinstance(width, int)
        assert width > 0

    def test_fallback_on_error(self):
        from ccutils.tui.layout import get_terminal_width

        with patch(
            "ccutils.tui.layout.shutil.get_terminal_size", side_effect=ValueError
        ):
            assert get_terminal_width() == 80


class TestColumnSpec:
    def test_defaults(self):
        from ccutils.tui.layout import ColumnSpec

        col = ColumnSpec(name="test", min_width=10)
        assert col.max_width == 0
        assert col.ratio == 1.0
        assert col.fixed is False


class TestCalculateColumnWidths:
    def test_fixed_columns(self):
        from ccutils.tui.layout import ColumnSpec, calculate_column_widths

        cols = [
            ColumnSpec(name="a", min_width=10, fixed=True),
            ColumnSpec(name="b", min_width=10, fixed=True),
        ]
        result = calculate_column_widths(cols, total_width=30, padding=2)
        assert result["a"] == 10
        assert result["b"] == 10

    def test_flex_columns_share_space(self):
        from ccutils.tui.layout import ColumnSpec, calculate_column_widths

        cols = [
            ColumnSpec(name="a", min_width=5, ratio=1.0),
            ColumnSpec(name="b", min_width=5, ratio=1.0),
        ]
        result = calculate_column_widths(cols, total_width=30, padding=2)
        # 30 - 2 padding = 28, split evenly = 14 each
        assert result["a"] == 14
        assert result["b"] == 14

    def test_mixed_fixed_and_flex(self):
        from ccutils.tui.layout import ColumnSpec, calculate_column_widths

        cols = [
            ColumnSpec(name="num", min_width=3, fixed=True),
            ColumnSpec(name="date", min_width=10, fixed=True),
            ColumnSpec(name="summary", min_width=20, ratio=1.0),
        ]
        result = calculate_column_widths(cols, total_width=80, padding=2)
        assert result["num"] == 3
        assert result["date"] == 10
        # 80 - 4 (padding*2) - 3 - 10 = 63 for summary
        assert result["summary"] == 63

    def test_max_width_respected(self):
        from ccutils.tui.layout import ColumnSpec, calculate_column_widths

        cols = [
            ColumnSpec(name="a", min_width=5, max_width=15, ratio=1.0),
        ]
        result = calculate_column_widths(cols, total_width=100, padding=0)
        assert result["a"] == 15

    def test_narrow_terminal(self):
        from ccutils.tui.layout import ColumnSpec, calculate_column_widths

        cols = [
            ColumnSpec(name="a", min_width=10, ratio=1.0),
            ColumnSpec(name="b", min_width=10, ratio=1.0),
        ]
        # Very narrow: each should get at least min_width
        result = calculate_column_widths(cols, total_width=15, padding=2)
        assert result["a"] >= 10
        assert result["b"] >= 10


# ---------------------------------------------------------------------------
# components.py tests
# ---------------------------------------------------------------------------


def _make_session_meta(**kwargs):
    """Create a minimal SessionMetadata for testing."""
    from ccutils.parsers.metadata import SessionMetadata

    defaults = {
        "path": Path("/tmp/test.jsonl"),
        "project_name": "test-project",
        "project_path": "-Users-test-workspace-test-project",
        "model_short": "opus-4.6",
        "git_branch": "main",
        "summary": "Test session summary",
        "mtime": time.time(),
        "size": 50000,
        "user_msg_count": 5,
        "assistant_msg_count": 3,
        "duration_minutes": 45,
        "slug": None,
    }
    defaults.update(kwargs)
    return SessionMetadata(**defaults)


class TestRenderProjectTable:
    def test_renders_without_error(self):
        from ccutils.tui.components import render_project_table

        sessions = [_make_session_meta()]
        grouped = {sessions[0].project_path: sessions}
        buf = StringIO()
        console = Console(file=buf, width=120, force_terminal=True)
        render_project_table(grouped, console=console)
        output = buf.getvalue()
        assert "test-project" in output
        assert "1" in output  # session count

    def test_shows_title_with_counts(self):
        from ccutils.tui.components import render_project_table

        s1 = _make_session_meta(project_name="proj-a")
        s2 = _make_session_meta(
            project_name="proj-b",
            project_path="-Users-test-workspace-proj-b",
        )
        grouped = {s1.project_path: [s1], s2.project_path: [s2]}
        buf = StringIO()
        console = Console(file=buf, width=120, force_terminal=True)
        render_project_table(grouped, console=console)
        output = buf.getvalue()
        assert "2 found" in output
        assert "2 sessions" in output

    def test_shows_models_and_branches(self):
        from ccutils.tui.components import render_project_table

        sessions = [
            _make_session_meta(model_short="opus-4.6", git_branch="main"),
            _make_session_meta(model_short="sonnet-4.5", git_branch="dev"),
        ]
        grouped = {sessions[0].project_path: sessions}
        buf = StringIO()
        console = Console(file=buf, width=120, force_terminal=True)
        render_project_table(grouped, console=console)
        output = buf.getvalue()
        assert "opus-4.6" in output
        assert "sonnet-4.5" in output
        assert "main" in output
        assert "dev" in output


class TestRenderSessionTable:
    def test_renders_without_error(self):
        from ccutils.tui.components import render_session_table

        sessions = [_make_session_meta()]
        buf = StringIO()
        console = Console(file=buf, width=120, force_terminal=True)
        render_session_table("test-project", sessions, console=console)
        output = buf.getvalue()
        assert "test-project" in output
        assert "Test session summary" in output

    def test_shows_duration(self):
        from ccutils.tui.components import render_session_table

        sessions = [_make_session_meta(duration_minutes=125)]
        buf = StringIO()
        console = Console(file=buf, width=120, force_terminal=True)
        render_session_table("proj", sessions, console=console)
        output = buf.getvalue()
        assert "2h 5m" in output

    def test_shows_msg_count(self):
        from ccutils.tui.components import render_session_table

        sessions = [_make_session_meta(user_msg_count=12, assistant_msg_count=8)]
        buf = StringIO()
        console = Console(file=buf, width=120, force_terminal=True)
        render_session_table("proj", sessions, console=console)
        output = buf.getvalue()
        assert "12/8" in output


class TestRenderStatusHeader:
    def test_renders_counts(self):
        from ccutils.tui.components import render_status_header

        buf = StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        render_status_header(42, 5, console=console)
        output = buf.getvalue()
        assert "42" in output
        assert "5" in output
