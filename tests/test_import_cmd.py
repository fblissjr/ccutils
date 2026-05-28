"""Tests for the import CLI command."""

import json
from pathlib import Path

from click.testing import CliRunner

from ccutils.cli import cli


def _create_mock_export(tmp_path, num_conversations=2):
    """Create a mock Claude.ai export directory structure.

    Returns the export directory path.
    """
    export_dir = tmp_path / "claude-export"
    export_dir.mkdir()

    conversations = []
    for i in range(num_conversations):
        conv = {
            "uuid": f"conv-{i:04d}-uuid-placeholder-value",
            "name": f"Test Conversation {i}",
            "created_at": f"2025-01-{15 + i:02d}T10:00:00.000Z",
            "updated_at": f"2025-01-{15 + i:02d}T11:00:00.000Z",
            "chat_messages": [
                {
                    "uuid": f"msg-{i}-001",
                    "text": f"Hello from conversation {i}",
                    "sender": "human",
                    "created_at": f"2025-01-{15 + i:02d}T10:00:00.000Z",
                    "updated_at": f"2025-01-{15 + i:02d}T10:00:00.000Z",
                    "content": [
                        {"type": "text", "text": f"Hello from conversation {i}"}
                    ],
                },
                {
                    "uuid": f"msg-{i}-002",
                    "text": f"Response to conversation {i}",
                    "sender": "assistant",
                    "created_at": f"2025-01-{15 + i:02d}T10:00:05.000Z",
                    "updated_at": f"2025-01-{15 + i:02d}T10:00:05.000Z",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Response to conversation {i}",
                        }
                    ],
                },
            ],
        }
        conversations.append(conv)

    (export_dir / "conversations.json").write_text(json.dumps(conversations))
    (export_dir / "projects.json").write_text(json.dumps([]))

    return export_dir


class TestImportCommandValidation:
    """Tests for import command input validation."""

    def test_missing_conversations_json_errors(self, tmp_path):
        """Error when conversations.json is missing."""
        empty_dir = tmp_path / "empty-export"
        empty_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["import", str(empty_dir)])

        assert result.exit_code != 0
        assert "conversations.json not found" in result.output


class TestImportCommandList:
    """Tests for import --list mode."""

    def test_list_mode_shows_conversations(self, tmp_path):
        """--list shows conversations without converting."""
        export_dir = _create_mock_export(tmp_path, num_conversations=3)

        runner = CliRunner()
        result = runner.invoke(cli, ["import", str(export_dir), "--list"])

        assert result.exit_code == 0
        assert "3 conversations" in result.output
        assert "Test Conversation 0" in result.output

    def test_list_mode_empty_export(self, tmp_path):
        """--list with empty conversations gives message."""
        export_dir = tmp_path / "empty-export"
        export_dir.mkdir()
        (export_dir / "conversations.json").write_text("[]")

        runner = CliRunner()
        result = runner.invoke(cli, ["import", str(export_dir), "--list"])

        assert result.exit_code == 0
        assert "No conversations found" in result.output


class TestImportCommandHTML:
    """Tests for import HTML export."""

    def test_html_export(self, tmp_path):
        """HTML export creates output files."""
        export_dir = _create_mock_export(tmp_path, num_conversations=1)
        output = tmp_path / "html-output"

        runner = CliRunner()
        result = runner.invoke(cli, ["import", str(export_dir), "-o", str(output)])

        assert result.exit_code == 0
        assert "Generated:" in result.output
        assert output.exists()

    def test_html_export_multiple_sessions(self, tmp_path):
        """HTML export with multiple conversations creates index."""
        export_dir = _create_mock_export(tmp_path, num_conversations=3)
        output = tmp_path / "html-output"

        runner = CliRunner()
        result = runner.invoke(cli, ["import", str(export_dir), "-o", str(output)])

        assert result.exit_code == 0
        assert output.exists()

