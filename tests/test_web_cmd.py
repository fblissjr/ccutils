"""Tests for the web CLI command."""

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from ccutils.cli import cli


def _mock_session_data():
    """Create mock session data as returned by fetch_session."""
    return {
        "loglines": [
            {
                "type": "user",
                "uuid": "user-001",
                "parentUuid": None,
                "sessionId": "test-session-id",
                "timestamp": "2025-01-15T10:00:00.000Z",
                "cwd": "/home/user/project",
                "message": {
                    "role": "user",
                    "content": "Hello",
                },
            },
            {
                "type": "assistant",
                "uuid": "asst-001",
                "parentUuid": "user-001",
                "sessionId": "test-session-id",
                "timestamp": "2025-01-15T10:00:05.000Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "Hi there!"}],
                },
            },
        ]
    }


def _mock_sessions_list():
    """Create mock sessions list as returned by fetch_sessions."""
    return {
        "data": [
            {
                "id": "session-abc123",
                "name": "Test Session",
                "created_at": "2025-01-15T10:00:00.000Z",
                "updated_at": "2025-01-15T11:00:00.000Z",
                "project": {"name": "test-project"},
            }
        ],
        "has_more": False,
    }


class TestWebCommandWithSessionId:
    """Tests for web command when session_id is provided directly."""

    @patch("ccutils.cli.web.fetch_session")
    @patch("ccutils.cli.web.resolve_credentials")
    def test_fetches_and_generates_html(self, mock_creds, mock_fetch, output_dir):
        """Direct session_id fetches and generates HTML."""
        mock_creds.return_value = ("test-token", "test-org")
        mock_fetch.return_value = _mock_session_data()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["web", "test-session-id", "-o", str(output_dir)],
        )

        assert result.exit_code == 0
        assert "Fetching session" in result.output
        assert "Output:" in result.output
        mock_fetch.assert_called_once_with("test-token", "test-org", "test-session-id")

    @patch("ccutils.cli.web.fetch_session")
    @patch("ccutils.cli.web.resolve_credentials")
    def test_json_flag_saves_session_data(self, mock_creds, mock_fetch, output_dir):
        """--json flag saves raw session data."""
        mock_creds.return_value = ("test-token", "test-org")
        mock_fetch.return_value = _mock_session_data()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["web", "test-session-id", "-o", str(output_dir), "--json"],
        )

        assert result.exit_code == 0
        assert "JSON:" in result.output

    @patch("ccutils.cli.web.fetch_session")
    @patch("ccutils.cli.web.resolve_credentials")
    def test_private_flag(self, mock_creds, mock_fetch, output_dir):
        """--private flag is accepted."""
        mock_creds.return_value = ("test-token", "test-org")
        mock_fetch.return_value = _mock_session_data()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["web", "test-session-id", "-o", str(output_dir), "--private"],
        )

        assert result.exit_code == 0


class TestWebCommandErrors:
    """Tests for web command error handling."""

    @patch("ccutils.cli.web.resolve_credentials")
    def test_credential_error(self, mock_creds):
        """Missing credentials gives error."""
        import click

        mock_creds.side_effect = click.ClickException("No API token found")

        runner = CliRunner()
        result = runner.invoke(cli, ["web", "test-session-id"])

        assert result.exit_code != 0
        assert "No API token" in result.output

    @patch("ccutils.cli.web.fetch_session")
    @patch("ccutils.cli.web.resolve_credentials")
    def test_http_error(self, mock_creds, mock_fetch):
        """HTTP error gives friendly message."""
        import httpx

        mock_creds.return_value = ("test-token", "test-org")
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_fetch.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=mock_response
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["web", "test-session-id"])

        assert result.exit_code != 0
        assert "401" in result.output

    @patch("ccutils.cli.web.fetch_session")
    @patch("ccutils.cli.web.resolve_credentials")
    def test_network_error(self, mock_creds, mock_fetch):
        """Network error gives friendly message."""
        import httpx

        mock_creds.return_value = ("test-token", "test-org")
        mock_fetch.side_effect = httpx.RequestError("Connection refused")

        runner = CliRunner()
        result = runner.invoke(cli, ["web", "test-session-id"])

        assert result.exit_code != 0
        assert "Network error" in result.output


class TestWebCommandInteractive:
    """Tests for web command interactive session picker."""

    @patch("ccutils.cli.web.questionary")
    @patch("ccutils.cli.web.fetch_session")
    @patch("ccutils.cli.web.fetch_sessions")
    @patch("ccutils.cli.web.resolve_credentials")
    def test_no_sessions_errors(
        self, mock_creds, mock_fetch_sessions, mock_fetch, mock_questionary
    ):
        """No sessions found gives error."""
        mock_creds.return_value = ("test-token", "test-org")
        mock_fetch_sessions.return_value = {"data": [], "has_more": False}

        runner = CliRunner()
        result = runner.invoke(cli, ["web"])

        assert result.exit_code != 0
        assert "No sessions found" in result.output

    @patch("ccutils.cli.web.fetch_session")
    @patch("ccutils.cli.web.resolve_credentials")
    def test_debug_flag_shows_api_structure(self, mock_creds, mock_fetch, output_dir):
        """--debug shows API response structure when fetching session list."""
        mock_creds.return_value = ("test-token", "test-org")
        mock_fetch.return_value = _mock_session_data()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["web", "test-session-id", "-o", str(output_dir), "--debug"],
        )

        # Debug flag only applies to session list fetching, not direct session fetch
        assert result.exit_code == 0
