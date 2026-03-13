"""Tests for the explore CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from click.testing import CliRunner

from ccutils.cli import cli


def _patch_server():
    """Patch the ReusableTCPServer class inside the explore module.

    The explore command defines ReusableTCPServer locally inside the function,
    so we need to patch socketserver.TCPServer which it subclasses, and
    also catch the actual instantiation.
    """
    return patch(
        "ccutils.cli.explore.socketserver.TCPServer.__init__", return_value=None
    )


class TestExploreCommand:
    """Tests for the explore command."""

    def test_starts_server(self, mock_webbrowser_open):
        """Explore command starts HTTP server and shuts down on KeyboardInterrupt."""
        with (
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.__init__", return_value=None
            ),
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.serve_forever",
                side_effect=KeyboardInterrupt(),
            ),
            patch("ccutils.cli.explore.socketserver.TCPServer.shutdown"),
            patch("ccutils.cli.explore.socketserver.TCPServer.server_close"),
        ):

            runner = CliRunner()
            result = runner.invoke(cli, ["explore"])

            assert result.exit_code == 0
            assert "Data Explorer running at" in result.output

    def test_custom_port(self, mock_webbrowser_open):
        """--port option changes the port."""
        with (
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.__init__", return_value=None
            ),
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.serve_forever",
                side_effect=KeyboardInterrupt(),
            ),
            patch("ccutils.cli.explore.socketserver.TCPServer.shutdown"),
            patch("ccutils.cli.explore.socketserver.TCPServer.server_close"),
        ):

            runner = CliRunner()
            result = runner.invoke(cli, ["explore", "-p", "9999"])

            assert result.exit_code == 0
            assert "9999" in result.output

    def test_no_open_flag(self, mock_webbrowser_open):
        """--no-open prevents browser from opening."""
        with (
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.__init__", return_value=None
            ),
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.serve_forever",
                side_effect=KeyboardInterrupt(),
            ),
            patch("ccutils.cli.explore.socketserver.TCPServer.shutdown"),
            patch("ccutils.cli.explore.socketserver.TCPServer.server_close"),
        ):

            runner = CliRunner()
            result = runner.invoke(cli, ["explore", "--no-open"])

            assert result.exit_code == 0
            assert len(mock_webbrowser_open) == 0

    def test_browser_opens_by_default(self, mock_webbrowser_open):
        """Browser opens by default without --no-open."""
        with (
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.__init__", return_value=None
            ),
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.serve_forever",
                side_effect=KeyboardInterrupt(),
            ),
            patch("ccutils.cli.explore.socketserver.TCPServer.shutdown"),
            patch("ccutils.cli.explore.socketserver.TCPServer.server_close"),
        ):

            runner = CliRunner()
            result = runner.invoke(cli, ["explore"])

            assert result.exit_code == 0
            assert len(mock_webbrowser_open) == 1
            assert "index.html" in mock_webbrowser_open[0]

    def test_port_conflict_error(self):
        """Port already in use gives specific error message."""
        with patch(
            "ccutils.cli.explore.socketserver.TCPServer.__init__",
            side_effect=OSError(48, "Address already in use"),
        ):

            runner = CliRunner()
            result = runner.invoke(cli, ["explore", "-p", "8765"])

            assert result.exit_code != 0
            assert "Port 8765 is already in use" in result.output

    def test_database_argument(self, mock_webbrowser_open, tmp_path):
        """Database argument shows load path."""
        db_file = tmp_path / "test.duckdb"
        db_file.write_text("")

        with (
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.__init__", return_value=None
            ),
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.serve_forever",
                side_effect=KeyboardInterrupt(),
            ),
            patch("ccutils.cli.explore.socketserver.TCPServer.shutdown"),
            patch("ccutils.cli.explore.socketserver.TCPServer.server_close"),
        ):

            runner = CliRunner()
            result = runner.invoke(cli, ["explore", str(db_file)])

            assert result.exit_code == 0
            assert "Load database:" in result.output

    def test_graceful_shutdown_message(self, mock_webbrowser_open):
        """KeyboardInterrupt triggers server stopped message."""
        with (
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.__init__", return_value=None
            ),
            patch(
                "ccutils.cli.explore.socketserver.TCPServer.serve_forever",
                side_effect=KeyboardInterrupt(),
            ),
            patch("ccutils.cli.explore.socketserver.TCPServer.shutdown"),
            patch("ccutils.cli.explore.socketserver.TCPServer.server_close"),
        ):

            runner = CliRunner()
            result = runner.invoke(cli, ["explore"])

            assert result.exit_code == 0
            assert "Server stopped" in result.output
