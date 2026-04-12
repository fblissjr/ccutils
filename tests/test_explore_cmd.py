"""Tests for the explore CLI command (harlequin shim)."""

from unittest.mock import patch

from click.testing import CliRunner

from ccutils.cli import cli


class TestExploreCommand:
    """Tests for the explore command."""

    def test_no_database_argument_shows_error(self):
        """Explore command requires a database argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["explore"])

        assert result.exit_code != 0

    def test_missing_database_file_shows_error(self):
        """Explore command errors if database file doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(cli, ["explore", "/nonexistent/path.duckdb"])

        assert result.exit_code != 0

    def test_runs_harlequin_when_installed(self, tmp_path):
        """Explore command execs harlequin with the database path."""
        db_file = tmp_path / "test.duckdb"
        db_file.write_bytes(b"")

        with patch("ccutils.cli.explore.subprocess.run") as mock_run:
            mock_run.return_value = None
            runner = CliRunner()
            result = runner.invoke(cli, ["explore", str(db_file)])

            assert result.exit_code == 0
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "harlequin"
            assert str(db_file) in args

    def test_shows_install_hint_when_harlequin_missing(self, tmp_path):
        """Explore command shows install instructions if harlequin not found."""
        db_file = tmp_path / "test.duckdb"
        db_file.write_bytes(b"")

        with patch(
            "ccutils.cli.explore.subprocess.run",
            side_effect=FileNotFoundError("harlequin not found"),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["explore", str(db_file)])

            assert result.exit_code != 0
            assert "uv pip install" in result.output or "ccutils[explore]" in result.output
