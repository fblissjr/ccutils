"""The default archive output location must resolve outside any worktree.

Claim, shared by every case here: an archive this tool generates holds
unredacted transcripts for EVERY project on the machine, so where it lands
when the user passes no `-o` is a privacy decision, not a convenience one.
A cwd-relative default (`./claude-archive`) writes that data into whatever
checkout the command happened to run in -- protected, if at all, by a single
.gitignore line in a repo that may be public.

Delete these and the default is free to drift back to cwd with nothing
watching. Each case pins one entry point (`all`, `local`) plus the helper
they share, and asserts BOTH halves: the archive is absent from cwd AND
present under the home-anchored default, so "absent from cwd" cannot pass
because the export silently did nothing.
"""

from pathlib import Path

import pytest
import questionary
from click.testing import CliRunner

from ccutils import cli
from ccutils.cli.utils import default_archive_output


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() so the default resolves inside the test sandbox."""
    home = tmp_path / "fake-home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    return home


class TestDefaultArchiveOutput:
    """The helper both CLI entry points resolve their default through."""

    def test_default_is_absolute_and_home_anchored(self, fake_home):
        """A relative default is the defect: it resolves against cwd, which
        is a repo worktree whenever the tool is run from one."""
        default = default_archive_output()
        assert default.is_absolute()
        assert fake_home in default.parents

    def test_default_is_independent_of_cwd(self, fake_home):
        """Resolved from home, not from wherever the process was started.

        Compared after ``resolve()``: a relative default compares equal as a
        Path object no matter the cwd, so the naive comparison cannot see
        the defect this pins.
        """
        runner = CliRunner()
        with runner.isolated_filesystem():
            from_sandbox = default_archive_output().resolve()
        assert from_sandbox == default_archive_output().resolve()


class TestAllCommandDefaultOutput:
    """`ccutils --source` with no -o."""

    def test_archive_lands_under_home_not_cwd(self, mock_projects_dir, fake_home):
        runner = CliRunner()
        with runner.isolated_filesystem() as sandbox:
            result = runner.invoke(
                cli, ["--source", str(mock_projects_dir)]
            )
            assert result.exit_code == 0, result.output
            assert not (Path(sandbox) / "claude-archive").exists()

        archive = default_archive_output()
        assert (archive / "index.html").exists()

    def test_explicit_output_still_wins(self, mock_projects_dir, fake_home, tmp_path):
        """The override is the escape hatch; moving the default must not
        take it away."""
        explicit = tmp_path / "explicit-archive"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--source", str(mock_projects_dir), "-o", str(explicit)],
        )
        assert result.exit_code == 0, result.output
        assert (explicit / "index.html").exists()
        assert not default_archive_output().exists()


class TestLocalPickerDefaultOutput:
    """Picker mode with no -o -- the other default site."""

    @pytest.fixture
    def picked_session(self, fake_home, monkeypatch):
        """One selectable session under the fake home, auto-selected."""
        project = fake_home / ".claude" / "projects" / "test-project"
        project.mkdir(parents=True)
        session = project / "session-123.jsonl"
        session.write_text(
            '{"type":"summary","summary":"Test session"}\n'
            '{"type":"user","timestamp":"2025-01-01T00:00:00Z",'
            '"message":{"role":"user","content":"Hello"}}\n'
        )

        class MockCheckbox:
            def __init__(self, *args, **kwargs):
                pass

            def ask(self):
                return [session]

        monkeypatch.setattr(questionary, "checkbox", MockCheckbox)
        return session

    def test_archive_lands_under_home_not_cwd(self, picked_session, fake_home):
        runner = CliRunner()
        with runner.isolated_filesystem() as sandbox:
            result = runner.invoke(cli, [])
            assert result.exit_code == 0, result.output
            assert not (Path(sandbox) / "claude-archive").exists()

        archive = default_archive_output()
        assert list(archive.glob("*.html"))
