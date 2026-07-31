# path-privacy: skip-file -- references universal Claude Code data paths (not personal)
"""Tests for find_local_sessions_rich's temp-dir session exclusion.

find_all_sessions coverage lives in test_all.py; this file covers the
picker-facing rich discovery path, which was previously untested."""

import tempfile
from pathlib import Path

from ccutils.parsers.discovery import find_local_sessions_rich, is_temp_dir_cwd


class TestIsTempDirCwd:
    def test_none_or_empty_is_not_temp(self):
        assert is_temp_dir_cwd(None) is False
        assert is_temp_dir_cwd("") is False

    def test_real_project_path_is_not_temp(self):
        assert is_temp_dir_cwd("/home/user/projects/real-project") is False

    def test_tmp_prefix_is_temp(self):
        assert is_temp_dir_cwd("/tmp/whatever") is True
        assert is_temp_dir_cwd("/tmp") is True

    def test_private_tmp_prefix_is_temp(self):
        assert is_temp_dir_cwd(
            "/private/tmp/claude-501/abc/scratchpad/evalroot/_run_xyz"
        ) is True

    def test_var_folders_prefix_is_temp(self):
        assert is_temp_dir_cwd("/var/folders/y1/abc/T/foo") is True

    def test_private_var_folders_prefix_is_temp(self):
        """/var is itself a macOS symlink to /private/var (same pattern as
        /tmp -> /private/tmp) -- the OS reports the fully-resolved form as
        cwd, so a real session's temp path looks like
        /private/var/folders/... not /var/folders/..."""
        assert is_temp_dir_cwd(
            "/private/var/folders/y1/abc/T/tmp.X8fVkhGU7w"
        ) is True

    def test_path_merely_containing_tmp_substring_is_not_temp(self):
        """A project literally named .../my-tmp-experiments/thing must not
        false-positive -- only a real /tmp prefix counts."""
        assert is_temp_dir_cwd("/home/user/my-tmp-experiments/thing") is False


class TestFindLocalSessionsRichTempDirExclusion:
    def _write(self, path, session_id, cwd):
        path.write_text(
            '{"type": "user", "sessionId": "%s", "cwd": "%s", '
            '"timestamp": "2025-01-01T10:00:00.000Z", '
            '"message": {"role": "user", "content": "hello"}}\n'
            % (session_id, cwd)
        )

    def test_excludes_temp_dir_sessions_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir)

            real = projects_dir / "-home-user-projects-real-project"
            real.mkdir(parents=True)
            self._write(real / "abc.jsonl", "s1", "/home/user/projects/real-project")

            sandbox = projects_dir / "-private-tmp-claude-501-scratchpad-run-xyz"
            sandbox.mkdir(parents=True)
            self._write(
                sandbox / "def.jsonl", "s2",
                "/private/tmp/claude-501/scratchpad/run_xyz",
            )

            results = find_local_sessions_rich(projects_dir)
            assert len(results) == 1
            assert results[0].session_id == "s1"

    def test_includes_temp_dir_sessions_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            projects_dir = Path(tmpdir)

            real = projects_dir / "-home-user-projects-real-project"
            real.mkdir(parents=True)
            self._write(real / "abc.jsonl", "s1", "/home/user/projects/real-project")

            sandbox = projects_dir / "-private-tmp-claude-501-scratchpad-run-xyz"
            sandbox.mkdir(parents=True)
            self._write(
                sandbox / "def.jsonl", "s2",
                "/private/tmp/claude-501/scratchpad/run_xyz",
            )

            results = find_local_sessions_rich(projects_dir, include_temp_sessions=True)
            assert len(results) == 2
