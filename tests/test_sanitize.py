# path-privacy: skip-file -- generic placeholders only
"""Tests for PathSanitizer privacy-preserving path sanitization.

The PathSanitizer unit tests below cover the sanitization logic itself.
Integration tests that exercised sanitization through the simple-schema
ETL were removed when the simple schema was dropped; the `--private`
flag's wiring through the v0.15 ETL is verified separately by the ETL
test suite.
"""

from ccutils.sanitize import PathSanitizer


class TestPathSanitizer:
    """Core path sanitization behavior."""

    def test_paths_under_cwd_become_relative(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        assert (
            san.sanitize_path("/Users/dev/workspace/project/src/main.py")
            == "src/main.py"
        )

    def test_deeply_nested_path_under_cwd(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        result = san.sanitize_path("/Users/dev/workspace/project/a/b/c/d.txt")
        assert result == "a/b/c/d.txt"

    def test_cwd_itself_becomes_dot(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        assert san.sanitize_path("/Users/dev/workspace/project") == "."

    def test_cwd_with_trailing_slash(self):
        san = PathSanitizer("/Users/dev/workspace/project/")
        assert (
            san.sanitize_path("/Users/dev/workspace/project/src/main.py")
            == "src/main.py"
        )

    def test_home_dir_paths_get_tilde(self):
        san = PathSanitizer("/Users/dev/workspace/project", home_dir="/Users/dev")
        result = san.sanitize_path("/Users/dev/.claude/settings.json")
        assert result == "~/.claude/settings.json"

    def test_home_dir_itself_becomes_tilde(self):
        san = PathSanitizer("/Users/dev/workspace/project", home_dir="/Users/dev")
        assert san.sanitize_path("/Users/dev") == "~"

    def test_system_paths_unchanged(self):
        san = PathSanitizer("/Users/dev/workspace/project", home_dir="/Users/dev")
        assert san.sanitize_path("/usr/local/bin/python") == "/usr/local/bin/python"

    def test_system_paths_var(self):
        san = PathSanitizer("/Users/dev/workspace/project", home_dir="/Users/dev")
        assert san.sanitize_path("/var/log/syslog") == "/var/log/syslog"

    def test_none_cwd_is_noop(self):
        san = PathSanitizer(None)
        path = "/Users/dev/workspace/project/src/main.py"
        assert san.sanitize_path(path) == path

    def test_none_cwd_text_is_noop(self):
        san = PathSanitizer(None)
        text = "Reading /Users/dev/workspace/project/src/main.py"
        assert san.sanitize_text(text) == text

    def test_empty_string_path(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        assert san.sanitize_path("") == ""

    def test_none_path(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        assert san.sanitize_path(None) is None

    def test_cwd_priority_over_home(self):
        """Paths under cwd should be made relative, not ~/..."""
        san = PathSanitizer("/Users/dev/workspace/project", home_dir="/Users/dev")
        result = san.sanitize_path("/Users/dev/workspace/project/src/main.py")
        assert result == "src/main.py"
        assert not result.startswith("~")


class TestSanitizeCwd:
    """sanitize_cwd returns project directory name."""

    def test_returns_dot(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        assert san.sanitize_cwd() == "."

    def test_none_cwd_returns_none(self):
        san = PathSanitizer(None)
        assert san.sanitize_cwd() is None


class TestSanitizeProjectPath:
    """sanitize_project_path strips to basename."""

    def test_strips_to_basename(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        result = san.sanitize_project_path(
            "/Users/dev/.claude/projects/-Users-fred-workspace-project"
        )
        assert result == "-Users-fred-workspace-project"

    def test_none_cwd_returns_original(self):
        san = PathSanitizer(None)
        original = "/Users/dev/.claude/projects/-Users-fred-workspace-project"
        assert san.sanitize_project_path(original) == original

    def test_none_path_returns_none(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        assert san.sanitize_project_path(None) is None

    def test_simple_path(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        assert san.sanitize_project_path("/some/path/session.jsonl") == "session.jsonl"


class TestSanitizeText:
    """sanitize_text does ordered string replacement in free text."""

    def test_replaces_cwd_in_commands(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        text = "cd /Users/dev/workspace/project && ls"
        assert san.sanitize_text(text) == "cd . && ls"

    def test_replaces_cwd_slash_prefix(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        text = "cat /Users/dev/workspace/project/README.md"
        assert san.sanitize_text(text) == "cat README.md"

    def test_replaces_home_dir(self):
        san = PathSanitizer("/Users/dev/workspace/project", home_dir="/Users/dev")
        text = "cat /Users/dev/.bashrc"
        assert san.sanitize_text(text) == "cat ~/.bashrc"

    def test_cwd_replaced_before_home(self):
        """cwd/ is more specific and should be replaced first."""
        san = PathSanitizer("/Users/dev/workspace/project", home_dir="/Users/dev")
        text = "Reading /Users/dev/workspace/project/src/main.py"
        result = san.sanitize_text(text)
        assert result == "Reading src/main.py"
        assert "~" not in result

    def test_multiple_paths_in_text(self):
        san = PathSanitizer("/Users/dev/workspace/project", home_dir="/Users/dev")
        text = (
            "Editing /Users/dev/workspace/project/a.py and "
            "/Users/dev/.config/settings.json"
        )
        result = san.sanitize_text(text)
        assert result == "Editing a.py and ~/.config/settings.json"

    def test_none_text(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        assert san.sanitize_text(None) is None


class TestSanitizeJsonString:
    """sanitize_json_string applies text replacement to JSON strings."""

    def test_replaces_paths_in_json(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        json_str = '{"file_path": "/Users/dev/workspace/project/src/main.py"}'
        result = san.sanitize_json_string(json_str)
        assert '"file_path": "src/main.py"' in result

    def test_replaces_home_in_json(self):
        san = PathSanitizer("/Users/dev/workspace/project", home_dir="/Users/dev")
        json_str = '{"path": "/Users/dev/.claude/config"}'
        result = san.sanitize_json_string(json_str)
        assert '"path": "~/.claude/config"' in result

    def test_none_json_returns_none(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        assert san.sanitize_json_string(None) is None

    def test_none_cwd_noop(self):
        san = PathSanitizer(None)
        json_str = '{"file_path": "/Users/dev/workspace/project/src/main.py"}'
        assert san.sanitize_json_string(json_str) == json_str


class TestAutoDetectHome:
    """When home_dir is not provided, auto-detect from cwd."""

    def test_auto_detect_from_users_prefix(self):
        san = PathSanitizer("/Users/dev/workspace/project")
        # Should auto-detect /Users/dev as home
        result = san.sanitize_path("/Users/dev/.claude/settings.json")
        assert result == "~/.claude/settings.json"

    def test_auto_detect_from_home_prefix(self):
        san = PathSanitizer("/home/fred/workspace/project")
        result = san.sanitize_path("/home/fred/.bashrc")
        assert result == "~/.bashrc"

