"""Privacy-preserving path sanitization for exports.

Converts absolute paths to relative/anonymized forms so exports
can be shared without leaking directory structure, usernames,
or machine-specific information.
"""

import os
import re


class PathSanitizer:
    """Sanitizes file paths and free text to remove sensitive path prefixes.

    Strategy:
    - Paths under cwd: make relative (e.g. /Users/fred/project/src/main.py -> src/main.py)
    - cwd itself: becomes "."
    - Paths under home dir: replace home prefix with ~ (e.g. /Users/fred/.claude -> ~/.claude)
    - System paths (e.g. /usr/local/bin/python): left as-is
    - Free text: ordered string replacement of cwd and home prefixes

    If cwd is None (e.g. web imports), the sanitizer becomes a no-op.
    """

    def __init__(self, cwd: str | None, home_dir: str | None = None):
        self.cwd = cwd.rstrip("/") if cwd else None
        self.is_noop = self.cwd is None

        if home_dir is not None:
            self.home_dir = home_dir.rstrip("/")
        elif self.cwd:
            self.home_dir = _detect_home_dir(self.cwd)
        else:
            self.home_dir = None

    def sanitize_path(self, path: str | None) -> str | None:
        """Sanitize a single file path."""
        if path is None or path == "" or self.is_noop:
            return path

        # cwd itself -> "."
        if path.rstrip("/") == self.cwd:
            return "."

        # Paths under cwd -> relative
        cwd_prefix = self.cwd + "/"
        if path.startswith(cwd_prefix):
            return path[len(cwd_prefix) :]

        # Paths under home dir -> ~/...
        if self.home_dir:
            if path.rstrip("/") == self.home_dir:
                return "~"
            home_prefix = self.home_dir + "/"
            if path.startswith(home_prefix):
                return "~/" + path[len(home_prefix) :]

        # System paths or unrecognized: leave as-is
        return path

    def sanitize_cwd(self) -> str | None:
        """Sanitize the cwd field itself."""
        if self.is_noop:
            return None
        return "."

    def sanitize_project_path(self, path: str | None) -> str | None:
        """Sanitize a project path by stripping to its basename."""
        if path is None or self.is_noop:
            return path
        # Extract just the last component
        return os.path.basename(path.rstrip("/"))

    def sanitize_text(self, text: str | None) -> str | None:
        """Sanitize free text by replacing path prefixes.

        Order matters: cwd/ first (most specific), then cwd alone,
        then home/, then home. This prevents partial replacements.
        """
        if text is None or self.is_noop:
            return text

        # Replace cwd/ first (files under cwd)
        text = text.replace(self.cwd + "/", "")
        # Replace cwd alone (e.g. "cd /path/to/project")
        text = text.replace(self.cwd, ".")

        # Replace home dir paths
        if self.home_dir:
            text = text.replace(self.home_dir + "/", "~/")
            text = text.replace(self.home_dir, "~")

        return text

    def sanitize_json_string(self, json_str: str | None) -> str | None:
        """Sanitize paths embedded in a JSON string.

        Applies the same text replacement as sanitize_text since
        paths appear as string values within JSON.
        """
        if json_str is None or self.is_noop:
            return json_str
        return self.sanitize_text(json_str)


def _detect_home_dir(cwd: str) -> str | None:
    """Detect the home directory from a cwd path.

    Recognizes /Users/<name> (macOS) and /home/<name> (Linux) patterns.
    """
    match = re.match(r"^(/(?:Users|home)/[^/]+)", cwd)
    if match:
        return match.group(1)
    return None
