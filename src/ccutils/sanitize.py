# path-privacy: skip-file -- references universal Claude Code data paths (not personal)
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
    - Paths under cwd: make relative (e.g. /Users/dev/project/src/main.py -> src/main.py)
    - cwd itself: becomes "."
    - Paths under home dir: replace home prefix with ~ (e.g. /Users/dev/.claude -> ~/.claude)
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

        # Each prefix is replaced only where it is a whole path: not preceded
        # by a path character (so `/mnt/home/u/p` is left alone) and, for the
        # bare forms, not followed by one (so cwd `/home/u/p` does not eat the
        # `/home/u/p` of `/home/u/p-other`, which plain str.replace did).
        text = _replace_path(text, self.cwd + "/", "")
        text = _replace_path(text, self.cwd, ".", bare=True)
        if self.home_dir:
            text = _replace_path(text, self.home_dir + "/", "~/")
            text = _replace_path(text, self.home_dir, "~", bare=True)

        return text

    def sanitize_json_string(self, json_str: str | None) -> str | None:
        """Sanitize paths embedded in a JSON string.

        Applies the same text replacement as sanitize_text since
        paths appear as string values within JSON.
        """
        if json_str is None or self.is_noop:
            return json_str
        return self.sanitize_text(json_str)


# Characters that can continue a path component. A prefix followed by one is
# part of a longer name; a trailing "." counts only before whitespace or the
# end, so "in /home/u/p." still reads as the path /home/u/p.
_NAME_CHAR = r"[A-Za-z0-9_\-~]"


def _replace_path(text: str, prefix: str, replacement: str, bare: bool = False) -> str:
    start = r"(?<![A-Za-z0-9_\-.~/])"
    end = rf"(?!{_NAME_CHAR}|\.(?!\s|$))" if bare else ""
    return re.sub(start + re.escape(prefix) + end, lambda _m: replacement, text)


def _detect_home_dir(cwd: str) -> str | None:
    """Detect the home directory from a cwd path.

    Recognizes /Users/<name> (macOS) and /home/<name> (Linux) patterns.
    """
    match = re.match(r"^(/(?:Users|home)/[^/]+)", cwd)
    if match:
        return match.group(1)
    return None
