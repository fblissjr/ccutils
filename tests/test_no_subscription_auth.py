"""Guard: ccutils reads no credential, least of all a Claude subscription login.

The deleted `web` command fetched sessions from an undocumented endpoint
using the Claude Code OAuth token (read from the keychain) and the
organization UUID (read from the user's Claude config file). Using
subscription credentials outside Anthropic's own apps is not permitted.
The one credential ccutils used to resolve, a developer API key for the
Tier 2 facet extractor, went with the facet layer on 2026-11-13, so the
source should now name no credential at all.

The scan covers `src/ccutils/` only, so this file's own literals never trip
it. What breaks without it: a helper that reads the keychain or an API key
comes back unnoticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ccutils

SRC_ROOT = Path(ccutils.__file__).resolve().parent

# Subscription-auth literals (the Claude Code keychain service, the OAuth
# keys in the stored credentials and config, the session-ingress path), and
# the developer-key path that no longer exists.
FORBIDDEN_LITERALS = (
    "Claude Code-credentials",
    "claudeAiOauth",
    "oauthAccount",
    "session_ingress",
    "ANTHROPIC_API_KEY",
    "ccutils-anthropic",
    "find-generic-password",
)


def _source_files():
    return sorted(SRC_ROOT.rglob("*.py"))


def test_scan_reaches_the_source_tree():
    """A scan over zero files passes vacuously; make sure it has files."""
    files = _source_files()
    assert len(files) > 50, f"expected the ccutils source tree, found {files}"
    assert SRC_ROOT / "cli" / "__init__.py" in files


@pytest.mark.parametrize("literal", FORBIDDEN_LITERALS)
def test_no_credential_literal_in_source(literal):
    hits = [
        str(path.relative_to(SRC_ROOT.parent))
        for path in _source_files()
        if literal in path.read_text(encoding="utf-8")
    ]
    assert not hits, f"{literal!r} found in: {hits}"
