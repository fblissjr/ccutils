"""Guard: ccutils never reads a Claude subscription login.

The deleted `web` command fetched sessions from an undocumented endpoint
using the Claude Code OAuth token (read from the keychain) and the
organization UUID (read from the user's Claude config file). Using
subscription credentials outside Anthropic's own apps is not permitted,
so that code and the helpers it left behind are gone, and this test keeps
them gone.

The only credential ccutils resolves is a developer API key for the
Tier 2 facet extractor (`ANTHROPIC_API_KEY` or the `ccutils-anthropic`
keychain entry). The positive checks below pin that path so the guard
cannot pass by deleting everything.

The string scan covers `src/ccutils/` only, so this file's own literals
never trip it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import ccutils

SRC_ROOT = Path(ccutils.__file__).resolve().parent

# Literals that only subscription-auth or web-session code would contain:
# the Claude Code keychain service, the two OAuth keys inside the stored
# credentials / config, and the session-ingress endpoint path.
FORBIDDEN_LITERALS = (
    "Claude Code-credentials",
    "claudeAiOauth",
    "oauthAccount",
    "session_ingress",
)

# (module, attribute) pairs that must no longer exist.
REMOVED_NAMES = [
    (module, name)
    for module, names in {
        "ccutils": (
            "API_BASE_URL",
            "ANTHROPIC_VERSION",
            "get_access_token_from_keychain",
            "get_org_uuid_from_config",
            "get_api_headers",
            "fetch_sessions",
            "fetch_session",
            "resolve_credentials",
            "format_session_for_display",
        ),
        "ccutils.api": (
            "API_BASE_URL",
            "ANTHROPIC_VERSION",
            "get_access_token_from_keychain",
            "get_org_uuid_from_config",
            "get_api_headers",
            "fetch_sessions",
            "fetch_session",
            "enrich_sessions_with_repos",
            "filter_sessions_by_repo",
        ),
        "ccutils.cli": (
            "resolve_credentials",
            "format_session_for_display",
        ),
        "ccutils.cli.utils": (
            "resolve_credentials",
            "format_session_for_display",
            "get_access_token_from_keychain",
            "get_org_uuid_from_config",
        ),
        "ccutils.tui": ("build_web_session_choices",),
        "ccutils.tui.selection": ("build_web_session_choices",),
        "ccutils.parsers": ("extract_repo_from_session",),
        "ccutils.parsers.session": ("extract_repo_from_session",),
    }.items()
    for name in names
]


def _source_files():
    return sorted(SRC_ROOT.rglob("*.py"))


def test_scan_reaches_the_source_tree():
    """A scan over zero files passes vacuously; make sure it has files."""
    files = _source_files()
    assert len(files) > 50, f"expected the ccutils source tree, found {files}"
    assert SRC_ROOT / "api" / "__init__.py" in files


@pytest.mark.parametrize("literal", FORBIDDEN_LITERALS)
def test_no_subscription_auth_literal_in_source(literal):
    hits = [
        str(path.relative_to(SRC_ROOT.parent))
        for path in _source_files()
        if literal in path.read_text(encoding="utf-8")
    ]
    assert not hits, f"{literal!r} found in: {hits}"


@pytest.mark.parametrize(("module", "name"), REMOVED_NAMES)
def test_removed_name_is_not_exposed(module, name):
    mod = importlib.import_module(module)
    assert not hasattr(mod, name), f"{module}.{name} still exists"
    assert name not in getattr(mod, "__all__", ()), (
        f"{module}.__all__ still lists {name}"
    )


def test_api_key_resolution_survives():
    """The permitted credential path is still importable where callers expect it."""
    from ccutils import CredentialsError as top_level_error
    from ccutils.api import CredentialsError, resolve_anthropic_key
    from ccutils.cli.utils import build_facet_extractor_or_exit

    assert top_level_error is CredentialsError
    assert callable(resolve_anthropic_key)
    assert callable(build_facet_extractor_or_exit)
