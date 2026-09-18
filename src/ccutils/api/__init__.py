"""Anthropic API key resolution for the Tier 2 facet extractor.

The only credential ccutils resolves is a developer API key. It never
reads a Claude subscription login: using those credentials outside
Anthropic's own apps is not permitted.
"""

import os
import platform
import subprocess


class CredentialsError(Exception):
    """Raised when credentials cannot be obtained."""

    pass


_ANTHROPIC_KEYCHAIN_SERVICE = "ccutils-anthropic"


def resolve_anthropic_key() -> str:
    """Resolve a developer-grade Anthropic API key for the Tier 2 facet
    extractor.

    Order: ANTHROPIC_API_KEY env var first (the SDK convention), then
    macOS keychain service `ccutils-anthropic` (a service name of its own,
    so it never touches Claude Code's keychain entry). Fails loud with
    both options spelled out if neither resolves.

    Returns:
        The API key string.

    Raises:
        CredentialsError: if neither source resolves a key.
    """
    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        return env_key

    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                [
                    "security", "find-generic-password",
                    "-a", os.environ.get("USER", ""),
                    "-s", _ANTHROPIC_KEYCHAIN_SERVICE,
                    "-w",
                ],
                capture_output=True,
                text=True,
            )
        except subprocess.SubprocessError as e:
            raise CredentialsError(
                f"Failed to query keychain: {e}"
            ) from e
        if result.returncode == 0:
            key = result.stdout.strip()
            if key:
                return key

    raise CredentialsError(
        "No Anthropic API key found.\n"
        "Set ANTHROPIC_API_KEY in the environment, or store it in macOS keychain:\n"
        f"  security add-generic-password -s {_ANTHROPIC_KEYCHAIN_SERVICE} "
        "-a $USER -w"
    )


__all__ = [
    "CredentialsError",
    "resolve_anthropic_key",
]
