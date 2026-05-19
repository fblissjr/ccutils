"""Tests for resolve_anthropic_key().

Step 3 of the facet pipeline. Pattern matches
ccutils.api.get_access_token_from_keychain (which fetches the Claude
Code OAuth token from macOS keychain) but for the developer-grade
Anthropic API key. Distinct keychain service name so the two
credentials don't collide.

Resolution order:
  1. ANTHROPIC_API_KEY environment variable (the SDK convention).
  2. macOS keychain service "ccutils-anthropic" (matches the web
     command's keychain pattern; Darwin only).
  3. Fail loud -- raise CredentialsError with both options spelled out.
"""

from __future__ import annotations

import platform
import subprocess

import pytest

from ccutils.api import CredentialsError, resolve_anthropic_key


class TestEnvVar:
    def test_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        assert resolve_anthropic_key() == "sk-ant-from-env"

    def test_empty_env_var_is_not_a_key(self, monkeypatch):
        # Treat empty string as "not set" so an explicit `export
        # ANTHROPIC_API_KEY=""` doesn't silently bypass the keychain
        # fallback.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        with pytest.raises(CredentialsError):
            resolve_anthropic_key()


class TestKeychain:
    def test_keychain_fallback_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(platform, "system", lambda: "Darwin")

        def fake_run(cmd, **kwargs):
            assert "find-generic-password" in cmd
            assert "ccutils-anthropic" in cmd
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout="sk-ant-from-keychain\n", stderr=""
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert resolve_anthropic_key() == "sk-ant-from-keychain"

    def test_keychain_not_consulted_on_non_darwin(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        called = []

        def fake_run(*args, **kwargs):
            called.append(args)
            return subprocess.CompletedProcess(args, 0, "x", "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CredentialsError):
            resolve_anthropic_key()
        assert called == [], "keychain should not be queried on non-Darwin"

    def test_keychain_miss_falls_through_to_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(platform, "system", lambda: "Darwin")

        def fake_run(cmd, **kwargs):
            # `security find-generic-password` returns 44 when not found.
            return subprocess.CompletedProcess(cmd, 44, "", "not found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(CredentialsError):
            resolve_anthropic_key()


class TestFailLoud:
    def test_error_message_names_both_options(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        with pytest.raises(CredentialsError) as exc:
            resolve_anthropic_key()
        msg = str(exc.value)
        # Both resolution paths must be discoverable from the error so
        # users don't have to dig in the source.
        assert "ANTHROPIC_API_KEY" in msg
        assert "keychain" in msg.lower() or "security add-generic-password" in msg
