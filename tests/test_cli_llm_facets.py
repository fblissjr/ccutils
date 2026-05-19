"""CLI integration tests for --with-llm-facets / --batch-llm-facets.

Step 4 follow-up. Verifies the flags wire end-to-end:
  - Construct AnthropicFacetExtractor at the CLI boundary.
  - Pass it to run_v15_etl (single-session) or generate_duckdb_archive
    (batch).
  - Resolve credentials via resolve_anthropic_key().
  - Catch CredentialsError at the CLI boundary -- helpful message,
    non-zero exit, no stack trace.

The tests don't actually hit the API: they patch
AnthropicFacetExtractor and resolve_anthropic_key to record their
arguments, then assert run_v15_etl received the constructed
extractor. The real network path is covered by the live-API smoke
test in tests/test_populate_tier2_facets.py.
"""

from __future__ import annotations

import importlib
import json

import pytest
from click.testing import CliRunner

from ccutils.cli import cli

# `cli.add_command(local_cmd, "local")` in `ccutils.cli/__init__.py`
# binds the click subcommand as the `local` attribute on the cli group,
# which shadows the `ccutils.cli.local` submodule for `getattr`-based
# attribute walks (including pytest monkeypatch's dotted-string form
# and even `import ccutils.cli.local as ...`). Use importlib to load
# the actual module directly.
local_module = importlib.import_module("ccutils.cli.local")
all_module = importlib.import_module("ccutils.cli.all")


@pytest.fixture
def sample_jsonl(tmp_path):
    jsonl = tmp_path / "s.jsonl"
    lines = [
        {"type": "user", "uuid": "u1", "sessionId": "cli-s",
         "timestamp": "2026-04-19T10:00:00Z",
         "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
         "permissionMode": "default",
         "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "uuid": "a1", "parentUuid": "u1",
         "sessionId": "cli-s", "timestamp": "2026-04-19T10:00:01Z",
         "requestId": "req_1",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [{"type": "text", "text": "ok"}],
                     "stop_reason": "end_turn",
                     "usage": {"input_tokens": 5, "output_tokens": 3,
                               "service_tier": "standard"}}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return jsonl


class _RecordingExtractor:
    """Test double that the CLI builds in place of AnthropicFacetExtractor.
    Captures construction args so tests can assert the wiring."""

    constructed_with: dict | None = None

    def __init__(self, **kwargs):
        type(self).constructed_with = kwargs

    def extract(self, _inputs, _specs):
        return {}


class TestLocalCommand:
    def test_default_does_not_construct_extractor(
        self, sample_jsonl, tmp_path, monkeypatch
    ):
        # Without --with-llm-facets, the CLI must not touch
        # resolve_anthropic_key or construct an extractor. Otherwise
        # users without an API key can't use the basic pipeline.
        called = []

        def _should_not_call(*_a, **_k):
            called.append(True)
            return "sk-test"

        monkeypatch.setattr(
            local_module, "resolve_anthropic_key", _should_not_call,
        )

        runner = CliRunner()
        db_path = tmp_path / "out.duckdb"
        result = runner.invoke(
            cli,
            [str(sample_jsonl), "--format", "duckdb-star",
             "-o", str(db_path)],
        )
        assert result.exit_code == 0, result.output
        assert called == [], (
            "Credentials must NOT be resolved when --with-llm-facets is absent"
        )

    def test_with_llm_facets_constructs_and_passes_extractor(
        self, sample_jsonl, tmp_path, monkeypatch
    ):
        _RecordingExtractor.constructed_with = None
        captured = {}

        monkeypatch.setattr(
            local_module, "resolve_anthropic_key",
            lambda: "sk-test-from-keychain",
        )
        monkeypatch.setattr(
            local_module, "AnthropicFacetExtractor", _RecordingExtractor,
        )

        from ccutils.etl import orchestrator as _orch
        real_run = _orch.run_v15_etl

        def _spy_run(conn, session_path, **kwargs):
            captured["facet_extractor"] = kwargs.get("facet_extractor")
            return real_run(conn, session_path, **kwargs)

        monkeypatch.setattr(local_module, "run_v15_etl", _spy_run)

        runner = CliRunner()
        db_path = tmp_path / "out.duckdb"
        result = runner.invoke(
            cli,
            [str(sample_jsonl), "--format", "duckdb-star",
             "-o", str(db_path), "--with-llm-facets"],
        )
        assert result.exit_code == 0, result.output
        assert isinstance(captured["facet_extractor"], _RecordingExtractor)
        # Credentials resolved through resolve_anthropic_key() and threaded
        # into the constructor.
        assert _RecordingExtractor.constructed_with["api_key"] == "sk-test-from-keychain"

    def test_credentials_error_exits_cleanly(
        self, sample_jsonl, tmp_path, monkeypatch
    ):
        from ccutils.api import CredentialsError

        def _raise(*_a, **_k):
            raise CredentialsError(
                "No Anthropic API key found.\n"
                "Set ANTHROPIC_API_KEY in the environment, or store it in "
                "macOS keychain:\n"
                "  security add-generic-password -s ccutils-anthropic "
                "-a $USER -w"
            )

        monkeypatch.setattr(
            local_module, "resolve_anthropic_key", _raise,
        )

        runner = CliRunner()
        db_path = tmp_path / "out.duckdb"
        result = runner.invoke(
            cli,
            [str(sample_jsonl), "--format", "duckdb-star",
             "-o", str(db_path), "--with-llm-facets"],
        )
        # Non-zero exit, helpful message, NOT a Python traceback.
        assert result.exit_code != 0
        assert "ANTHROPIC_API_KEY" in result.output
        assert "keychain" in result.output.lower()
        assert "Traceback" not in result.output


class TestAllCommand:
    def test_batch_llm_facets_constructs_extractor(
        self, sample_jsonl, tmp_path, monkeypatch
    ):
        # `all` resolves a directory of projects; build a minimal layout.
        projects_root = tmp_path / "projects"
        proj = projects_root / "myproj"
        proj.mkdir(parents=True)
        (proj / "cli-s.jsonl").write_text(sample_jsonl.read_text())

        _RecordingExtractor.constructed_with = None

        monkeypatch.setattr(
            all_module, "resolve_anthropic_key",
            lambda: "sk-test-from-env",
        )
        monkeypatch.setattr(
            all_module, "AnthropicFacetExtractor", _RecordingExtractor,
        )

        captured = {}
        from ccutils.export import duckdb_archive as _da
        real_generate = _da.generate_duckdb_archive

        def _spy_generate(*args, **kwargs):
            captured["facet_extractor"] = kwargs.get("facet_extractor")
            return real_generate(*args, **kwargs)

        monkeypatch.setattr(
            all_module, "generate_duckdb_archive", _spy_generate,
        )

        runner = CliRunner()
        out = tmp_path / "out"
        result = runner.invoke(
            cli,
            ["all", "--source", str(projects_root),
             "--format", "duckdb-star", "-o", str(out),
             "--batch-llm-facets", "--quiet"],
        )
        assert result.exit_code == 0, result.output
        assert isinstance(captured["facet_extractor"], _RecordingExtractor)

    def test_batch_default_does_not_resolve_credentials(
        self, sample_jsonl, tmp_path, monkeypatch
    ):
        projects_root = tmp_path / "projects"
        proj = projects_root / "myproj"
        proj.mkdir(parents=True)
        (proj / "cli-s.jsonl").write_text(sample_jsonl.read_text())

        called = []
        monkeypatch.setattr(
            all_module, "resolve_anthropic_key",
            lambda: called.append(True) or "should-not-be-used",
        )

        runner = CliRunner()
        out = tmp_path / "out"
        result = runner.invoke(
            cli,
            ["all", "--source", str(projects_root),
             "--format", "duckdb-star", "-o", str(out), "--quiet"],
        )
        assert result.exit_code == 0, result.output
        assert called == [], (
            "Credentials must NOT be resolved when --batch-llm-facets is absent"
        )
