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
# sets the click command as `cli.local`, which means dotted-string
# monkeypatch resolution that walks `getattr(cli_package, "local")`
# returns the Click command object instead of the Python submodule.
# `importlib.import_module` bypasses that attribute walk and gives us
# the module namespace where the helpers actually live.
local_module = importlib.import_module("ccutils.cli.local")
all_module = importlib.import_module("ccutils.cli.all")
utils_module = importlib.import_module("ccutils.cli.utils")


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
    Captures construction args on the INSTANCE (not the class) so tests
    don't leak state between runs. Each test that uses this also
    captures the constructed instance via a spy hook so the assertion
    reads from a known-good reference."""

    def __init__(self, **kwargs):
        self.constructed_with = kwargs

    def extract(self, _inputs, _specs):
        return {}


@pytest.fixture
def recorded_extractors():
    """A list that captures every _RecordingExtractor instance the CLI
    constructs during a test. Used by tests asserting on construction
    args without relying on shared class state."""
    return []


def _make_recording_class(captures: list):
    """Returns a class that records every instance in `captures`."""
    class _Recorder(_RecordingExtractor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captures.append(self)
    return _Recorder


def _combined_output(result):
    """CLI errors land on stderr via `click.echo(err=True)`. Click's
    `CliRunner` mixes stdout+stderr into `result.output` by default
    (and raises ValueError on `.stderr` in that mode); if a future Click
    version separates them, fall back to concatenating. Either way the
    caller gets a single string to search."""
    out = result.output or ""
    try:
        err = result.stderr or ""
    except ValueError:
        # Streams are mixed -- everything is already in result.output.
        err = ""
    return out + err


class TestLocalCommand:
    def test_default_does_not_construct_extractor(
        self, sample_jsonl, tmp_path, monkeypatch
    ):
        # Without --with-llm-facets, the CLI must not touch
        # resolve_anthropic_key or construct an extractor. Otherwise
        # users without an API key can't use the basic pipeline.
        called = []

        def _should_not_call():
            called.append(True)
            return "sk-test"

        # Spy on the shared helper's resolver -- patching utils_module
        # covers both `local` and `all` since both import from there.
        monkeypatch.setattr(
            utils_module, "resolve_anthropic_key", _should_not_call,
        )

        runner = CliRunner()
        db_path = tmp_path / "out.duckdb"
        result = runner.invoke(
            cli,
            [str(sample_jsonl), "--format", "duckdb",
             "-o", str(db_path)],
        )
        assert result.exit_code == 0, result.output
        assert called == [], (
            "Credentials must NOT be resolved when --with-llm-facets is absent"
        )

    def test_with_llm_facets_constructs_and_passes_extractor(
        self, sample_jsonl, tmp_path, monkeypatch, recorded_extractors
    ):
        captured = {}

        monkeypatch.setattr(
            utils_module, "resolve_anthropic_key",
            lambda: "sk-test-from-keychain",
        )
        monkeypatch.setattr(
            utils_module, "AnthropicFacetExtractor",
            _make_recording_class(recorded_extractors),
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
            [str(sample_jsonl), "--format", "duckdb",
             "-o", str(db_path), "--with-llm-facets"],
        )
        assert result.exit_code == 0, result.output
        assert isinstance(captured["facet_extractor"], _RecordingExtractor)
        # Credentials resolved through resolve_anthropic_key() and threaded
        # into the constructor.
        assert len(recorded_extractors) == 1
        assert recorded_extractors[0].constructed_with["api_key"] == (
            "sk-test-from-keychain"
        )

    def test_credentials_error_exits_cleanly(
        self, sample_jsonl, tmp_path, monkeypatch
    ):
        from ccutils.api import CredentialsError

        def _raise():
            raise CredentialsError(
                "No Anthropic API key found.\n"
                "Set ANTHROPIC_API_KEY in the environment, or store it in "
                "macOS keychain:\n"
                "  security add-generic-password -s ccutils-anthropic "
                "-a $USER -w"
            )

        monkeypatch.setattr(
            utils_module, "resolve_anthropic_key", _raise,
        )

        runner = CliRunner()
        db_path = tmp_path / "out.duckdb"
        result = runner.invoke(
            cli,
            [str(sample_jsonl), "--format", "duckdb",
             "-o", str(db_path), "--with-llm-facets"],
        )
        # Exit code 2 is the contract from build_facet_extractor_or_exit.
        assert result.exit_code == 2
        combined = _combined_output(result)
        assert "ANTHROPIC_API_KEY" in combined
        assert "keychain" in combined.lower()
        assert "Traceback" not in combined


class TestAllCommand:
    def test_batch_llm_facets_constructs_extractor(
        self, sample_jsonl, tmp_path, monkeypatch, recorded_extractors
    ):
        # `all` resolves a directory of projects; build a minimal layout.
        projects_root = tmp_path / "projects"
        proj = projects_root / "myproj"
        proj.mkdir(parents=True)
        (proj / "cli-s.jsonl").write_text(sample_jsonl.read_text())

        monkeypatch.setattr(
            utils_module, "resolve_anthropic_key",
            lambda: "sk-test-from-env",
        )
        monkeypatch.setattr(
            utils_module, "AnthropicFacetExtractor",
            _make_recording_class(recorded_extractors),
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
             "--format", "duckdb", "-o", str(out),
             "--batch-llm-facets", "--quiet"],
        )
        assert result.exit_code == 0, result.output
        assert isinstance(captured["facet_extractor"], _RecordingExtractor)
        assert len(recorded_extractors) == 1

    def test_batch_default_does_not_resolve_credentials(
        self, sample_jsonl, tmp_path, monkeypatch
    ):
        projects_root = tmp_path / "projects"
        proj = projects_root / "myproj"
        proj.mkdir(parents=True)
        (proj / "cli-s.jsonl").write_text(sample_jsonl.read_text())

        called = []
        monkeypatch.setattr(
            utils_module, "resolve_anthropic_key",
            lambda: called.append(True) or "should-not-be-used",
        )

        runner = CliRunner()
        out = tmp_path / "out"
        result = runner.invoke(
            cli,
            ["all", "--source", str(projects_root),
             "--format", "duckdb", "-o", str(out), "--quiet"],
        )
        assert result.exit_code == 0, result.output
        assert called == [], (
            "Credentials must NOT be resolved when --batch-llm-facets is absent"
        )

    def test_json_star_format_forwards_extractor(
        self, sample_jsonl, tmp_path, monkeypatch, recorded_extractors
    ):
        # R-6 from simplify review: --format json + --batch-llm-facets
        # goes through generate_json_archive, which delegates to
        # generate_duckdb_archive internally. Ensure the extractor
        # actually reaches that internal path.
        projects_root = tmp_path / "projects"
        proj = projects_root / "myproj"
        proj.mkdir(parents=True)
        (proj / "cli-s.jsonl").write_text(sample_jsonl.read_text())

        monkeypatch.setattr(
            utils_module, "resolve_anthropic_key",
            lambda: "sk-test-from-env",
        )
        monkeypatch.setattr(
            utils_module, "AnthropicFacetExtractor",
            _make_recording_class(recorded_extractors),
        )

        captured = {}
        from ccutils.export import duckdb_archive as _da
        real_star_json = _da.generate_json_archive

        def _spy_star_json(*args, **kwargs):
            captured["facet_extractor"] = kwargs.get("facet_extractor")
            return real_star_json(*args, **kwargs)

        monkeypatch.setattr(
            all_module, "generate_json_archive", _spy_star_json,
        )

        runner = CliRunner()
        out = tmp_path / "out-json"
        result = runner.invoke(
            cli,
            ["all", "--source", str(projects_root),
             "--format", "json", "-o", str(out),
             "--batch-llm-facets", "--quiet"],
        )
        assert result.exit_code == 0, result.output
        assert isinstance(captured["facet_extractor"], _RecordingExtractor)
