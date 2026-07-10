"""Tests for the Tier 2 LLM facet populator.

Step 4 of the facet & cluster pipeline. populate_tier2_facets reads
staged facts, builds a SessionInputs per session, calls the injected
FacetExtractor, and writes one fact_session_facets row per
(session × FacetSpec).

Coverage:
  - CannedFacetExtractor with a known value lands a clean F20 row
    (correct facet_type_key resolving through dim_facet_type, correct
    prompt_version, is_fallback=False, value_text populated).
  - CannedFacetExtractor with no canned value lands an is_fallback row
    (value_text=NULL, is_fallback=True) instead of going missing.
  - Multiple sessions populate independently.
  - Re-running is idempotent (extracted_at + metadata can churn; valid
    values don't shift hash_diff so last_updated_at is stable).
  - dim_facet_type seed includes F20 v1 row.
  - Live API smoke test skips unless ANTHROPIC_API_KEY is set; when
    enabled, hits Haiku and asserts F20 comes back non-empty.
"""

from __future__ import annotations

import json
import os

import pytest

from ccutils import create_star_schema
from ccutils.etl.facets import (
    AnthropicFacetExtractor,
    CannedFacetExtractor,
    FACET_SPECS,
    SessionInputs,
)
from ccutils.etl.orchestrator import run_v15_etl


class _AlwaysRaisingExtractor:
    """Extractor that raises on every call. Used to verify per-session
    failure isolation in the populator."""

    def extract(self, _inputs, _specs):
        raise RuntimeError("simulated extractor failure")


def _session_jsonl(tmp_path, session_id: str, first_user: str,
                   last_asst: str) -> str:
    """Write a minimal but well-shaped JSONL session file."""
    jsonl = tmp_path / f"{session_id}.jsonl"
    lines = [
        {"type": "user", "uuid": f"{session_id}-u1", "sessionId": session_id,
         "timestamp": "2026-04-19T10:00:00Z",
         "cwd": "/p", "gitBranch": "main", "version": "2.1.114",
         "permissionMode": "default",
         "message": {"role": "user", "content": first_user}},
        {"type": "assistant", "uuid": f"{session_id}-a1",
         "parentUuid": f"{session_id}-u1",
         "sessionId": session_id, "timestamp": "2026-04-19T10:00:05Z",
         "requestId": "req_1",
         "message": {"role": "assistant", "model": "claude-opus-4-7",
                     "content": [
                         {"type": "text", "text": last_asst},
                         {"type": "tool_use", "id": "tu1", "name": "Bash",
                          "input": {"command": "ls"}},
                     ],
                     "stop_reason": "tool_use",
                     "usage": {"input_tokens": 50, "output_tokens": 12,
                               "service_tier": "standard"}}},
        {"type": "user", "uuid": f"{session_id}-u2",
         "parentUuid": f"{session_id}-a1",
         "sessionId": session_id, "timestamp": "2026-04-19T10:00:10Z",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"},
         ]},
         "toolUseResult": {"stdout": "ok", "interrupted": False, "exitCode": 0}},
    ]
    jsonl.write_text("\n".join(json.dumps(d) for d in lines))
    return str(jsonl)


@pytest.fixture
def conn(tmp_path):
    return create_star_schema(tmp_path / "test.duckdb")


@pytest.fixture
def session_path(tmp_path):
    return _session_jsonl(
        tmp_path, "tier2-s",
        "fix the failing test_widget regression",
        "patched test_widget.py and reran the suite; all green",
    )


class TestSeedingAndDDL:
    def test_dim_facet_type_carries_f20(self, conn):
        row = conn.execute(
            "SELECT facet_id, facet_name, tier, method, prompt_version, "
            "prompt_text "
            "FROM dim_facet_type WHERE facet_id = 'F20'"
        ).fetchone()
        assert row is not None
        facet_id, facet_name, tier, method, prompt_version, prompt_text = row
        assert facet_id == "F20"
        assert facet_name == "task_description"
        assert tier == 2
        assert method == "llm"
        assert prompt_version == "v1"
        # prompt_text was seeded from FacetSpec.description -- it's the
        # description shown to the model in the system prompt's facet
        # schema, queryable post-hoc for auditing.
        assert prompt_text is not None
        assert len(prompt_text) > 0


class TestCannedExtractor:
    def test_canned_value_lands_in_value_text(
        self, conn, session_path, tmp_path
    ):
        canned = CannedFacetExtractor({
            ("tier2-s", "F20"): "debugged a failing test in a Python suite",
        })
        run_v15_etl(
            conn, session_path,
            project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
            facet_extractor=canned,
        )
        row = conn.execute(
            """
            SELECT fsf.value_text, fsf.is_fallback, fsf.prompt_version,
                   fsf.facet_type_key
            FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s'
              AND dft.facet_id = 'F20'
              AND dft.tier = 2
            """
        ).fetchone()
        assert row is not None, "F20 row did not land"
        value_text, is_fallback, prompt_version, facet_type_key = row
        assert value_text == "debugged a failing test in a Python suite"
        assert is_fallback is False
        assert prompt_version == "v1"
        # facet_type_key resolves the join to dim_facet_type, which is
        # the contract pinned by TestFacetTypeKeyVersionProperty.
        seeded_key = conn.execute(
            "SELECT facet_type_key FROM dim_facet_type "
            "WHERE facet_id = 'F20' AND prompt_version = 'v1'"
        ).fetchone()[0]
        assert facet_type_key == seeded_key

    def test_missing_canned_value_emits_fallback_row(
        self, conn, session_path, tmp_path
    ):
        # No canned entry for this session -> CannedFacetExtractor returns
        # is_fallback=True. Populator must still write the row (graceful
        # absence) rather than letting F20 go missing for the session.
        canned = CannedFacetExtractor({})
        run_v15_etl(
            conn, session_path,
            project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
            facet_extractor=canned,
        )
        row = conn.execute(
            """
            SELECT fsf.value_text, fsf.is_fallback
            FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s' AND dft.facet_id = 'F20'
            """
        ).fetchone()
        assert row is not None, "F20 fallback row did not land"
        value_text, is_fallback = row
        assert value_text is None
        assert is_fallback is True

    def test_default_run_skips_tier2_entirely(self, conn, session_path, tmp_path):
        # facet_extractor=None (default) means NO F20 row at all.
        run_v15_etl(
            conn, session_path,
            project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
        )
        count = conn.execute(
            """
            SELECT COUNT(*) FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s' AND dft.tier = 2
            """
        ).fetchone()[0]
        assert count == 0

    def test_extractor_exception_isolated_per_session(
        self, conn, session_path, tmp_path
    ):
        # An extractor that raises on every call: per-session failures are
        # logged but don't abort run_v15_etl. The session's Tier 2 rows are
        # absent (failure) but the rest of the pipeline still completes.
        run_v15_etl(
            conn, session_path,
            project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
            facet_extractor=_AlwaysRaisingExtractor(),
        )
        # Tier 1 should still have landed. Asserting "> 0" rather than
        # the exact count so the test stays green when the catalog grows.
        tier1_count = conn.execute(
            """
            SELECT COUNT(*) FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s' AND dft.tier = 1
            """
        ).fetchone()[0]
        assert tier1_count > 0, (
            "Tier 1 facets must still land when Tier 2 extraction fails"
        )
        tier2_count = conn.execute(
            """
            SELECT COUNT(*) FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s' AND dft.tier = 2
            """
        ).fetchone()[0]
        assert tier2_count == 0

    def test_failed_reextraction_preserves_prior_tier2_facets(
        self, conn, session_path, tmp_path
    ):
        # Run 1 lands a good F20 row. Run 2's extractor raises (transient API
        # failure), so it produces no inbound rows. The prior good facet MUST
        # survive -- an extraction failure is "unknown", not "this session has
        # no facets", so the soft-delete must not treat the empty inbound as a
        # signal to wipe existing rows. Regression guard for the data-loss bug
        # where the widened soft-delete scope keyed off staging, not inbound.
        lake = tmp_path / "lake"
        run_v15_etl(
            conn, session_path,
            project_name="test-project",
            parquet_lake_root=lake,
            facet_extractor=CannedFacetExtractor(
                {("tier2-s", "F20"): "did a real thing"}
            ),
        )
        live_before = conn.execute(
            """
            SELECT COUNT(*) FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s' AND dft.tier = 2
              AND fsf.is_deleted = FALSE
            """
        ).fetchone()[0]
        assert live_before == 1, "run 1 should land one live F20 row"

        run_v15_etl(
            conn, session_path,
            project_name="test-project",
            parquet_lake_root=lake,
            facet_extractor=_AlwaysRaisingExtractor(),
        )
        survivors = conn.execute(
            """
            SELECT fsf.value_text FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s' AND dft.tier = 2
              AND fsf.is_deleted = FALSE
            """
        ).fetchall()
        assert len(survivors) == 1, (
            "a failed re-extraction soft-deleted the prior Tier 2 facet"
        )
        assert survivors[0][0] == "did a real thing"


class TestMetadataAndIdempotency:
    def test_extraction_metadata_json_is_stored(
        self, conn, session_path, tmp_path
    ):
        # The Canned extractor uses metadata_json="{}" by default. The
        # populator passes it through verbatim. Schema-level contract
        # (full set of keys when AnthropicFacetExtractor is the source)
        # is enforced in test_anthropic_facet_extractor.py.
        canned = CannedFacetExtractor({("tier2-s", "F20"): "summary"})
        run_v15_etl(
            conn, session_path,
            project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
            facet_extractor=canned,
        )
        meta = conn.execute(
            """
            SELECT extraction_metadata_json::VARCHAR
            FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s' AND dft.facet_id = 'F20'
            """
        ).fetchone()[0]
        assert meta is not None  # column was populated, not NULL

    def test_idempotent_value_does_not_bump_last_updated_at(
        self, conn, session_path, tmp_path
    ):
        canned = CannedFacetExtractor({("tier2-s", "F20"): "summary"})
        run_v15_etl(
            conn, session_path,
            project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
            facet_extractor=canned,
        )
        before = conn.execute(
            """
            SELECT last_updated_at FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s' AND dft.facet_id = 'F20'
            """
        ).fetchone()[0]
        # Re-run with the same canned value. hash_diff excludes
        # extraction_metadata_json (which can churn) and extracted_at
        # (lineage metadata) -- so identical model value means
        # last_updated_at should NOT advance.
        run_v15_etl(
            conn, session_path,
            project_name="test-project",
            parquet_lake_root=tmp_path / "lake",
            facet_extractor=canned,
        )
        after = conn.execute(
            """
            SELECT last_updated_at FROM fact_session_facets fsf
            JOIN dim_facet_type dft USING (facet_type_key)
            WHERE fsf.session_id = 'tier2-s' AND dft.facet_id = 'F20'
            """
        ).fetchone()[0]
        assert before == after, (
            "Re-run with identical model value should be a no-op; "
            "last_updated_at must not advance"
        )


# ---------------------------------------------------------------------------
# Live API smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; live API smoke test skipped",
)
class TestLiveApiSmoke:
    """Hits Haiku once with a realistic SessionInputs and asserts F20 comes
    back non-empty. Skipped by default; enable by exporting
    ANTHROPIC_API_KEY before invoking pytest.

    This is the only test that actually spends money; it spends
    pennies."""

    def test_haiku_returns_non_empty_task_description(self):
        extractor = AnthropicFacetExtractor(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-haiku-4-5-20251001",
        )
        inputs = SessionInputs(
            session_id="live-smoke",
            first_user_message=(
                "The pre-commit hook is failing because pyright complains "
                "about a missing return annotation on a private helper. "
                "Can you add the annotation and rerun?"
            ),
            last_assistant_message=(
                "Added `-> None` to the helper signature and reran "
                "pre-commit; all hooks now pass."
            ),
            tool_mix_summary="Edit×2, Bash×1, Read×1",
            model_used="claude-opus-4-7",
            duration_seconds=180,
        )
        out = extractor.extract(inputs, FACET_SPECS)
        f20 = out["F20"]
        assert f20.is_fallback is False, (
            f"Haiku failed to extract F20: metadata={f20.metadata_json}"
        )
        assert f20.value is not None
        assert len(f20.value) > 0
        # Quick sanity: the privacy guardrail should suppress specific
        # tool names from the summary (model doesn't say "pyright" if
        # privacy guidance worked). Soft check; not a hard contract.
        meta = json.loads(f20.metadata_json)
        assert "raw_response" in meta
        assert meta["input_tokens"] > 0
