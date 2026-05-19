"""Tests for the FacetExtractor protocol module and the Canned stub.

Step 3 of the facet & cluster pipeline. Covers the boundary contract that
both AnthropicFacetExtractor and any future local-model implementation
must satisfy.

Three classes here:
  * TestDataclasses -- the FacetSpec / SessionInputs / FacetOutput
    dataclasses are frozen, immutable, and field-typed.
  * TestProtocol -- runtime_checkable so populators can assert at the
    boundary.
  * TestCannedFacetExtractor -- the fake-backend stub used by populator
    tests. Returns canned values; emits is_fallback=True when no canned
    value is set.
"""

from __future__ import annotations

import pytest

from ccutils.etl.facets.extractor import (
    CannedFacetExtractor,
    FacetExtractor,
    FacetOutput,
    FacetSpec,
    SessionInputs,
)


@pytest.fixture
def session_inputs():
    return SessionInputs(
        session_id="s1",
        first_user_message="please debug the failing test",
        last_assistant_message="fixed the assertion in test_foo",
        tool_mix_summary="Bash×3, Edit×2, Read×1",
        model_used="claude-opus-4-7",
        duration_seconds=120,
    )


@pytest.fixture
def f20_spec():
    return FacetSpec(
        facet_id="F20",
        facet_name="task_description",
        output_type="text",
        prompt_version="v1",
        description="One- or two-sentence summary of what the session was about.",
    )


@pytest.fixture
def f22_spec():
    return FacetSpec(
        facet_id="F22",
        facet_name="blocker_type",
        output_type="enum",
        prompt_version="v1",
        description="What stopped progress when outcome wasn't success.",
        enum_values=("none", "knowledge", "environment", "tool-limit",
                     "unclear-req", "external-dep"),
    )


class TestDataclasses:
    def test_facet_spec_is_frozen(self, f20_spec):
        with pytest.raises(Exception):
            f20_spec.prompt_version = "v2"  # type: ignore[misc]

    def test_session_inputs_is_frozen(self, session_inputs):
        with pytest.raises(Exception):
            session_inputs.session_id = "s2"  # type: ignore[misc]

    def test_facet_output_carries_version_and_metadata(self):
        out = FacetOutput(
            facet_id="F20",
            prompt_version="v1",
            value="task summary",
            is_fallback=False,
            metadata_json='{"retry_count": 0, "cache_hit": true}',
        )
        assert out.facet_id == "F20"
        assert out.prompt_version == "v1"
        assert out.is_fallback is False
        assert "retry_count" in out.metadata_json

    def test_facet_output_default_metadata_is_empty_object(self):
        # Default `metadata_json` is "{}" so downstream JSON parsing
        # always succeeds even when the extractor didn't bother to set
        # anything (e.g. CannedFacetExtractor).
        out = FacetOutput(facet_id="F20", prompt_version="v1", value="x")
        assert out.metadata_json == "{}"

    def test_facet_spec_enum_values_only_for_enums(self, f20_spec, f22_spec):
        assert f20_spec.enum_values is None
        assert f22_spec.enum_values is not None
        assert "tool-limit" in f22_spec.enum_values


class TestProtocol:
    def test_canned_is_a_facet_extractor(self):
        # Protocol is runtime_checkable: isinstance() works.
        canned = CannedFacetExtractor({})
        assert isinstance(canned, FacetExtractor)

    def test_protocol_has_extract_method(self):
        assert hasattr(FacetExtractor, "extract")


class TestCannedFacetExtractor:
    def test_returns_one_output_per_spec(self, session_inputs, f20_spec, f22_spec):
        canned = CannedFacetExtractor({
            ("s1", "F20"): "fixed a flaky test",
            ("s1", "F22"): "knowledge",
        })
        out = canned.extract(session_inputs, [f20_spec, f22_spec])
        assert set(out.keys()) == {"F20", "F22"}

    def test_returns_canned_value(self, session_inputs, f20_spec):
        canned = CannedFacetExtractor({("s1", "F20"): "summary text"})
        out = canned.extract(session_inputs, [f20_spec])
        assert out["F20"].value == "summary text"
        assert out["F20"].is_fallback is False
        assert out["F20"].prompt_version == "v1"

    def test_fallback_when_no_canned_value(self, session_inputs, f20_spec):
        # Every spec MUST get a FacetOutput, even if extraction "failed".
        # Missing canned key => is_fallback=True with value=None.
        canned = CannedFacetExtractor({})
        out = canned.extract(session_inputs, [f20_spec])
        assert out["F20"].value is None
        assert out["F20"].is_fallback is True
        assert out["F20"].prompt_version == "v1"

    def test_per_facet_independence(self, session_inputs, f20_spec, f22_spec):
        # Some facets canned, others not. Each independently set vs fallback.
        canned = CannedFacetExtractor({("s1", "F20"): "summary"})
        out = canned.extract(session_inputs, [f20_spec, f22_spec])
        assert out["F20"].is_fallback is False
        assert out["F22"].is_fallback is True
