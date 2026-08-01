"""Tests for AnthropicFacetExtractor.

Mocks the HTTP layer (via pytest-httpx), not the API SDK -- we want to
exercise the real httpx code path with canned responses, so the tests
catch issues in request shape (cache_control wiring, system/user
split, JSON schema in the prompt) as well as response parsing.

**Note on `_no_sleep`:** the autouse fixture below patches `time.sleep`
to a no-op for the entire module. The retry tests would otherwise add
seconds (1s + 4s backoff per validation attempt; up to 21s for HTTP
retries). Patching is the right call for unit speed but means we have
no end-to-end test asserting that real backoff intervals fire as
designed. If backoff correctness ever needs a real-time test, write it
as a separate module without `_no_sleep` and gate it behind a marker.

Coverage targets correspond to the malformed-output policy table in
internal/plans/facet_extractor_protocol.md §3:
  - happy path: single + multi facet
  - invalid JSON -> retry once -> still bad -> all fallback
  - Pydantic enum validation fail -> retry -> per-facet fallback (only
    the offending facet falls back; valid facets are kept)
  - missing facet key -> fallback for that facet only
  - hallucinated extra key -> silently dropped (extra="ignore")
  - empty string for required text facet -> fallback for that facet
  - 429 rate-limit -> retry with backoff
  - 401 auth failure -> raise (no retry)
  - 5xx -> retry, then all fallback on exhaustion
  - request shape: cache_control on system, sentinel-wrapped user prompt
"""

from __future__ import annotations

import json

import pytest

from ccutils.etl.facets.anthropic import (
    AnthropicFacetExtractor,
    FacetExtractionError,
)
from ccutils.etl.facets.extractor import FacetSpec, SessionInputs


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Retry backoff would make every retry test wait real seconds. Patch
    time.sleep across all tests so the loop runs at memory speed."""
    monkeypatch.setattr("time.sleep", lambda *_: None)


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
        description="One- or two-sentence summary.",
    )


@pytest.fixture
def f22_spec():
    return FacetSpec(
        facet_id="F22",
        facet_name="blocker_type",
        output_type="enum",
        prompt_version="v1",
        description="What stopped progress.",
        enum_values=("none", "knowledge", "environment", "tool-limit",
                     "unclear-req", "external-dep"),
    )


@pytest.fixture
def extractor():
    # Low max_retries keeps the retry tests fast and bounded.
    return AnthropicFacetExtractor(
        api_key="sk-ant-test",
        model="claude-haiku-4-5-20251001",
        max_retries=2,
    )


def _api_response(json_text: str):
    """Shape of a successful Anthropic /v1/messages response."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5-20251001",
        "content": [{"type": "text", "text": json_text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100, "output_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


class TestFencedResponse:
    """Models wrap JSON in a markdown code fence; the parser must survive it.

    Claim: delete these and a fenced response hard-fails every facet, burns
    a retry that returns the identical text, and falls back. Observed live
    against Haiku -- the response was a valid object inside a ```json fence
    and F20 came back as a fallback with the JSON sitting in raw_response.
    """

    def test_json_fence_is_stripped(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        httpx_mock.add_response(
            json=_api_response(
                '```json\n{"F20": "fixed a flaky test"}\n```'
            ),
        )
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].value == "fixed a flaky test"
        assert out["F20"].is_fallback is False

    def test_bare_fence_is_stripped(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        httpx_mock.add_response(
            json=_api_response('```\n{"F20": "did the thing"}\n```'),
        )
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].value == "did the thing"
        assert out["F20"].is_fallback is False

    def test_fenced_response_costs_no_retry(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        """It parsed on the first attempt, so retry_count must stay 0."""
        httpx_mock.add_response(
            json=_api_response('```json\n{"F20": "no retry"}\n```'),
        )
        out = extractor.extract(session_inputs, [f20_spec])
        assert json.loads(out["F20"].metadata_json)["retry_count"] == 0

    def test_unfenced_response_still_parses(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        """The strip must not disturb the normal path."""
        httpx_mock.add_response(json=_api_response('{"F20": "plain"}'))
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].value == "plain"

    def test_genuine_garbage_still_hard_fails(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        """Stripping fences must not turn a parse failure into a false pass."""
        httpx_mock.add_response(json=_api_response("not json at all"))
        httpx_mock.add_response(json=_api_response("still not json"))
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].is_fallback is True


class TestHappyPath:
    def test_single_facet(self, httpx_mock, extractor, session_inputs, f20_spec):
        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            json=_api_response('{"F20": "fixed a flaky test"}'),
        )
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].value == "fixed a flaky test"
        assert out["F20"].is_fallback is False
        assert out["F20"].prompt_version == "v1"

    def test_multiple_facets(
        self, httpx_mock, extractor, session_inputs, f20_spec, f22_spec
    ):
        httpx_mock.add_response(
            json=_api_response(
                '{"F20": "fixed a flaky test", "F22": "knowledge"}'
            ),
        )
        out = extractor.extract(session_inputs, [f20_spec, f22_spec])
        assert out["F20"].value == "fixed a flaky test"
        assert out["F22"].value == "knowledge"
        assert all(not o.is_fallback for o in out.values())

    def test_metadata_includes_raw_response_and_retry_count(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        httpx_mock.add_response(
            json=_api_response('{"F20": "summary"}'),
        )
        out = extractor.extract(session_inputs, [f20_spec])
        meta = json.loads(out["F20"].metadata_json)
        assert meta["retry_count"] == 0
        assert "raw_response" in meta
        assert "F20" in meta["raw_response"]
        assert "cache_hit" in meta

    def test_metadata_schema_matches_documented_contract(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        # Schema per protocol §3.1 -- downstream QA queries depend on
        # these field names being stable. Adding fields is fine; removing
        # or renaming requires a coordinated change.
        httpx_mock.add_response(json=_api_response('{"F20": "ok"}'))
        out = extractor.extract(session_inputs, [f20_spec])
        meta = json.loads(out["F20"].metadata_json)
        for key in (
            "raw_response", "prompt_version", "retry_count",
            "cache_hit", "input_tokens", "output_tokens", "latency_ms",
        ):
            assert key in meta, f"metadata schema missing {key}"
        # prompt_version threads through from the spec to the metadata
        # so logs can GROUP BY prompt_version cleanly.
        assert meta["prompt_version"] == "v1"
        # Tokens come from the API response usage block (100 / 20 in
        # _api_response helper).
        assert meta["input_tokens"] == 100
        assert meta["output_tokens"] == 20
        # latency_ms is non-negative (time.sleep is patched to no-op so
        # the lower bound is just "set, not None").
        assert meta["latency_ms"] >= 0


class TestMalformedOutput:
    def test_invalid_json_falls_back_after_retries(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        # Both attempts return non-JSON => all fallback.
        httpx_mock.add_response(json=_api_response("this is not json"))
        httpx_mock.add_response(json=_api_response("still not json"))
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].is_fallback is True
        assert out["F20"].value is None
        meta = json.loads(out["F20"].metadata_json)
        assert meta["retry_count"] == 1

    def test_invalid_enum_falls_back_for_offending_only(
        self, httpx_mock, extractor, session_inputs, f20_spec, f22_spec
    ):
        # F22 returns a value not in the allowed enum; F20 is fine.
        # After one retry with the same bad value, F22 falls back but
        # F20 keeps its value (per-facet granularity).
        bad = '{"F20": "fixed it", "F22": "hallucinated-blocker"}'
        httpx_mock.add_response(json=_api_response(bad))
        httpx_mock.add_response(json=_api_response(bad))
        out = extractor.extract(session_inputs, [f20_spec, f22_spec])
        assert out["F20"].is_fallback is False
        assert out["F20"].value == "fixed it"
        assert out["F22"].is_fallback is True
        assert out["F22"].value is None

    def test_missing_facet_key_falls_back(
        self, httpx_mock, extractor, session_inputs, f20_spec, f22_spec
    ):
        # Response only has F20; F22 is silently missing.
        httpx_mock.add_response(json=_api_response('{"F20": "ok"}'))
        out = extractor.extract(session_inputs, [f20_spec, f22_spec])
        assert out["F20"].value == "ok"
        assert out["F22"].is_fallback is True
        assert out["F22"].value is None

    def test_hallucinated_extra_field_ignored(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        # F99 wasn't enabled; the response includes it. Drop silently.
        httpx_mock.add_response(
            json=_api_response('{"F20": "ok", "F99": "hallucination"}'),
        )
        out = extractor.extract(session_inputs, [f20_spec])
        assert "F99" not in out
        assert out["F20"].value == "ok"

    def test_empty_string_for_text_facet_is_fallback(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        # Empty value for a required text facet -> treated as
        # "couldn't extract" rather than a literal empty string.
        httpx_mock.add_response(json=_api_response('{"F20": ""}'))
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].is_fallback is True
        assert out["F20"].value is None

    def test_explicit_null_is_fallback(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        # The model says "I cannot extract." Honored as a genuine null
        # (still is_fallback so downstream queries can filter).
        httpx_mock.add_response(json=_api_response('{"F20": null}'))
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].is_fallback is True
        assert out["F20"].value is None


class TestRetryPolicy:
    def test_429_retries_with_backoff(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        # First call: 429. Second: 200 with valid JSON.
        httpx_mock.add_response(status_code=429, headers={"retry-after": "1"})
        httpx_mock.add_response(json=_api_response('{"F20": "succeeded on retry"}'))
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].value == "succeeded on retry"
        assert out["F20"].is_fallback is False

    def test_401_raises_no_retry(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        # Auth failure is non-retryable; raise so the CLI can exit
        # cleanly rather than burn the rest of the corpus.
        httpx_mock.add_response(status_code=401, json={"error": "bad key"})
        with pytest.raises(FacetExtractionError) as exc:
            extractor.extract(session_inputs, [f20_spec])
        assert "401" in str(exc.value) or "auth" in str(exc.value).lower()

    def test_5xx_retries_then_falls_back(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        # Three 503s, max_retries=2 => 3 total attempts, all 503.
        for _ in range(3):
            httpx_mock.add_response(status_code=503)
        out = extractor.extract(session_inputs, [f20_spec])
        assert out["F20"].is_fallback is True


class TestRequestShape:
    def test_system_prompt_carries_privacy_guardrail(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        httpx_mock.add_response(json=_api_response('{"F20": "ok"}'))
        extractor.extract(session_inputs, [f20_spec])
        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        # System is a list of content blocks; the first carries the
        # cache_control marker so subsequent calls hit the prompt cache.
        assert isinstance(body["system"], list)
        system_text = body["system"][0]["text"]
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
        # Privacy contract wording must reach the model.
        assert "Omit specific names" in system_text or "Omit" in system_text
        assert "F20" in system_text  # facet schema is embedded

    def test_user_prompt_has_xml_sentinels(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        httpx_mock.add_response(json=_api_response('{"F20": "ok"}'))
        extractor.extract(session_inputs, [f20_spec])
        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        user_text = body["messages"][0]["content"][0]["text"]
        # Sentinel-wrapped session inputs.
        assert "<first_user_message>" in user_text
        assert "</first_user_message>" in user_text
        assert "<last_assistant_message>" in user_text
        assert "<tool_mix>" in user_text
        # The actual inputs are inside the sentinels.
        assert "please debug" in user_text
        assert "Bash×3" in user_text

    def test_auth_header(
        self, httpx_mock, extractor, session_inputs, f20_spec
    ):
        httpx_mock.add_response(json=_api_response('{"F20": "ok"}'))
        extractor.extract(session_inputs, [f20_spec])
        request = httpx_mock.get_requests()[0]
        # Anthropic API expects x-api-key, not Authorization Bearer.
        assert request.headers.get("x-api-key") == "sk-ant-test"
        assert request.headers.get("anthropic-version") is not None


class TestTruncation:
    def test_long_first_user_message_is_truncated_to_prefix(
        self, httpx_mock, extractor, f20_spec
    ):
        long_first = "a" * 2000 + " END"
        inputs = SessionInputs(
            session_id="s2",
            first_user_message=long_first,
            last_assistant_message="short",
            tool_mix_summary="",
            model_used=None,
            duration_seconds=None,
        )
        httpx_mock.add_response(json=_api_response('{"F20": "ok"}'))
        extractor.extract(inputs, [f20_spec])
        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        user_text = body["messages"][0]["content"][0]["text"]
        # The "END" sentinel must NOT survive the truncation (it's
        # past the 800-char prefix cut).
        assert " END" not in user_text
        assert "[truncated]" in user_text

    def test_long_last_assistant_message_is_truncated_to_suffix(
        self, httpx_mock, extractor, f20_spec
    ):
        long_last = "START " + "b" * 2000
        inputs = SessionInputs(
            session_id="s3",
            first_user_message="short",
            last_assistant_message=long_last,
            tool_mix_summary="",
            model_used=None,
            duration_seconds=None,
        )
        httpx_mock.add_response(json=_api_response('{"F20": "ok"}'))
        extractor.extract(inputs, [f20_spec])
        request = httpx_mock.get_requests()[0]
        body = json.loads(request.content)
        user_text = body["messages"][0]["content"][0]["text"]
        # The "START" sentinel is at the prefix, which is dropped
        # when truncating to suffix.
        assert "START " not in user_text
        assert "[truncated]" in user_text
