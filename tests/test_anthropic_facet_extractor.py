"""Tests for AnthropicFacetExtractor.

Mocks the HTTP layer (via pytest-httpx), not the API SDK -- we want to
exercise the real httpx code path with canned responses, so the tests
catch issues in request shape (cache_control wiring, system/user
split, JSON schema in the prompt) as well as response parsing.

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
