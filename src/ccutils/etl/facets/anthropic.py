"""Anthropic-API-backed FacetExtractor (Haiku 4.5 by default).

Implements the FacetExtractor protocol defined in
ccutils.etl.facets.extractor. See
internal/plans/facet_extractor_protocol.md for the full design rationale.

What this module does NOT do: build the Tier 2 populator. That's the
caller's job (step 4 of the build order). This module only owns the
boundary -- a SessionInputs goes in, a dict[facet_id -> FacetOutput]
comes out, and per-facet validation + retry semantics are handled
inside.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError as PydanticValidationError,
    create_model,
)

from ccutils.etl.facets.extractor import FacetOutput, FacetSpec, SessionInputs

# A markdown fence wrapping the ENTIRE response. Anchored at both ends on
# purpose: a fence in the middle of prose is not a wrapped payload, and
# stripping it would turn a genuine parse failure into a silent wrong answer.
_FENCE_RE = re.compile(
    r"^\s*```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)\r?\n?[ \t]*```\s*$",
    re.DOTALL,
)


def _strip_code_fence(text: str) -> str:
    """Return `text` with a wrapping markdown code fence removed.

    Models sometimes wrap JSON in ```json ... ``` despite being asked for a
    bare object. Observed live against Haiku: a perfectly valid object inside
    a fence hard-failed every facet, burned a retry that returned the
    identical text, and fell back -- the JSON was sitting intact in
    `raw_response` the whole time.
    """
    match = _FENCE_RE.match(text)
    return match.group(1) if match else text



_log = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# 800 chars ~ 200 tokens. Two text fields × 200 tokens = ~400 tokens of
# session-specific input per call; with system + tool_mix metadata about
# 450 tokens total. Well under any rate ceiling.
_MAX_TEXT_CHARS = 800

_TRUNCATION_SUFFIX = "…[truncated]"


class FacetExtractionError(Exception):
    """Raised on non-retryable failures (auth, malformed request).
    Retryable failures (429, 5xx, network) are absorbed by the retry
    loop; only hard errors escape."""


class _RetryableHTTPError(Exception):
    """Internal: signals the retry loop. Never escapes the module."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"{status_code}: {message}")


# ---------------------------------------------------------------------------
# Dynamic Pydantic response model
# ---------------------------------------------------------------------------


def _annotation_for_spec(spec: FacetSpec):
    """Return the Optional[type] annotation for a facet's value field.

    Enum facets get a dynamically-built Enum class so Pydantic enforces
    the allowed-values list. Non-enum facets get the Optional primitive.
    Unknown types fall through to Optional[object] which accepts
    anything JSON-parseable.
    """
    if spec.output_type == "enum" and spec.enum_values:
        enum_cls = Enum(
            f"E_{spec.facet_id}",
            {v: v for v in spec.enum_values},
        )
        return Optional[enum_cls]
    if spec.output_type in ("text", "enum"):
        return Optional[str]
    if spec.output_type == "int":
        return Optional[int]
    if spec.output_type == "float":
        return Optional[float]
    if spec.output_type == "bool":
        return Optional[bool]
    return Optional[object]


def _build_response_model(specs: list[FacetSpec]) -> type[BaseModel]:
    """One Pydantic model with one optional field per spec. extra='ignore'
    drops hallucinated extra keys silently."""
    fields = {spec.facet_id: (_annotation_for_spec(spec), None) for spec in specs}
    return create_model(
        "FacetResponse",
        __config__=ConfigDict(extra="ignore"),
        **fields,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _truncate_prefix(text: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """Keep the FIRST max_chars of the text. Used for first_user_message
    where the opening tends to be the highest-signal stretch."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + _TRUNCATION_SUFFIX


def _truncate_suffix(text: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    """Keep the LAST max_chars of the text. Used for last_assistant_message
    where conclusions land at the end."""
    if len(text) <= max_chars:
        return text
    return _TRUNCATION_SUFFIX + text[-max_chars:]


def _format_facet_schema(specs: list[FacetSpec]) -> str:
    lines = []
    for spec in specs:
        line = f"- `{spec.facet_id}` ({spec.facet_name}, {spec.output_type}): {spec.description}"
        if spec.output_type == "enum" and spec.enum_values:
            line += f" Allowed: {list(spec.enum_values)}."
        lines.append(line)
    return "\n".join(lines)


def _build_system_prompt(specs: list[FacetSpec]) -> str:
    """Cacheable system prompt. Embeds the facet schema (so changing the
    enabled facet set invalidates the cache by design) and the privacy
    guardrail wording (which is the load-bearing contract for export
    readiness)."""
    return f"""You are an analyst summarizing what happened in software-development sessions between a human and an AI coding assistant. You receive a privacy-sanitized summary of one session and return a JSON object describing it on several structured axes.

Privacy rules — these are NOT optional:
- Describe the session in general terms only.
- Omit specific names, file paths, repository names, organization names, API keys, secrets, personal information, and project-specific identifiers.
- Generalize: say "a Python project" rather than naming the project; say "a database migration" rather than naming the table; say "a web frontend" rather than naming the framework version.
- If you encounter a name or path you can't generalize, omit the detail entirely rather than invent a generic version.

Output format:
- A single JSON object. Keys are the facet identifiers below. No prose before or after the JSON. No markdown code fences.
- For enum-valued facets, the value MUST be one of the listed options.
- If you genuinely cannot extract a facet from the input, set its value to null. Do not invent.

Facets to extract this call:
{_format_facet_schema(specs)}
"""


def _build_user_prompt(inputs: SessionInputs) -> str:
    """Per-session user prompt. Sentinel-wrapped fields give the model
    explicit boundaries."""
    parts = [
        "<first_user_message>",
        _truncate_prefix(inputs.first_user_message),
        "</first_user_message>",
        "",
        "<last_assistant_message>",
        _truncate_suffix(inputs.last_assistant_message),
        "</last_assistant_message>",
        "",
        f"<tool_mix>{inputs.tool_mix_summary}</tool_mix>",
    ]
    if inputs.model_used:
        parts.append(f"<model_used>{inputs.model_used}</model_used>")
    if inputs.duration_seconds is not None:
        parts.append(f"<duration_seconds>{inputs.duration_seconds}</duration_seconds>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response validation with per-facet fallback
# ---------------------------------------------------------------------------


@dataclass
class _ValidationResult:
    """Distinguishes "model said it couldn't extract" (soft_failed) from
    "the model produced something we couldn't parse" (hard_failed). Both
    end up as fallback rows in the output, but only hard_failed triggers
    a retry — null / missing / empty is the model honestly answering
    'I don't know', not a parse failure to retry through."""

    valid: dict[str, str]       # facet_id -> stringified value
    soft_failed: set[str]       # null / missing / empty
    hard_failed: set[str]       # bad enum / wrong type / JSON error

    @property
    def all_failed(self) -> set[str]:
        return self.soft_failed | self.hard_failed


def _validate_response(
    data: dict, specs: list[FacetSpec], response_model: type[BaseModel]
) -> _ValidationResult:
    """Per-facet validation with soft/hard separation."""
    valid: dict[str, str] = {}
    soft_failed: set[str] = set()
    hard_failed: set[str] = set()

    try:
        model_instance = response_model(**data)
        for spec in specs:
            raw = getattr(model_instance, spec.facet_id, None)
            stringified = _stringify_value(raw)
            if stringified is None:
                soft_failed.add(spec.facet_id)
            else:
                valid[spec.facet_id] = stringified
        return _ValidationResult(valid=valid, soft_failed=soft_failed,
                                 hard_failed=hard_failed)
    except PydanticValidationError as e:
        # ValidationError.errors() gives [{"loc": (facet_id,), ...}, ...]
        offending = {err["loc"][0] for err in e.errors() if err.get("loc")}
        hard_failed = offending
        # Non-offending facets get re-parsed from raw data with the same
        # soft-vs-valid distinction.
        for spec in specs:
            if spec.facet_id in offending:
                continue
            raw = data.get(spec.facet_id)
            stringified = _stringify_value(raw)
            if stringified is None:
                soft_failed.add(spec.facet_id)
            else:
                valid[spec.facet_id] = stringified
        return _ValidationResult(valid=valid, soft_failed=soft_failed,
                                 hard_failed=hard_failed)


def _stringify_value(raw) -> str | None:
    """Convert a validated Pydantic field value to the canonical str the
    populator stores. Returns None for empty / null / unparseable, which
    the caller treats as fallback."""
    if raw is None:
        return None
    if isinstance(raw, Enum):
        return str(raw.value)
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if isinstance(raw, (int, float)):
        return str(raw)
    if isinstance(raw, str):
        return raw if raw else None  # empty string -> None -> fallback
    # json catch-all: serialize compactly
    return json.dumps(raw, separators=(",", ":"))


# ---------------------------------------------------------------------------
# The extractor
# ---------------------------------------------------------------------------


class AnthropicFacetExtractor:
    """Production FacetExtractor. Talks to api.anthropic.com via httpx;
    no SDK dependency. Retry, backoff, and per-facet fallback all happen
    inside extract(); callers see a clean dict[facet_id -> FacetOutput]."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = _API_URL,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        # HTTP retry budget (429, 5xx, network errors). Validation retry
        # budget is fixed at 1 per the design contract in §5.1 of the
        # protocol proposal: "One retry with identical prompt -- Haiku
        # frequently emits cleaner JSON on a second draw."
        self._max_retries = max_retries
        self._max_validation_retries = 1
        self._timeout = timeout

    def extract(
        self,
        session_inputs: SessionInputs,
        enabled_facets: list[FacetSpec],
    ) -> dict[str, FacetOutput]:
        response_model = _build_response_model(enabled_facets)
        payload = self._build_payload(session_inputs, enabled_facets)

        raw_response = ""
        cache_hit = False
        retry_count = 0
        input_tokens = 0
        output_tokens = 0
        all_facet_ids = {s.facet_id for s in enabled_facets}
        result = _ValidationResult(
            valid={}, soft_failed=set(), hard_failed=all_facet_ids
        )

        # latency_ms includes everything inside extract(): HTTP calls,
        # backoff sleeps, validation retries. Wall-clock as the user
        # experiences it.
        start = time.monotonic()

        # `retry_count` reflects the number of retries actually performed
        # (HTTP-level + validation-level), incremented only when we COMMIT
        # to another attempt -- not at attempt conclusion. A first-try
        # success yields retry_count=0.
        for validation_attempt in range(self._max_validation_retries + 1):
            try:
                api_response, http_retries = self._call_api_with_retries(payload)
            except _RetryableHTTPError as e:
                _log.warning(
                    "facet extraction exhausted HTTP retries: session=%s status=%s",
                    session_inputs.session_id, e.status_code,
                )
                retry_count += self._max_retries
                break
            retry_count += http_retries

            raw_response = self._extract_text(api_response)
            cache_hit = self._compute_cache_hit(api_response)
            input_tokens, output_tokens = self._extract_token_counts(api_response)

            try:
                # raw_response is kept verbatim for the metadata audit trail;
                # only the parse sees the unfenced form.
                data = json.loads(_strip_code_fence(raw_response))
                if not isinstance(data, dict):
                    raise json.JSONDecodeError("not an object", raw_response, 0)
                result = _validate_response(data, enabled_facets, response_model)
            except json.JSONDecodeError:
                # JSON-parse failure is a hard fail for every facet.
                result = _ValidationResult(
                    valid={}, soft_failed=set(), hard_failed=all_facet_ids
                )

            if not result.hard_failed:
                # No retry-worthy failures. soft_failed (null / missing /
                # empty) is the model honestly saying "I don't know" and
                # is NOT retried -- those facets fall back as-is.
                break

            if validation_attempt < self._max_validation_retries:
                retry_count += 1
                continue
            break

        latency_ms = int((time.monotonic() - start) * 1000)

        return self._build_outputs(
            enabled_facets, result, raw_response, retry_count, cache_hit,
            input_tokens, output_tokens, latency_ms,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        inputs: SessionInputs,
        specs: list[FacetSpec],
    ) -> dict:
        return {
            "model": self._model,
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": _build_system_prompt(specs),
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _build_user_prompt(inputs)},
                    ],
                },
            ],
        }

    def _call_api_with_retries(self, payload: dict) -> tuple[dict, int]:
        """Returns (response_body, http_retry_count). Raises
        _RetryableHTTPError when retries exhaust, FacetExtractionError on
        non-retryable HTTP failures (auth, etc.)."""
        last_error: _RetryableHTTPError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._call_api_once(payload), attempt
            except _RetryableHTTPError as e:
                last_error = e
                if attempt < self._max_retries:
                    time.sleep(self._backoff(attempt))
                    continue
        # Exhausted retries; bubble the last error up so the validation
        # loop can record the failure.
        raise last_error  # type: ignore[misc]

    def _call_api_once(self, payload: dict) -> dict:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    self._base_url,
                    json=payload,
                    headers={
                        "x-api-key": self._api_key,
                        "anthropic-version": _ANTHROPIC_VERSION,
                        "content-type": "application/json",
                    },
                )
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            raise _RetryableHTTPError(0, str(e)) from e

        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 or response.status_code >= 500:
            raise _RetryableHTTPError(response.status_code, response.text[:200])
        # 4xx other -- auth (401), bad request (400), forbidden (403). Don't
        # retry; surface to caller as FacetExtractionError.
        raise FacetExtractionError(
            f"API call failed with status {response.status_code}: "
            f"{response.text[:200]}"
        )

    @staticmethod
    def _backoff(attempt: int) -> float:
        # 1s, 4s, 16s, ...
        return float(4 ** attempt)

    @staticmethod
    def _extract_text(api_response: dict) -> str:
        content_blocks = api_response.get("content", [])
        for block in content_blocks:
            if block.get("type") == "text":
                return block.get("text", "")
        return ""

    @staticmethod
    def _compute_cache_hit(api_response: dict) -> bool:
        usage = api_response.get("usage", {})
        return usage.get("cache_read_input_tokens", 0) > 0

    @staticmethod
    def _extract_token_counts(api_response: dict) -> tuple[int, int]:
        usage = api_response.get("usage", {})
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    @staticmethod
    def _build_outputs(
        specs: list[FacetSpec],
        result: _ValidationResult,
        raw_response: str,
        retry_count: int,
        cache_hit: bool,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
    ) -> dict[str, FacetOutput]:
        # Schema documented in protocol §3.1; downstream QA queries
        # depend on these field names being stable. prompt_version is
        # per-spec (different facets could carry different versions in
        # principle, though in practice one call uses one registry).
        out: dict[str, FacetOutput] = {}
        for spec in specs:
            metadata = json.dumps(
                {
                    "raw_response": raw_response,
                    "prompt_version": spec.prompt_version,
                    "retry_count": retry_count,
                    "cache_hit": cache_hit,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": latency_ms,
                },
                separators=(",", ":"),
            )
            if spec.facet_id in result.valid:
                out[spec.facet_id] = FacetOutput(
                    facet_id=spec.facet_id,
                    prompt_version=spec.prompt_version,
                    value=result.valid[spec.facet_id],
                    is_fallback=False,
                    metadata_json=metadata,
                )
            else:
                out[spec.facet_id] = FacetOutput(
                    facet_id=spec.facet_id,
                    prompt_version=spec.prompt_version,
                    value=None,
                    is_fallback=True,
                    metadata_json=metadata,
                )
        return out
