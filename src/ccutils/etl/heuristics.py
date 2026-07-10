"""Zero-dependency session classifiers for dim_session enrichment.

Ported from the v0.14 schemas.star.heuristics module. The classifiers
themselves are unchanged -- the only difference is that v0.15 runs them
from a post-fact populator (populate_dim_session_heuristics) that reads
inputs out of v0.15 facts instead of from the legacy per-session ETL
extraction result.

No LLM, no API key. Pure keyword + scoring rules.
"""

from __future__ import annotations

import re

# Intent keywords -- score-based: count matches per intent and pick the
# winner with ties broken by priority (bug_fix > refactor > debug > test
# > docs > review > feature).
_INTENT_RULES = [
    ("bug_fix", re.compile(r"\b(fix|bug|broken|error|crash|crashes)\b", re.I)),
    ("refactor", re.compile(r"\b(refactor|clean|reorganize)\b", re.I)),
    ("debug", re.compile(r"\b(debug|investigate|why|trace)\b", re.I)),
    ("test", re.compile(r"\b(tests?|spec|coverage)\b", re.I)),
    ("docs", re.compile(r"\b(doc|docs|documentation|readme|comment|explain)\b", re.I)),
    ("review", re.compile(r"\b(review|check|audit)\b", re.I)),
    ("feature", re.compile(r"\b(add|new|feature|implement|create)\b", re.I)),
]

_DOMAIN_MAP = {
    "web": {".tsx", ".jsx", ".css", ".scss", ".html", ".vue", ".svelte", ".js", ".ts"},
    "backend": {".py", ".rs", ".go", ".java", ".rb"},
    "data": {".sql", ".parquet", ".csv"},
    "devops": {".yaml", ".yml", ".tf", ".dockerfile", ".sh"},
    "docs": {".md", ".rst", ".txt"},
}

_SUCCESS_PATTERN = re.compile(
    r"\b(done|completed|fixed|finished|created|resolved)\b", re.I
)
_FAILURE_PATTERN = re.compile(
    r"\b(error|failed|couldn't|cannot|unable)\b", re.I
)

# Error type: first match wins.
_ERROR_RULES = [
    ("permission_denied", re.compile(r"permission denied|EACCES", re.I)),
    ("file_not_found", re.compile(r"not found|ENOENT|no such file", re.I)),
    ("syntax_error", re.compile(r"syntax error|SyntaxError", re.I)),
    ("timeout", re.compile(r"timeout|ETIMEDOUT", re.I)),
    ("import_error", re.compile(r"ImportError|ModuleNotFoundError", re.I)),
]


def classify_intent(first_user_message: str | None) -> str:
    """Classify session intent from the first user message.

    Returns one of: bug_fix, feature, refactor, debug, test, docs, review,
    explore.
    """
    if not first_user_message:
        return "explore"

    scores = {}
    for intent, pattern in _INTENT_RULES:
        hits = len(pattern.findall(first_user_message))
        if hits > 0:
            scores[intent] = hits

    if not scores:
        return "explore"

    max_score = max(scores.values())
    # Priority order is the iteration order of _INTENT_RULES, used as tiebreaker.
    for intent, _ in _INTENT_RULES:
        if scores.get(intent) == max_score:
            return intent
    return "explore"


def classify_complexity(
    tool_count: int, msg_count: int, agent_depth: int | None, error_count: int
) -> str:
    """Classify session complexity from metrics.

    Returns one of: trivial, simple, moderate, complex.
    """
    score = 0
    if tool_count > 20:
        score += 2
    elif tool_count > 2:
        score += 1

    if msg_count > 30:
        score += 2
    elif msg_count > 8:
        score += 1

    depth = agent_depth if agent_depth is not None else 0
    if depth > 0:
        score += 2

    if error_count > 3:
        score += 1

    if score >= 5:
        return "complex"
    if score >= 3:
        return "moderate"
    if score >= 1:
        return "simple"
    return "trivial"


def classify_outcome(
    last_assistant_text: str | None, error_rate: float = 0.0
) -> str:
    """Classify session outcome from last assistant message and error rate.

    Returns one of: success, failure, unknown.
    """
    if error_rate > 0.5:
        return "failure"
    if not last_assistant_text:
        return "unknown"
    if _SUCCESS_PATTERN.search(last_assistant_text):
        return "success"
    if _FAILURE_PATTERN.search(last_assistant_text):
        return "failure"
    return "unknown"


def classify_error_type(error_message: str | None) -> str:
    """Classify a tool error message into one of a small set of categories.

    Returns one of: permission_denied, file_not_found, syntax_error,
    timeout, import_error, tool_error.
    """
    if not error_message:
        return "tool_error"
    for error_type, pattern in _ERROR_RULES:
        if pattern.search(error_message):
            return error_type
    return "tool_error"


def classify_domain(file_extensions: list[str]) -> str:
    """Classify session domain from file extensions touched.

    Returns one of: web, backend, data, devops, docs, mixed, unknown.
    """
    if not file_extensions:
        return "unknown"

    scores: dict[str, int] = {}
    for ext in file_extensions:
        ext_lower = ext.lower()
        if not ext_lower.startswith("."):
            ext_lower = "." + ext_lower
        for domain, exts in _DOMAIN_MAP.items():
            if ext_lower in exts:
                scores[domain] = scores.get(domain, 0) + 1

    if not scores:
        return "unknown"

    max_score = max(scores.values())
    winners = [d for d, s in scores.items() if s == max_score]
    return winners[0] if len(winners) == 1 else "mixed"
