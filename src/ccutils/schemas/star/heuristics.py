"""Heuristic classification for star schema sessions.

Runs during ETL after extraction. No LLM required. No API key.
Classifies sessions by intent, complexity, outcome, domain, and error type
using keyword matching and session metrics.
"""

import re

# Intent keywords: first match wins
_INTENT_RULES = [
    ("bug_fix", re.compile(r"\b(fix|bug|broken|error|crash|crashes)\b", re.I)),
    ("refactor", re.compile(r"\b(refactor|clean|reorganize)\b", re.I)),
    ("debug", re.compile(r"\b(debug|investigate|why|trace)\b", re.I)),
    ("test", re.compile(r"\b(tests?|spec|coverage)\b", re.I)),
    ("docs", re.compile(r"\b(doc|docs|documentation|readme|comment|explain)\b", re.I)),
    ("review", re.compile(r"\b(review|check|audit)\b", re.I)),
    ("feature", re.compile(r"\b(add|new|feature|implement|create)\b", re.I)),
]

# Domain: extension -> category
_DOMAIN_MAP = {
    "web": {".tsx", ".jsx", ".css", ".scss", ".html", ".vue", ".svelte"},
    "backend": {".py", ".rs", ".go", ".java", ".rb"},
    "data": {".sql", ".parquet", ".csv"},
    "devops": {".yaml", ".yml", ".tf", ".dockerfile", ".sh"},
    "docs": {".md", ".rst", ".txt"},
}

# Error type: first match wins
_ERROR_RULES = [
    ("permission_denied", re.compile(r"permission denied|EACCES", re.I)),
    ("file_not_found", re.compile(r"not found|ENOENT|no such file", re.I)),
    ("syntax_error", re.compile(r"syntax error|SyntaxError", re.I)),
    ("timeout", re.compile(r"timeout|ETIMEDOUT", re.I)),
    ("import_error", re.compile(r"ImportError|ModuleNotFoundError", re.I)),
]


def classify_intent(first_user_message):
    """Classify session intent from the first user message.

    Returns one of: bug_fix, feature, refactor, debug, test, docs, review, explore.
    """
    if not first_user_message:
        return "explore"

    for intent, pattern in _INTENT_RULES:
        if pattern.search(first_user_message):
            return intent

    return "explore"


def classify_complexity(tool_count, msg_count, agent_depth, error_count):
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

    if agent_depth > 0:
        score += 2

    if error_count > 3:
        score += 1

    if score >= 5:
        return "complex"
    elif score >= 3:
        return "moderate"
    elif score >= 1:
        return "simple"
    return "trivial"


def classify_outcome(last_assistant_text, error_rate=0.0):
    """Classify session outcome from last assistant message and error rate.

    Returns one of: success, failure, unknown.
    """
    if error_rate > 0.5:
        return "failure"

    if not last_assistant_text:
        return "unknown"

    lower = last_assistant_text.lower()

    if re.search(r"\b(done|completed|fixed|finished|created|resolved)\b", lower):
        return "success"

    if re.search(r"\b(error|failed|couldn't|cannot|unable)\b", lower):
        return "failure"

    return "unknown"


def classify_domain(file_extensions):
    """Classify session domain from file extensions touched.

    Returns one of: web, backend, data, devops, docs, mixed, unknown.
    """
    if not file_extensions:
        return "unknown"

    scores = {}
    for ext in file_extensions:
        ext_lower = ext.lower()
        for domain, exts in _DOMAIN_MAP.items():
            if ext_lower in exts:
                scores[domain] = scores.get(domain, 0) + 1

    if not scores:
        return "unknown"

    max_score = max(scores.values())
    winners = [d for d, s in scores.items() if s == max_score]

    if len(winners) == 1:
        return winners[0]
    return "mixed"


def classify_error_type(error_message):
    """Classify error type from error message text.

    Returns one of: permission_denied, file_not_found, syntax_error,
    timeout, import_error, tool_error.
    """
    if not error_message:
        return "tool_error"

    for error_type, pattern in _ERROR_RULES:
        if pattern.search(error_message):
            return error_type

    return "tool_error"
