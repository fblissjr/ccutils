"""Shared utilities for v0.15 ETL populators.

Functions here are used by multiple populators (currently
`dim_session_heuristics` and `facets.populator`) and don't fit any
single populator's module. The bar for adding something here is "needed
by at least two populators."
"""

from __future__ import annotations

import json


def extract_text_from_content_json(content_json_raw: str | None) -> str:
    """Pull plain text out of a `message.content` JSON payload.

    Claude Code emits user content as either a bare JSON string or a
    list of content blocks; assistant content is always a list of
    blocks. This helper concatenates every text-bearing block
    (`type='text'` and `type='thinking'`); `tool_result` and other
    block types are skipped because they're tool output, not user
    intent or assistant conclusion.

    Returns an empty string when the input is None, unparseable, or
    contains no text blocks. The empty-string fallback (vs. None) lets
    downstream classifiers and prompt builders treat "no extractable
    text" and "literally empty text" the same way.
    """
    if not content_json_raw:
        return ""
    try:
        content = json.loads(content_json_raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "thinking":
                parts.append(block.get("thinking", ""))
        return " ".join(p for p in parts if p)
    return ""
