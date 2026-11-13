"""Transcript text that reaches HTML through a pre-escaped `|safe` slot is escaped.

Autoescape guards every value a template interpolates, except the slots the
templates mark `|safe` because Python already built the HTML: tool results
(string, a list of text blocks, the text around a commit card) and tool
input. Those are escaped with `html.escape` in `export/html.py`, and on
2026-11-13 the test audit removed that call from the string result and the
whole suite stayed green. Tool output is the most attacker-shaped text a
transcript holds: whatever a command printed, a page fetched, a file held.

Each case plants a payload in one channel and asserts two things: the raw
markup never appears, and its escaped form does (so the payload reached the
renderer rather than being dropped, which would pass the first check alone).

What breaks without it: a dropped `html.escape` before a `|safe` slot ships
stored XSS in every exported transcript holding hostile tool output.
"""

import pytest

from ccutils import render_content_block

PAYLOAD = '<script>alert(1)</script><img src=x onerror="alert(2)">'
ESCAPED = "&lt;script&gt;alert(1)&lt;/script&gt;"

CASES = {
    "string result": {"type": "tool_result", "tool_use_id": "t1", "content": PAYLOAD},
    "list of text result": {"type": "tool_result", "tool_use_id": "t1",
                            "content": [{"type": "text", "text": PAYLOAD}]},
    "text before a commit card": {"type": "tool_result", "tool_use_id": "t1",
                                  "content": f"{PAYLOAD}\n[main abc1234] fix the thing\n"},
    "text after a commit card": {"type": "tool_result", "tool_use_id": "t1",
                                 "content": f"[main abc1234] fix the thing\n{PAYLOAD}"},
    "tool input": {"type": "tool_use", "id": "t1", "name": "Bash",
                   "input": {"command": PAYLOAD, "description": PAYLOAD}},
    "generic tool input": {"type": "tool_use", "id": "t1", "name": "SomeNewTool",
                           "input": {"x": PAYLOAD}},
}


@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_a_planted_payload_is_rendered_escaped(case):
    html = render_content_block(CASES[case])
    assert "<script>alert(1)" not in html
    assert '<img src=x onerror' not in html
    assert ESCAPED in html, html[:400]
