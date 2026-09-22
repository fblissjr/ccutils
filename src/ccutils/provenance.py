"""Record-source allow-list: the provenance labels any tier may stamp.

Lives outside etl/ and schemas/ so Tier 1 writers under parsers/ can validate
their labels without importing the warehouse layers (etl.lineage pulls in the
star-schema package). `ccutils.etl.lineage` re-exports both names, so
warehouse code keeps importing them from there.
"""

from __future__ import annotations

# Provenance label allow-list. Add new values here when a new source goes live.
_RECORD_SOURCES: frozenset[str] = frozenset({
    "claude_code_jsonl",   # Tier 0 Claude Code project session JSONL
    "history_jsonl",       # Claude Code prompt-history JSONL
    "claude_ai_export",    # Claude.ai account export
    "derived_post_etl",    # DAG-invariant facts derived from other facts
    "claude_code_memory",  # Claude Code auto-memory markdown directories
    # Antigravity raw lake (docs/ANTIGRAVITY_CONTRACT.md). One label per
    # kind of source file, so a lake row says which Tier 0 file it mirrors.
    "antigravity_conversation_db",    # conversations/<id>.db table rows
    "antigravity_summaries_db",       # conversation_summaries.db rows
    "antigravity_conversation_file",  # brain/<id>/** and browser_recordings/<id>/**
    "antigravity_store_file",         # annotations, history.jsonl, *_summaries_proto.pb
    "antigravity_encrypted",          # legacy conversations/*.pb and implicit/*.pb ciphertext
    "antigravity_config_projects",    # config/projects/*.json
})


def record_source_label(name: str) -> str:
    if name not in _RECORD_SOURCES:
        raise ValueError(
            f"Unknown record_source {name!r}. Add it to _RECORD_SOURCES in "
            f"ccutils/provenance.py if it's a new legitimate source."
        )
    return name
