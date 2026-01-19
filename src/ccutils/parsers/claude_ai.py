"""Parser adapter for Claude.ai account data exports.

Converts the 4-file export format from Claude.ai (Settings > Privacy)
to the ccutils normalized loglines format.

Export format:
- conversations.json: Messages with content blocks (text, thinking, tool_use, tool_result)
- projects.json: Project metadata
- memories.json: Account-level learned context
- users.json: Account info

Target format:
{
    "loglines": [
        {
            "type": "user" | "assistant",
            "timestamp": "ISO8601",
            "sessionId": "conversation-uuid",
            "uuid": "message-uuid",
            "message": {"role": "user" | "assistant", "content": [...]}
        }
    ],
    "_metadata": {...}
}
"""

import json
from pathlib import Path
from typing import Optional


def convert_content_block(block: dict) -> dict:
    """Convert a Claude.ai content block to ccutils format.

    Claude.ai content blocks have extra fields like start_timestamp, stop_timestamp,
    citations, summaries, etc. We strip these and keep the essential data.

    Args:
        block: A content block from Claude.ai export

    Returns:
        A normalized content block for ccutils
    """
    block_type = block.get("type")

    if block_type == "text":
        return {
            "type": "text",
            "text": block.get("text", ""),
        }

    elif block_type == "thinking":
        result = {
            "type": "thinking",
            "thinking": block.get("thinking", ""),
        }
        # Preserve summaries as metadata (useful for display)
        if block.get("summaries"):
            result["_summaries"] = block["summaries"]
        return result

    elif block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.get("id"),  # May be None in Claude.ai exports
            "name": block.get("name", ""),
            "input": block.get("input", {}),
        }

    elif block_type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block.get("tool_use_id"),
            "content": block.get("content", ""),
            "is_error": block.get("is_error", False),
        }

    else:
        # Unknown block type - pass through with minimal transformation
        return {"type": block_type, **{k: v for k, v in block.items() if k != "type"}}


def convert_message_to_logline(message: dict, session_id: str) -> dict:
    """Convert a single Claude.ai chat_message to logline format.

    Args:
        message: A chat_message from Claude.ai export
        session_id: The conversation UUID to use as sessionId

    Returns:
        A logline dict in ccutils format
    """
    # Map sender: "human" -> type: "user", "assistant" stays "assistant"
    sender = message.get("sender", "")
    msg_type = "user" if sender == "human" else sender

    # Convert content blocks
    content_blocks = []
    for block in message.get("content", []):
        if isinstance(block, dict):
            content_blocks.append(convert_content_block(block))

    return {
        "type": msg_type,
        "timestamp": message.get("created_at", ""),
        "sessionId": session_id,
        "uuid": message.get("uuid", ""),
        "message": {
            "role": msg_type,
            "content": content_blocks,
        },
    }


def convert_conversation_to_loglines(conversation: dict) -> list[dict]:
    """Convert a Claude.ai conversation to list of loglines.

    Args:
        conversation: A conversation object from Claude.ai export

    Returns:
        List of logline dicts for all messages in the conversation
    """
    session_id = conversation.get("uuid", "")
    loglines = []

    for message in conversation.get("chat_messages", []):
        logline = convert_message_to_logline(message, session_id)
        loglines.append(logline)

    return loglines


def load_export_files(export_path: Path) -> dict:
    """Load all JSON files from a Claude.ai export directory.

    Args:
        export_path: Path to the export directory

    Returns:
        Dict with keys: conversations, projects, users, memories

    Raises:
        FileNotFoundError: If conversations.json is missing
    """
    export_path = Path(export_path)

    # conversations.json is required
    conversations_file = export_path / "conversations.json"
    if not conversations_file.exists():
        raise FileNotFoundError(
            f"conversations.json not found in {export_path}. "
            "This doesn't appear to be a valid Claude.ai export."
        )

    with open(conversations_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    # Other files are optional
    def load_optional(filename: str) -> list:
        filepath = export_path / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    return {
        "conversations": conversations,
        "projects": load_optional("projects.json"),
        "users": load_optional("users.json"),
        "memories": load_optional("memories.json"),
    }


def parse_claude_ai_export(
    export_path: Path,
    conversation_ids: Optional[list[str]] = None,
    include_thinking: bool = True,
) -> dict:
    """Parse a Claude.ai export directory into ccutils loglines format.

    This is the main entry point for converting Claude.ai exports.

    Args:
        export_path: Path to the export directory containing the JSON files
        conversation_ids: Optional list of conversation UUIDs to include.
                         If None, all conversations are included.
        include_thinking: Whether to include thinking blocks (default: True)

    Returns:
        Dict with keys:
        - loglines: List of normalized logline dicts
        - _metadata: Export metadata (source, projects, memories, users)
    """
    export_path = Path(export_path)
    data = load_export_files(export_path)

    all_loglines = []

    for conversation in data["conversations"]:
        conv_id = conversation.get("uuid", "")

        # Filter by conversation IDs if specified
        if conversation_ids is not None and conv_id not in conversation_ids:
            continue

        loglines = convert_conversation_to_loglines(conversation)

        # Optionally filter out thinking blocks
        if not include_thinking:
            for logline in loglines:
                content = logline.get("message", {}).get("content", [])
                logline["message"]["content"] = [
                    block for block in content if block.get("type") != "thinking"
                ]

        all_loglines.extend(loglines)

    # Build metadata
    metadata = {
        "source": "claude_ai_export",
        "export_path": str(export_path),
        "projects": data["projects"],
        "memories": data["memories"],
        "users": data["users"],
        "conversation_count": len(data["conversations"]),
    }

    return {
        "loglines": all_loglines,
        "_metadata": metadata,
    }
