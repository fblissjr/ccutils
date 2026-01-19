"""Tests for Claude.ai export parser.

Tests the adapter that converts Claude.ai account exports (from Settings > Privacy)
to the ccutils normalized loglines format.
"""

import json
import pytest
from pathlib import Path


# Import will fail until we implement the parser
from ccutils.parsers.claude_ai import (
    load_export_files,
    convert_message_to_logline,
    convert_conversation_to_loglines,
    convert_content_block,
    parse_claude_ai_export,
)


# --- Test Fixtures ---


@pytest.fixture
def sample_text_block():
    """A basic text content block from Claude.ai export."""
    return {
        "start_timestamp": "2026-01-06T21:03:55.204746Z",
        "stop_timestamp": "2026-01-06T21:03:55.204746Z",
        "flags": None,
        "type": "text",
        "text": "Hello, I need help with something.",
        "citations": [],
    }


@pytest.fixture
def sample_thinking_block():
    """A thinking content block with summaries."""
    return {
        "start_timestamp": "2026-01-06T21:03:57.539260Z",
        "stop_timestamp": "2026-01-06T21:04:15.298243Z",
        "flags": None,
        "type": "thinking",
        "thinking": "Let me analyze this request carefully...",
        "summaries": [
            {"summary": "Analyzing the user request."},
            {"summary": "Considering different approaches."},
        ],
        "cut_off": False,
        "alternative_display_type": None,
    }


@pytest.fixture
def sample_tool_use_block():
    """A tool_use content block."""
    return {
        "start_timestamp": "2026-01-10T18:51:19.867951Z",
        "stop_timestamp": "2026-01-10T18:51:20.343602Z",
        "flags": None,
        "type": "tool_use",
        "id": "toolu_123",
        "name": "project_knowledge_search",
        "input": {"query": "test query", "max_results": 10},
        "message": "Searching project",
        "integration_name": "Search Project Knowledge",
        "integration_icon_url": None,
        "context": None,
        "display_content": None,
        "approval_options": None,
        "approval_key": None,
    }


@pytest.fixture
def sample_tool_result_block():
    """A tool_result content block."""
    return {
        "start_timestamp": None,
        "stop_timestamp": None,
        "flags": None,
        "type": "tool_result",
        "tool_use_id": "toolu_123",
        "name": "project_knowledge_search",
        "content": "Found 3 results: ...",
        "is_error": False,
    }


@pytest.fixture
def sample_human_message(sample_text_block):
    """A human (user) message."""
    return {
        "uuid": "019b951f-9af0-71be-86b2-5484cc86644c",
        "text": "Hello, I need help with something.",
        "content": [sample_text_block],
        "sender": "human",
        "created_at": "2026-01-06T21:03:55.210564Z",
        "updated_at": "2026-01-06T21:03:55.210564Z",
        "attachments": [],
        "files": [],
    }


@pytest.fixture
def sample_assistant_message(sample_thinking_block, sample_text_block):
    """An assistant message with thinking and text."""
    return {
        "uuid": "019b951f-9af0-71be-86b2-response123",
        "text": "Hello, I need help with something.",
        "content": [sample_thinking_block, sample_text_block],
        "sender": "assistant",
        "created_at": "2026-01-06T21:04:00.000000Z",
        "updated_at": "2026-01-06T21:04:15.000000Z",
        "attachments": [],
        "files": [],
    }


@pytest.fixture
def sample_conversation(sample_human_message, sample_assistant_message):
    """A complete conversation with messages."""
    return {
        "uuid": "d3dc7225-1cfd-4e73-90c4-9fc54cbf7f87",
        "name": "Test conversation",
        "summary": "A test conversation for unit tests",
        "created_at": "2026-01-06T21:03:50.592817Z",
        "updated_at": "2026-01-06T21:14:03.149457Z",
        "account": {"uuid": "user-123", "full_name": "Test User"},
        "chat_messages": [sample_human_message, sample_assistant_message],
    }


@pytest.fixture
def export_dir(tmp_path, sample_conversation):
    """Create a temporary export directory with test data."""
    # conversations.json
    conversations = [sample_conversation]
    (tmp_path / "conversations.json").write_text(json.dumps(conversations))

    # projects.json
    projects = [
        {
            "uuid": "proj-123",
            "name": "Test Project",
            "description": "A test project",
            "is_private": True,
            "created_at": "2025-04-06T20:12:50.935903+00:00",
            "updated_at": "2025-04-06T20:12:50.935903+00:00",
        }
    ]
    (tmp_path / "projects.json").write_text(json.dumps(projects))

    # users.json
    users = [
        {
            "uuid": "user-123",
            "full_name": "Test User",
            "email_address": "test@example.com",
        }
    ]
    (tmp_path / "users.json").write_text(json.dumps(users))

    # memories.json
    memories = [
        {
            "conversations_memory": "Test memory content",
            "account_uuid": "user-123",
        }
    ]
    (tmp_path / "memories.json").write_text(json.dumps(memories))

    return tmp_path


# --- Content Block Conversion Tests ---


class TestConvertContentBlock:
    """Tests for content block conversion."""

    def test_text_block(self, sample_text_block):
        """Text blocks preserve text content."""
        result = convert_content_block(sample_text_block)
        assert result["type"] == "text"
        assert result["text"] == "Hello, I need help with something."
        # Should not include Claude.ai-specific fields
        assert "start_timestamp" not in result
        assert "citations" not in result

    def test_thinking_block(self, sample_thinking_block):
        """Thinking blocks preserve thinking content."""
        result = convert_content_block(sample_thinking_block)
        assert result["type"] == "thinking"
        assert result["thinking"] == "Let me analyze this request carefully..."
        # Summaries should be preserved as metadata
        assert "_summaries" in result
        assert len(result["_summaries"]) == 2

    def test_tool_use_block(self, sample_tool_use_block):
        """Tool use blocks map to standard format."""
        result = convert_content_block(sample_tool_use_block)
        assert result["type"] == "tool_use"
        assert result["name"] == "project_knowledge_search"
        assert result["input"] == {"query": "test query", "max_results": 10}
        # ID might be None in Claude.ai exports
        assert "id" in result

    def test_tool_result_block(self, sample_tool_result_block):
        """Tool result blocks map to standard format."""
        result = convert_content_block(sample_tool_result_block)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_123"
        assert result["content"] == "Found 3 results: ..."
        assert result["is_error"] is False


# --- Message Conversion Tests ---


class TestConvertMessageToLogline:
    """Tests for message to logline conversion."""

    def test_human_message_becomes_user_type(self, sample_human_message):
        """Human messages convert to user type."""
        result = convert_message_to_logline(sample_human_message, "session-123")
        assert result["type"] == "user"
        assert result["sessionId"] == "session-123"
        assert result["uuid"] == sample_human_message["uuid"]
        assert result["timestamp"] == sample_human_message["created_at"]

    def test_assistant_message_type(self, sample_assistant_message):
        """Assistant messages keep assistant type."""
        result = convert_message_to_logline(sample_assistant_message, "session-123")
        assert result["type"] == "assistant"
        assert result["sessionId"] == "session-123"

    def test_message_content_blocks_converted(self, sample_assistant_message):
        """Content blocks in messages are converted."""
        result = convert_message_to_logline(sample_assistant_message, "session-123")
        content = result["message"]["content"]
        assert isinstance(content, list)
        # Should have thinking and text blocks
        block_types = [b["type"] for b in content]
        assert "thinking" in block_types
        assert "text" in block_types

    def test_message_role_is_set(self, sample_human_message):
        """Message role is set correctly."""
        result = convert_message_to_logline(sample_human_message, "session-123")
        assert result["message"]["role"] == "user"


# --- Conversation Conversion Tests ---


class TestConvertConversationToLoglines:
    """Tests for full conversation conversion."""

    def test_returns_list_of_loglines(self, sample_conversation):
        """Conversion returns a list of loglines."""
        result = convert_conversation_to_loglines(sample_conversation)
        assert isinstance(result, list)
        assert len(result) == 2  # human + assistant messages

    def test_session_id_from_conversation_uuid(self, sample_conversation):
        """Session ID is taken from conversation UUID."""
        result = convert_conversation_to_loglines(sample_conversation)
        for logline in result:
            assert logline["sessionId"] == sample_conversation["uuid"]

    def test_messages_in_chronological_order(self, sample_conversation):
        """Messages are returned in order."""
        result = convert_conversation_to_loglines(sample_conversation)
        assert result[0]["type"] == "user"
        assert result[1]["type"] == "assistant"


# --- Export Loading Tests ---


class TestLoadExportFiles:
    """Tests for loading export files from directory."""

    def test_loads_all_files(self, export_dir):
        """All four export files are loaded."""
        result = load_export_files(export_dir)
        assert "conversations" in result
        assert "projects" in result
        assert "users" in result
        assert "memories" in result

    def test_missing_conversations_raises_error(self, tmp_path):
        """Missing conversations.json raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_export_files(tmp_path)

    def test_optional_files_default_to_empty(self, tmp_path):
        """Optional files (memories, etc.) default to empty lists."""
        (tmp_path / "conversations.json").write_text("[]")
        result = load_export_files(tmp_path)
        assert result["conversations"] == []
        assert result["projects"] == []  # Optional, defaults to []


# --- Main Parser Tests ---


class TestParseClaudeAiExport:
    """Tests for the main parse_claude_ai_export function."""

    def test_returns_loglines_format(self, export_dir):
        """Returns data in {loglines: [...]} format."""
        result = parse_claude_ai_export(export_dir)
        assert "loglines" in result
        assert isinstance(result["loglines"], list)

    def test_includes_metadata(self, export_dir):
        """Includes metadata about the export."""
        result = parse_claude_ai_export(export_dir)
        assert "_metadata" in result
        assert "source" in result["_metadata"]
        assert result["_metadata"]["source"] == "claude_ai_export"

    def test_all_conversations_converted(self, export_dir):
        """All conversations are converted to loglines."""
        result = parse_claude_ai_export(export_dir)
        # Our fixture has 1 conversation with 2 messages
        assert len(result["loglines"]) == 2

    def test_conversation_filter(self, export_dir):
        """Can filter by conversation UUID."""
        result = parse_claude_ai_export(
            export_dir, conversation_ids=["d3dc7225-1cfd-4e73-90c4-9fc54cbf7f87"]
        )
        assert len(result["loglines"]) == 2

        # Non-existent ID returns empty
        result = parse_claude_ai_export(export_dir, conversation_ids=["nonexistent"])
        assert len(result["loglines"]) == 0

    def test_preserves_projects_in_metadata(self, export_dir):
        """Projects are preserved in metadata for reference."""
        result = parse_claude_ai_export(export_dir)
        assert "projects" in result["_metadata"]
        assert len(result["_metadata"]["projects"]) == 1

    def test_preserves_memories_in_metadata(self, export_dir):
        """Memories are preserved in metadata."""
        result = parse_claude_ai_export(export_dir)
        assert "memories" in result["_metadata"]


# --- Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_conversation(self, tmp_path):
        """Empty conversation with no messages."""
        conv = {
            "uuid": "empty-conv",
            "name": "Empty",
            "chat_messages": [],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        (tmp_path / "conversations.json").write_text(json.dumps([conv]))
        result = parse_claude_ai_export(tmp_path)
        assert result["loglines"] == []

    def test_message_with_empty_content(self, tmp_path):
        """Message with empty content array."""
        conv = {
            "uuid": "conv-1",
            "name": "Test",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "content": [],
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        (tmp_path / "conversations.json").write_text(json.dumps([conv]))
        result = parse_claude_ai_export(tmp_path)
        # Message should still be included
        assert len(result["loglines"]) == 1
        assert result["loglines"][0]["message"]["content"] == []

    def test_tool_use_with_none_id(self, tmp_path):
        """Tool use blocks can have None as ID in Claude.ai exports."""
        conv = {
            "uuid": "conv-1",
            "name": "Test",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": None,  # Claude.ai sometimes has None here
                            "name": "some_tool",
                            "input": {},
                        }
                    ],
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
        (tmp_path / "conversations.json").write_text(json.dumps([conv]))
        result = parse_claude_ai_export(tmp_path)
        # Should handle gracefully
        tool_block = result["loglines"][0]["message"]["content"][0]
        assert tool_block["type"] == "tool_use"
        # ID should be None or generated
        assert "id" in tool_block
