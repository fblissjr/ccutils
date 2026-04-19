"""Pytest configuration and fixtures for claude-code-transcripts tests."""

import json
import tempfile
from pathlib import Path

import pytest
import webbrowser


@pytest.fixture(autouse=True)
def mock_webbrowser_open(monkeypatch):
    """Automatically mock webbrowser.open to prevent browsers opening during tests."""
    opened_urls = []

    def mock_open(url):
        opened_urls.append(url)
        return True

    # Patch the stdlib webbrowser.open directly
    monkeypatch.setattr(webbrowser, "open", mock_open)
    return opened_urls


@pytest.fixture
def sample_session_file():
    """Create a sample JSONL session file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # User message
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-001",
                    "parentUuid": None,
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:00.000Z",
                    "cwd": "/home/user/project",
                    "gitBranch": "main",
                    "version": "2.0.0",
                    "message": {
                        "role": "user",
                        "content": "Help me write a hello world program",
                    },
                }
            )
            + "\n"
        )
        # Assistant message with tool_use
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-001",
                    "parentUuid": "user-001",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {"type": "text", "text": "I'll create that for you."},
                            {
                                "type": "tool_use",
                                "id": "tool-001",
                                "name": "Write",
                                "input": {
                                    "file_path": "/home/user/project/hello.py",
                                    "content": "print('Hello, World!')",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # User message with tool_result
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-002",
                    "parentUuid": "asst-001",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-001",
                                "content": "File written successfully",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Assistant message with Read tool
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-002",
                    "parentUuid": "user-002",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:15.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": "The file was created. Let me verify it.",
                            },
                            {"type": "text", "text": "Let me verify the file."},
                            {
                                "type": "tool_use",
                                "id": "tool-002",
                                "name": "Read",
                                "input": {"file_path": "/home/user/project/hello.py"},
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # User message with tool_result for Read
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-003",
                    "parentUuid": "asst-002",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:20.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-002",
                                "content": "print('Hello, World!')",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # Final assistant message
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-003",
                    "parentUuid": "user-003",
                    "sessionId": "session-123",
                    "timestamp": "2025-01-15T10:00:25.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-sonnet-4-20250514",
                        "content": [
                            {
                                "type": "text",
                                "text": "Done! I've created hello.py with a hello world program.",
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        f.flush()
        yield Path(f.name)


@pytest.fixture
def interrupted_session_file(output_dir):
    """Session where the last tool_use never received a result (orphan tool use)."""
    session_file = output_dir / "interrupted.jsonl"
    session_file.write_text(
        json.dumps(
            {
                "type": "user",
                "uuid": "user-001",
                "parentUuid": None,
                "sessionId": "session-interrupted",
                "timestamp": "2025-01-01T10:00:00.000Z",
                "cwd": "/home/user/project",
                "message": {"role": "user", "content": "Read and edit the config"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "uuid": "asst-001",
                "parentUuid": "user-001",
                "sessionId": "session-interrupted",
                "timestamp": "2025-01-01T10:00:05.000Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [
                        {"type": "text", "text": "Let me read that."},
                        {
                            "type": "tool_use",
                            "id": "tool-matched",
                            "name": "Read",
                            "input": {"file_path": "/home/user/project/config.yaml"},
                        },
                    ],
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "user",
                "uuid": "user-002",
                "parentUuid": "asst-001",
                "sessionId": "session-interrupted",
                "timestamp": "2025-01-01T10:00:10.000Z",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool-matched",
                            "content": "key: value",
                        }
                    ],
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "uuid": "asst-002",
                "parentUuid": "user-002",
                "sessionId": "session-interrupted",
                "timestamp": "2025-01-01T10:00:15.000Z",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [
                        {"type": "text", "text": "Now let me edit it."},
                        {
                            "type": "tool_use",
                            "id": "tool-orphan",
                            "name": "Edit",
                            "input": {
                                "file_path": "/home/user/project/config.yaml",
                                "old_string": "key: value",
                                "new_string": "key: new_value",
                            },
                        },
                    ],
                },
            }
        )
        + "\n"
    )
    return session_file


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def new_format_session_file():
    """Create a session file with new-format entries: usage data, system entries, attachments."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # User message with entrypoint
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "user-001",
                    "parentUuid": None,
                    "sessionId": "session-new",
                    "timestamp": "2025-06-01T14:00:00.000Z",
                    "cwd": "/dev/workspace/project",
                    "gitBranch": "main",
                    "version": "2.1.97",
                    "entrypoint": "cli",
                    "promptId": "prompt-001",
                    "message": {
                        "role": "user",
                        "content": "Fix the authentication bug",
                    },
                }
            )
            + "\n"
        )
        # Assistant message with usage data
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-001",
                    "parentUuid": "user-001",
                    "sessionId": "session-new",
                    "timestamp": "2025-06-01T14:00:05.000Z",
                    "entrypoint": "cli",
                    "requestId": "req_abc123",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-6",
                        "content": [
                            {"type": "text", "text": "I'll fix that bug for you."},
                        ],
                        "usage": {
                            "input_tokens": 1500,
                            "output_tokens": 200,
                            "cache_creation_input_tokens": 3000,
                            "cache_read_input_tokens": 500,
                            "service_tier": "standard",
                            "speed": "standard",
                            "cache_creation": {
                                "ephemeral_1h_input_tokens": 3000,
                                "ephemeral_5m_input_tokens": 0,
                            },
                        },
                    },
                }
            )
            + "\n"
        )
        # Second assistant with usage
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "asst-002",
                    "parentUuid": "asst-001",
                    "sessionId": "session-new",
                    "timestamp": "2025-06-01T14:00:15.000Z",
                    "entrypoint": "cli",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-6",
                        "content": [
                            {"type": "text", "text": "Done, the bug is fixed."},
                        ],
                        "usage": {
                            "input_tokens": 2000,
                            "output_tokens": 300,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 1000,
                            "service_tier": "standard",
                            "speed": "standard",
                            "cache_creation": {
                                "ephemeral_1h_input_tokens": 0,
                                "ephemeral_5m_input_tokens": 0,
                            },
                        },
                    },
                }
            )
            + "\n"
        )
        # System: turn_duration
        f.write(
            json.dumps(
                {
                    "type": "system",
                    "subtype": "turn_duration",
                    "durationMs": 45000,
                    "messageCount": 12,
                    "timestamp": "2025-06-01T14:00:50.000Z",
                    "uuid": "sys-001",
                    "isSidechain": False,
                    "sessionId": "session-new",
                    "cwd": "/dev/workspace/project",
                    "entrypoint": "cli",
                    "version": "2.1.97",
                    "gitBranch": "main",
                    "userType": "external",
                    "parentUuid": "asst-002",
                    "isMeta": True,
                    "slug": "test-slug",
                }
            )
            + "\n"
        )
        # System: stop_hook_summary
        f.write(
            json.dumps(
                {
                    "type": "system",
                    "subtype": "stop_hook_summary",
                    "stopReason": "end_turn",
                    "hookCount": 2,
                    "hasOutput": True,
                    "preventedContinuation": False,
                    "hookInfos": [
                        {"command": "hooks/stop.py", "durationMs": 47},
                        {"command": "hooks/log.py", "durationMs": 12},
                    ],
                    "hookErrors": [],
                    "timestamp": "2025-06-01T14:00:51.000Z",
                    "uuid": "sys-002",
                    "isSidechain": False,
                    "sessionId": "session-new",
                    "cwd": "/dev/workspace/project",
                    "entrypoint": "cli",
                    "version": "2.1.97",
                    "gitBranch": "main",
                    "userType": "external",
                    "parentUuid": "asst-002",
                    "level": "info",
                    "toolUseID": "stop-001",
                    "slug": "test-slug",
                }
            )
            + "\n"
        )
        # Attachment: diagnostics
        f.write(
            json.dumps(
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "diagnostics",
                        "files": [
                            {
                                "uri": "/dev/workspace/project/auth.py",
                                "diagnostics": [
                                    {
                                        "message": "Undefined variable 'token'",
                                        "severity": "Error",
                                        "range": {
                                            "start": {"line": 42, "character": 8},
                                            "end": {"line": 42, "character": 13},
                                        },
                                        "source": "Pyright",
                                        "code": "reportUndefinedVariable",
                                    }
                                ],
                            }
                        ],
                        "isNew": True,
                    },
                    "timestamp": "2025-06-01T14:00:10.000Z",
                    "uuid": "att-001",
                    "isSidechain": False,
                    "sessionId": "session-new",
                    "entrypoint": "cli",
                    "cwd": "/dev/workspace/project",
                    "version": "2.1.97",
                    "gitBranch": "main",
                    "parentUuid": "asst-001",
                    "userType": "external",
                }
            )
            + "\n"
        )
        # Attachment: hook_success (just needs to be counted)
        f.write(
            json.dumps(
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "hook_success",
                        "hookName": "PreToolUse:Bash",
                        "durationMs": 35,
                        "exitCode": 0,
                    },
                    "timestamp": "2025-06-01T14:00:06.000Z",
                    "uuid": "att-002",
                    "isSidechain": False,
                    "sessionId": "session-new",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "hook_success",
                        "hookName": "PostToolUse:Bash",
                        "durationMs": 20,
                        "exitCode": 0,
                    },
                    "timestamp": "2025-06-01T14:00:07.000Z",
                    "uuid": "att-003",
                    "isSidechain": False,
                    "sessionId": "session-new",
                }
            )
            + "\n"
        )
        # custom-title
        f.write(
            json.dumps(
                {
                    "type": "custom-title",
                    "customTitle": "fix-auth-bug",
                    "sessionId": "session-new",
                }
            )
            + "\n"
        )
        # permission-mode
        f.write(
            json.dumps(
                {
                    "type": "permission-mode",
                    "permissionMode": "normal",
                    "sessionId": "session-new",
                }
            )
            + "\n"
        )
        f.flush()
        yield Path(f.name)


@pytest.fixture
def agent_session_with_meta(output_dir):
    """Create an agent session JSONL with a .meta.json sidecar file."""
    # Create session directory structure: parent/subagents/agent-xxx.jsonl + .meta.json
    subagents_dir = output_dir / "session-parent" / "subagents"
    subagents_dir.mkdir(parents=True)

    # Write the .meta.json sidecar
    meta_path = subagents_dir / "agent-a1234567890abcdef.meta.json"
    meta_path.write_text(json.dumps({
        "agentType": "Explore",
        "description": "Explore the authentication module",
    }))

    # Write the agent JSONL
    jsonl_path = subagents_dir / "agent-a1234567890abcdef.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(
            json.dumps({
                "type": "user",
                "uuid": "user-001",
                "parentUuid": None,
                "sessionId": "session-parent",
                "agentId": "a1234567890abcdef",
                "isSidechain": True,
                "timestamp": "2025-06-01T14:00:00.000Z",
                "cwd": "/dev/workspace/project",
                "gitBranch": "main",
                "version": "2.1.97",
                "entrypoint": "cli",
                "message": {
                    "role": "user",
                    "content": "Explore the authentication module and report back",
                },
            })
            + "\n"
        )
        f.write(
            json.dumps({
                "type": "assistant",
                "uuid": "asst-001",
                "parentUuid": "user-001",
                "sessionId": "session-parent",
                "timestamp": "2025-06-01T14:00:10.000Z",
                "entrypoint": "cli",
                "message": {
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [
                        {"type": "text", "text": "I found the auth module."},
                    ],
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 200,
                    },
                },
            })
            + "\n"
        )

    return jsonl_path


@pytest.fixture
def plan_mode_session_file():
    """Session with two ExitPlanMode calls: first rejected, second approved.

    Flow:
      1. User asks for a plan.
      2. Assistant invokes ExitPlanMode (plan v1).
      3. tool_result carries user feedback rejecting the plan.
      4. User text message with more feedback.
      5. Assistant invokes ExitPlanMode (plan v2).
      6. tool_result carries Claude Code's approval signature.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # User asks for a plan
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "pm-user-001",
                    "parentUuid": None,
                    "sessionId": "plan-session-1",
                    "timestamp": "2025-03-10T10:00:00.000Z",
                    "cwd": "/home/user/project",
                    "gitBranch": "main",
                    "version": "2.1.0",
                    "message": {
                        "role": "user",
                        "content": "Plan how to refactor the auth module",
                    },
                }
            )
            + "\n"
        )
        # Assistant proposes plan v1
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "pm-asst-001",
                    "parentUuid": "pm-user-001",
                    "sessionId": "plan-session-1",
                    "timestamp": "2025-03-10T10:00:05.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {"type": "text", "text": "Here's my plan."},
                            {
                                "type": "tool_use",
                                "id": "plan-call-001",
                                "name": "ExitPlanMode",
                                "input": {
                                    "plan": "1. Read auth.py\n2. Rewrite login flow",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # tool_result for plan 1 -- rejection signal
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "pm-user-002",
                    "parentUuid": "pm-asst-001",
                    "sessionId": "plan-session-1",
                    "timestamp": "2025-03-10T10:00:10.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "plan-call-001",
                                "content": "The user has requested changes to the plan.",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        # User text feedback between plans
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "pm-user-003",
                    "parentUuid": "pm-user-002",
                    "sessionId": "plan-session-1",
                    "timestamp": "2025-03-10T10:00:11.000Z",
                    "message": {
                        "role": "user",
                        "content": "Also add tests for the new flow.",
                    },
                }
            )
            + "\n"
        )
        # Assistant proposes plan v2
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "pm-asst-002",
                    "parentUuid": "pm-user-003",
                    "sessionId": "plan-session-1",
                    "timestamp": "2025-03-10T10:00:20.000Z",
                    "message": {
                        "role": "assistant",
                        "model": "claude-opus-4-5-20251101",
                        "content": [
                            {"type": "text", "text": "Updated plan with tests."},
                            {
                                "type": "tool_use",
                                "id": "plan-call-002",
                                "name": "ExitPlanMode",
                                "input": {
                                    "plan": "1. Read auth.py\n2. Rewrite login flow\n3. Add integration tests",
                                },
                            },
                        ],
                    },
                }
            )
            + "\n"
        )
        # tool_result for plan 2 -- approval signature
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "uuid": "pm-user-004",
                    "parentUuid": "pm-asst-002",
                    "sessionId": "plan-session-1",
                    "timestamp": "2025-03-10T10:00:25.000Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "plan-call-002",
                                "content": "User has approved your plan. You can now start coding.",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
        f.flush()
        yield Path(f.name)


@pytest.fixture
def mock_projects_dir(sample_session_file):
    """Create a mock projects directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir)

        # Create a project folder
        project_dir = projects_dir / "-home-user-project"
        project_dir.mkdir(parents=True)

        # Copy sample session to project
        session_file = project_dir / "session-123.jsonl"
        session_file.write_text(sample_session_file.read_text())

        yield projects_dir
