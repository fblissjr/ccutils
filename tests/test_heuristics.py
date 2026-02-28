"""Tests for heuristic classification module."""

import pytest

from ccutils.schemas.star.heuristics import (
    classify_intent,
    classify_complexity,
    classify_outcome,
    classify_domain,
    classify_error_type,
)


class TestClassifyIntent:
    """Tests for intent classification from first user message."""

    def test_bug_fix_keywords(self):
        assert classify_intent("fix the broken login page") == "bug_fix"
        assert classify_intent("there's a bug in the parser") == "bug_fix"
        assert classify_intent("this error keeps happening") == "bug_fix"
        assert classify_intent("the app crashes on startup") == "bug_fix"

    def test_feature_keywords(self):
        assert classify_intent("add a new search feature") == "feature"
        assert classify_intent("implement user authentication") == "feature"
        assert classify_intent("create a new endpoint for users") == "feature"

    def test_refactor_keywords(self):
        assert classify_intent("refactor the auth module") == "refactor"
        assert classify_intent("clean up the utils file") == "refactor"
        assert classify_intent("reorganize the project structure") == "refactor"

    def test_debug_keywords(self):
        assert classify_intent("debug the memory leak") == "debug"
        assert classify_intent("investigate why tests fail") == "debug"
        assert classify_intent("why is this returning None?") == "debug"
        assert classify_intent("trace the execution path") == "debug"

    def test_test_keywords(self):
        assert classify_intent("write tests for the parser") == "test"
        assert classify_intent("add spec coverage for auth") == "test"
        assert classify_intent("increase test coverage") == "test"

    def test_docs_keywords(self):
        assert classify_intent("update the readme") == "docs"
        assert classify_intent("add documentation for the API") == "docs"
        assert classify_intent("write a comment explaining this") == "docs"

    def test_review_keywords(self):
        assert classify_intent("review this pull request") == "review"
        assert classify_intent("check the code for issues") == "review"
        assert classify_intent("audit the security config") == "review"

    def test_fallback_to_explore(self):
        assert classify_intent("hello") == "explore"
        assert classify_intent("what does this project do?") == "explore"
        assert classify_intent("") == "explore"

    def test_none_input(self):
        assert classify_intent(None) == "explore"

    def test_case_insensitive(self):
        assert classify_intent("FIX the BUG") == "bug_fix"
        assert classify_intent("ADD a feature") == "feature"

    def test_compound_intent_most_keywords_wins(self):
        # "Create tests for the refactor" - 2 test keywords (tests, coverage implied) vs 1 refactor
        assert (
            classify_intent(
                "write tests and add test coverage for the refactored module"
            )
            == "test"
        )

    def test_compound_bug_vs_feature(self):
        # "Implement new error handling" - "implement", "new" = 2 feature keywords vs "error" = 1 bug_fix
        assert classify_intent("implement new error handling") == "feature"

    def test_compound_debug_vs_bugfix(self):
        # "Fix the debugging logic" - "fix" = 1 bug_fix, "debug" = 1 debug; tie goes to priority (bug_fix)
        assert classify_intent("fix the debugging logic") == "bug_fix"

    def test_compound_multiple_feature_keywords(self):
        # Multiple feature keywords should beat single other
        assert classify_intent("add a new feature to create user profiles") == "feature"

    def test_single_keyword_unchanged(self):
        # Existing single-keyword behavior must be preserved
        assert classify_intent("fix the login") == "bug_fix"
        assert classify_intent("refactor the module") == "refactor"
        assert classify_intent("debug the issue") == "debug"
        assert classify_intent("write a test") == "test"
        assert classify_intent("update the docs") == "docs"
        assert classify_intent("review the PR") == "review"
        assert classify_intent("add a button") == "feature"


class TestClassifyComplexity:
    """Tests for complexity classification from session metrics."""

    def test_trivial(self):
        assert (
            classify_complexity(tool_count=0, msg_count=2, agent_depth=0, error_count=0)
            == "trivial"
        )

    def test_simple(self):
        assert (
            classify_complexity(tool_count=3, msg_count=5, agent_depth=0, error_count=0)
            == "simple"
        )

    def test_moderate(self):
        assert (
            classify_complexity(
                tool_count=10, msg_count=15, agent_depth=0, error_count=5
            )
            == "moderate"
        )

    def test_complex(self):
        assert (
            classify_complexity(
                tool_count=25, msg_count=35, agent_depth=1, error_count=5
            )
            == "complex"
        )

    def test_agent_depth_adds_complexity(self):
        # Agent depth > 0 adds +2
        assert (
            classify_complexity(tool_count=3, msg_count=5, agent_depth=1, error_count=0)
            == "moderate"
        )

    def test_high_error_count(self):
        # error_count > 3 adds +1
        assert (
            classify_complexity(tool_count=3, msg_count=5, agent_depth=0, error_count=5)
            == "simple"
        )


class TestClassifyOutcome:
    """Tests for outcome classification from last messages."""

    def test_success_from_done(self):
        assert classify_outcome("I've completed the task.", error_rate=0.0) == "success"
        assert (
            classify_outcome("Done! The file has been created.", error_rate=0.1)
            == "success"
        )
        assert classify_outcome("I fixed the bug.", error_rate=0.0) == "success"

    def test_failure_from_error(self):
        assert (
            classify_outcome("I couldn't complete this.", error_rate=0.0) == "failure"
        )
        assert classify_outcome("The operation failed.", error_rate=0.0) == "failure"

    def test_failure_from_high_error_rate(self):
        assert classify_outcome("Here are the results.", error_rate=0.6) == "failure"

    def test_unknown_when_ambiguous(self):
        assert (
            classify_outcome("Here are some suggestions.", error_rate=0.1) == "unknown"
        )

    def test_none_input(self):
        assert classify_outcome(None, error_rate=0.0) == "unknown"

    def test_empty_input(self):
        assert classify_outcome("", error_rate=0.0) == "unknown"


class TestClassifyDomain:
    """Tests for domain classification from file extensions."""

    def test_web_domain(self):
        assert classify_domain([".tsx", ".css", ".html"]) == "web"

    def test_backend_domain(self):
        assert classify_domain([".py", ".py", ".py"]) == "backend"

    def test_data_domain(self):
        assert classify_domain([".sql", ".csv", ".parquet"]) == "data"

    def test_devops_domain(self):
        assert classify_domain([".yaml", ".tf", ".sh"]) == "devops"

    def test_docs_domain(self):
        assert classify_domain([".md", ".rst", ".txt"]) == "docs"

    def test_mixed_domain(self):
        # Equal scores -> mixed
        assert classify_domain([".py", ".tsx"]) == "mixed"

    def test_unknown_when_empty(self):
        assert classify_domain([]) == "unknown"

    def test_unknown_extensions(self):
        assert classify_domain([".xyz", ".abc"]) == "unknown"

    def test_highest_score_wins(self):
        assert classify_domain([".py", ".py", ".tsx"]) == "backend"


class TestClassifyErrorType:
    """Tests for error type classification from error message text."""

    def test_permission_denied(self):
        assert (
            classify_error_type("permission denied: /etc/passwd") == "permission_denied"
        )
        assert (
            classify_error_type("EACCES: operation not permitted")
            == "permission_denied"
        )

    def test_file_not_found(self):
        assert (
            classify_error_type("ENOENT: no such file or directory") == "file_not_found"
        )
        assert (
            classify_error_type("Error: file not found at /foo/bar") == "file_not_found"
        )

    def test_syntax_error(self):
        assert classify_error_type("SyntaxError: unexpected token") == "syntax_error"
        assert classify_error_type("syntax error near 'FROM'") == "syntax_error"

    def test_timeout(self):
        assert classify_error_type("ETIMEDOUT: connection timed out") == "timeout"
        assert classify_error_type("Error: timeout waiting for response") == "timeout"

    def test_import_error(self):
        assert (
            classify_error_type("ModuleNotFoundError: No module named 'foo'")
            == "import_error"
        )
        assert (
            classify_error_type("ImportError: cannot import name 'bar'")
            == "import_error"
        )

    def test_fallback_to_tool_error(self):
        assert classify_error_type("something went wrong") == "tool_error"
        assert classify_error_type("unexpected result") == "tool_error"

    def test_none_input(self):
        assert classify_error_type(None) == "tool_error"

    def test_empty_input(self):
        assert classify_error_type("") == "tool_error"
