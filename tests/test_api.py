"""Tests for the API client and pagination handling."""

import pytest
from unittest.mock import Mock, patch

from ccutils.api import fetch_sessions, get_api_headers


class TestFetchSessions:
    """Tests for fetch_sessions pagination."""

    def test_single_page_response(self):
        """When has_more is False, returns single page of sessions."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "session-1", "title": "First"},
                {"id": "session-2", "title": "Second"},
            ],
            "has_more": False,
        }
        mock_response.raise_for_status = Mock()

        with patch("ccutils.api.httpx.get", return_value=mock_response) as mock_get:
            result = fetch_sessions("token", "org-uuid")

        assert len(result["data"]) == 2
        assert result["data"][0]["id"] == "session-1"
        assert result["has_more"] is False
        mock_get.assert_called_once()

    def test_pagination_fetches_all_pages(self):
        """When has_more is True, fetches subsequent pages using last_id."""
        # First page response
        page1_response = Mock()
        page1_response.json.return_value = {
            "data": [
                {"id": "session-1", "title": "First"},
                {"id": "session-2", "title": "Second"},
            ],
            "has_more": True,
            "last_id": "session-2",
        }
        page1_response.raise_for_status = Mock()

        # Second page response
        page2_response = Mock()
        page2_response.json.return_value = {
            "data": [
                {"id": "session-3", "title": "Third"},
                {"id": "session-4", "title": "Fourth"},
            ],
            "has_more": True,
            "last_id": "session-4",
        }
        page2_response.raise_for_status = Mock()

        # Third page (final)
        page3_response = Mock()
        page3_response.json.return_value = {
            "data": [
                {"id": "session-5", "title": "Fifth"},
            ],
            "has_more": False,
        }
        page3_response.raise_for_status = Mock()

        with patch(
            "ccutils.api.httpx.get",
            side_effect=[page1_response, page2_response, page3_response],
        ) as mock_get:
            result = fetch_sessions("token", "org-uuid")

        # Should have all 5 sessions combined
        assert len(result["data"]) == 5
        assert [s["id"] for s in result["data"]] == [
            "session-1",
            "session-2",
            "session-3",
            "session-4",
            "session-5",
        ]
        assert result["has_more"] is False

        # Should have made 3 API calls
        assert mock_get.call_count == 3

        # Second call should include after_id parameter
        second_call_kwargs = mock_get.call_args_list[1][1]
        assert second_call_kwargs["params"]["after_id"] == "session-2"

        # Third call should include after_id parameter
        third_call_kwargs = mock_get.call_args_list[2][1]
        assert third_call_kwargs["params"]["after_id"] == "session-4"

    def test_debug_mode_returns_first_page_only(self):
        """In debug mode, returns raw first page response without pagination."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "session-1", "title": "First"},
            ],
            "has_more": True,
            "last_id": "session-1",
            "first_id": "session-1",
        }
        mock_response.raise_for_status = Mock()

        with patch("ccutils.api.httpx.get", return_value=mock_response) as mock_get:
            result = fetch_sessions("token", "org-uuid", debug=True)

        # Should return raw response with has_more=True (not paginated)
        assert result["has_more"] is True
        assert result["last_id"] == "session-1"
        assert len(result["data"]) == 1
        mock_get.assert_called_once()

    def test_empty_response(self):
        """Handles empty data array gracefully."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [],
            "has_more": False,
        }
        mock_response.raise_for_status = Mock()

        with patch("ccutils.api.httpx.get", return_value=mock_response):
            result = fetch_sessions("token", "org-uuid")

        assert result["data"] == []
        assert result["has_more"] is False

    def test_missing_has_more_treated_as_false(self):
        """When has_more is missing, treats it as False (no pagination)."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "session-1"}],
            # No has_more field
        }
        mock_response.raise_for_status = Mock()

        with patch("ccutils.api.httpx.get", return_value=mock_response) as mock_get:
            result = fetch_sessions("token", "org-uuid")

        assert len(result["data"]) == 1
        mock_get.assert_called_once()

    def test_limit_parameter_passed_to_api(self):
        """When limit is provided, it's included in API request params."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"id": "session-1"}],
            "has_more": False,
        }
        mock_response.raise_for_status = Mock()

        with patch("ccutils.api.httpx.get", return_value=mock_response) as mock_get:
            fetch_sessions("token", "org-uuid", limit=100)

        # Verify limit was passed in params
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["params"] == {"limit": 100}


class TestGetApiHeaders:
    """Tests for API header generation."""

    def test_headers_include_required_fields(self):
        """Headers include authorization, version, content-type, and org UUID."""
        headers = get_api_headers("my-token", "my-org-uuid")

        assert headers["Authorization"] == "Bearer my-token"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["Content-Type"] == "application/json"
        assert headers["x-organization-uuid"] == "my-org-uuid"
