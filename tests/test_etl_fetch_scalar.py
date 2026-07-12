"""Tests for fetch_scalar -- the typed one-row-one-column query helper.

Replaces the bare `.fetchone()[0]` pattern (which pyright flags because
fetchone() is Optional) with a helper that fails loud on zero rows.
"""

from __future__ import annotations

import duckdb
import pytest

from ccutils.etl.utils import fetch_scalar


@pytest.fixture
def conn():
    return duckdb.connect(":memory:")


class TestFetchScalar:
    def test_returns_scalar_value(self, conn):
        assert fetch_scalar(conn, "SELECT 42") == 42

    def test_passes_parameters(self, conn):
        assert fetch_scalar(conn, "SELECT md5(?)", ["abc"]) == (
            "900150983cd24fb0d6963f7d28e17f72"
        )

    def test_returns_first_column_only(self, conn):
        assert fetch_scalar(conn, "SELECT 'a', 'b'") == "a"

    def test_raises_on_zero_rows(self, conn):
        conn.execute("CREATE TABLE empty_t (x INTEGER)")
        with pytest.raises(RuntimeError, match="no rows"):
            fetch_scalar(conn, "SELECT x FROM empty_t LIMIT 1")

    def test_null_scalar_is_returned_not_raised(self, conn):
        """A row containing NULL is a valid result -- only zero ROWS raise."""
        assert fetch_scalar(conn, "SELECT NULL") is None
