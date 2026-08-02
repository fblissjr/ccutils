"""Tests for the semantic_context_growth view.

`fact_token_usage.total_uncached_equivalent_tokens` (input + cache_creation
+ cache_read) is the EXACT prompt size the API saw for one request. Diffing
it across consecutive requests in a session prices everything that entered
the context between two assistant turns -- tool results above all, which
carry no `usage` of their own and are ~90% of transcript text.

Each test names the claim it defends; deleting it should lose that claim.
"""

import pytest

from ccutils.schemas.star.schema import create_star_schema


def _insert_usage(conn, *, entry_id, session_id, timestamp, context, output):
    """Insert one fact_token_usage row with a known prompt size.

    `context` is the total prompt size for the request; it is split across
    input/cache_read only so that total_uncached_equivalent_tokens -- the
    column the view diffs -- lands on exactly `context`.
    """
    conn.execute(
        """
        INSERT INTO fact_token_usage (
            created_by_version_key, last_updated_by_version_key,
            etl_run_id, record_source, hash_diff,
            entry_id, session_id, session_key, timestamp,
            input_tokens, output_tokens,
            cache_creation_5m_tokens, cache_creation_1h_tokens,
            cache_creation_total_tokens, cache_read_tokens,
            total_uncached_equivalent_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, ?)
        """,
        [
            "v-test", "v-test", "run-test", "test", f"hash-{entry_id}",
            entry_id, session_id, f"sk-{session_id}", timestamp,
            context, output, context,
        ],
    )


@pytest.fixture
def conn(output_dir):
    connection = create_star_schema(output_dir / "test.duckdb")
    yield connection
    connection.close()


def _rows(conn, session_id="s1"):
    return conn.execute(
        """
        SELECT request_seq, context_tokens, context_delta_tokens,
               inbound_tokens, prev_output_tokens, is_context_reset
        FROM semantic_context_growth
        WHERE session_id = ?
        ORDER BY request_seq
        """,
        [session_id],
    ).fetchall()


class TestSemanticContextGrowthView:
    def test_creates_view(self, conn):
        """Claim: the view ships in the DDL, not just in a query recipe."""
        result = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='view' AND name='semantic_context_growth'"
        ).fetchone()
        assert result is not None

    def test_delta_is_difference_from_previous_request(self, conn):
        """Claim: context_delta_tokens diffs consecutive prompt sizes.

        Without this the view reports absolute context size, which says
        nothing about what any one turn added.
        """
        _insert_usage(conn, entry_id="e1", session_id="s1",
                      timestamp="2026-01-01 10:00:00", context=1000, output=50)
        _insert_usage(conn, entry_id="e2", session_id="s1",
                      timestamp="2026-01-01 10:01:00", context=9000, output=20)

        rows = _rows(conn)
        assert [r[0] for r in rows] == [1, 2]
        assert rows[1][1] == 9000
        assert rows[1][2] == 8000

    def test_first_request_has_null_delta(self, conn):
        """Claim: the first request has no predecessor, so no delta.

        Reporting 0 or the absolute size here would silently book the
        session's baseline overhead (system prompt, tools, CLAUDE.md) as
        growth attributable to a turn.
        """
        _insert_usage(conn, entry_id="e1", session_id="s1",
                      timestamp="2026-01-01 10:00:00", context=1000, output=50)

        row = _rows(conn)[0]
        assert row[1] == 1000          # context_tokens
        assert row[2] is None          # context_delta_tokens
        assert row[3] is None          # inbound_tokens
        assert row[5] is False         # is_context_reset

    def test_sessions_do_not_diff_against_each_other(self, conn):
        """Claim: the window partitions by session.

        Unpartitioned, the first request of every session would diff
        against an unrelated session's last one.
        """
        _insert_usage(conn, entry_id="e1", session_id="s1",
                      timestamp="2026-01-01 10:00:00", context=50000, output=50)
        _insert_usage(conn, entry_id="e2", session_id="s2",
                      timestamp="2026-01-01 10:01:00", context=1000, output=50)

        s2 = _rows(conn, "s2")
        assert len(s2) == 1
        assert s2[0][0] == 1
        assert s2[0][2] is None

    def test_inbound_excludes_previous_assistant_output(self, conn):
        """Claim: inbound_tokens nets out the echoed-back assistant turn.

        The delta contains the previous response's own output tokens,
        which re-enter the context as history. Only the remainder came
        from the user side (prompt text and tool results). Without this
        subtraction every tool result is over-priced by the assistant
        turn that requested it.
        """
        _insert_usage(conn, entry_id="e1", session_id="s1",
                      timestamp="2026-01-01 10:00:00", context=1000, output=200)
        _insert_usage(conn, entry_id="e2", session_id="s1",
                      timestamp="2026-01-01 10:01:00", context=6200, output=10)

        row = _rows(conn)[1]
        assert row[2] == 5200          # raw context delta
        assert row[4] == 200           # previous response's output
        assert row[3] == 5000          # tokens that came from the user side

    def test_context_reset_is_flagged_not_reported_as_negative_inbound(self, conn):
        """Claim: compaction/clear is flagged, and suppresses inbound.

        After a compact the prompt shrinks. Treating that as a turn's
        contribution yields a large negative 'inbound' that corrupts any
        SUM over the column.
        """
        _insert_usage(conn, entry_id="e1", session_id="s1",
                      timestamp="2026-01-01 10:00:00", context=90000, output=100)
        _insert_usage(conn, entry_id="e2", session_id="s1",
                      timestamp="2026-01-01 10:05:00", context=12000, output=100)

        row = _rows(conn)[1]
        assert row[5] is True
        assert row[3] is None
        assert row[2] == -78000        # the raw delta stays visible

    def test_inbound_may_be_negative_and_is_not_clipped(self, conn):
        """Claim: a response whose output was not retained shows negative inbound.

        Thinking tokens are billed as output but do not re-enter the
        context on later turns, so context can grow by less than the
        previous response's output_tokens. Measured on a real corpus: 9 of
        3,443 rows, worst case 8,001 output tokens against 3,991 of
        growth. Clipping to zero would hide a real and quantifiable
        effect; the honest residue is left visible.
        """
        _insert_usage(conn, entry_id="e1", session_id="s1",
                      timestamp="2026-01-01 10:00:00", context=1000, output=8001)
        _insert_usage(conn, entry_id="e2", session_id="s1",
                      timestamp="2026-01-01 10:01:00", context=4991, output=10)

        row = _rows(conn)[1]
        assert row[5] is False         # context grew, so not a reset
        assert row[2] == 3991
        assert row[3] == -4010

    def test_soft_deleted_rows_are_excluded(self, conn):
        """Claim: the view honours the soft-delete flag.

        A soft-deleted request left in the window would both appear as a
        row and skew its neighbour's delta.
        """
        _insert_usage(conn, entry_id="e1", session_id="s1",
                      timestamp="2026-01-01 10:00:00", context=1000, output=50)
        _insert_usage(conn, entry_id="e2", session_id="s1",
                      timestamp="2026-01-01 10:01:00", context=5000, output=50)
        _insert_usage(conn, entry_id="e3", session_id="s1",
                      timestamp="2026-01-01 10:02:00", context=9000, output=50)
        conn.execute(
            "UPDATE fact_token_usage SET is_deleted = TRUE WHERE entry_id = 'e2'"
        )

        rows = _rows(conn)
        assert len(rows) == 2
        assert rows[1][1] == 9000
        assert rows[1][2] == 8000      # diffed against e1, not the deleted e2
