"""Cross-feature checks of the star warehouse that no per-fact file holds.

What is left after the 2026-11-13 audit removed the duplicates: rows that
survive a reopen, an in-place update scoped to its session, staging cleared
on success and on exception, subagent depth over a deep tree, domain and
depth heuristics on mixed inputs, and the HTML CSP's `default-src 'none'`
(the only test that catches a widened default-src) and no-innerHTML scans.
Each asserts on stored data or rendered output.
"""

import hashlib
import json
import re
import tempfile
from pathlib import Path
import pytest
import duckdb

def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

from ccutils import create_star_schema
from ccutils.etl.orchestrator import run_v15_etl
from ccutils.etl.heuristics import (
    classify_complexity,
    classify_domain,
    classify_intent,
    classify_outcome,
)
from ccutils.export.html import generate_html


@pytest.fixture
def conn(tmp_path):
    """Fixture to create a fresh DuckDB connection with the star schema initialized."""
    db_path = tmp_path / "test_e2e_star.duckdb"
    return create_star_schema(db_path)


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture returning a temporary directory Path."""
    return tmp_path


def insert_mock_fact(conn, table, natural_key, key_value, **kwargs):
    """Insert a mock fact row with valid lineage metadata defaults."""
    lineage = {
        "created_by_version_key": "test",
        "last_updated_by_version_key": "test",
        "etl_run_id": "test_run",
        "record_source": "test",
        "hash_diff": "test_hash",
        "is_deleted": False,
        natural_key: key_value,
    }
    lineage.update(kwargs)
    
    columns = list(lineage.keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_names = ", ".join(columns)
    
    conn.execute(
        f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})",
        list(lineage.values())
    )


def create_mock_session_file(tmp_path, session_id, loglines):
    """Helper to write a mock session JSONL file."""
    path = tmp_path / f"{session_id}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for line in loglines:
            f.write(json.dumps(line) + "\n")
    return path


def make_basic_loglines(session_id, user_msg="fix the broken parser bug", tool_error=False):
    """Helper to construct basic loglines for a session."""
    lines = [
        {
            "type": "user",
            "uuid": "u1",
            "sessionId": session_id,
            "timestamp": "2026-04-19T10:00:00Z",
            "cwd": "/work",
            "gitBranch": "main",
            "version": "2.1.114",
            "message": {"role": "user", "content": user_msg},
        },
        {
            "type": "assistant",
            "uuid": "a1",
            "parentUuid": "u1",
            "sessionId": session_id,
            "timestamp": "2026-04-19T10:00:01Z",
            "requestId": "req_1",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [
                    {"type": "text", "text": "Ok, running a tool."},
                    {"type": "tool_use", "id": f"tu_{session_id}", "name": "Bash", "input": {"command": "cat app.js"}},
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u2",
            "parentUuid": "a1",
            "sessionId": session_id,
            "timestamp": "2026-04-19T10:00:02Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"tu_{session_id}", "content": "error details" if tool_error else "file content", "is_error": True if tool_error else False}],
            },
            "toolUseResult": {
                "stdout": "error details" if tool_error else "file content",
                "interrupted": False,
                "exitCode": 1 if tool_error else 0,
            },
        },
        {
            "type": "assistant",
            "uuid": "a2",
            "parentUuid": "u2",
            "sessionId": session_id,
            "timestamp": "2026-04-19T10:00:03Z",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "Done, resolved successfully."}],
            },
        },
    ]
    return lines


# =========================================================================
# TIER 1: Feature Coverage (Tests 1-20)
# =========================================================================

# --- Feature 1: Incremental Load & DDL Migration Fixes ---

def test_t1_ddl_create_table_if_not_exists_dim_session(conn, temp_dir):
    """Test 1: Verify dim_session DDL uses CREATE TABLE IF NOT EXISTS.
    Ensures recreating the schema does not drop existing data in dim_session.
    """
    # Insert mock row into dim_session
    conn.execute("INSERT INTO dim_session (session_key, session_id) VALUES ('key123', 'sess123')")
    
    # Re-run schema initialization
    db_path = temp_dir / "test_e2e_star.duckdb"
    create_star_schema(db_path)
    
    # Assert row is still present
    row = conn.execute("SELECT COUNT(*) FROM dim_session WHERE session_id = 'sess123'").fetchone()
    assert row[0] == 1, "dim_session was dropped and recreated, losing historical data"


def test_t1_ddl_create_table_if_not_exists_fact_messages(conn, temp_dir):
    """Test 2: Verify fact_messages DDL uses CREATE TABLE IF NOT EXISTS.
    Ensures recreating the schema does not drop existing data in fact_messages.
    """
    conn.execute(
        """
        INSERT INTO fact_messages (
            created_by_version_key, last_updated_by_version_key, etl_run_id, record_source, hash_diff,
            entry_id, message_id, session_id, message_type, is_deleted
        ) VALUES ('test', 'test', 'test', 'test', 'test', 'entry123', 'msg123', 'sess123', 'user', FALSE)
        """
    )
    
    db_path = temp_dir / "test_e2e_star.duckdb"
    create_star_schema(db_path)
    
    row = conn.execute("SELECT COUNT(*) FROM fact_messages WHERE message_id = 'msg123'").fetchone()
    assert row[0] == 1, "fact_messages was dropped and recreated, losing historical data"


def test_t1_ddl_create_table_if_not_exists_fact_token_usage(conn, temp_dir):
    """Test 4: Verify fact_token_usage DDL uses CREATE TABLE IF NOT EXISTS.
    Ensures recreating the schema does not drop existing data in fact_token_usage.
    """
    conn.execute(
        """
        INSERT INTO fact_token_usage (
            created_by_version_key, last_updated_by_version_key, etl_run_id, record_source, hash_diff,
            entry_id, session_id, is_deleted
        ) VALUES ('test', 'test', 'test', 'test', 'test', 'tok123', 'sess123', FALSE)
        """
    )
    
    db_path = temp_dir / "test_e2e_star.duckdb"
    create_star_schema(db_path)
    
    row = conn.execute("SELECT COUNT(*) FROM fact_token_usage WHERE entry_id = 'tok123'").fetchone()
    assert row[0] == 1, "fact_token_usage was dropped and recreated, losing historical data"


def test_t1_incremental_load_retains_different_session_data(conn, temp_dir):
    """Test 6: Verify sequential ETL runs on different session files do not drop previous session data.
    Ensures incremental updates append session history.
    """
    session_a = create_mock_session_file(temp_dir, "sess_a", make_basic_loglines("sess_a"))
    session_b = create_mock_session_file(temp_dir, "sess_b", make_basic_loglines("sess_b"))
    
    run_v15_etl(conn, session_a, project_name="test-project", parquet_lake_root=temp_dir / "lake")
    assert conn.execute("SELECT COUNT(*) FROM dim_session WHERE session_id = 'sess_a'").fetchone()[0] == 1
    
    run_v15_etl(conn, session_b, project_name="test-project", parquet_lake_root=temp_dir / "lake")
    
    # Both sessions must exist in dim_session
    assert conn.execute("SELECT COUNT(*) FROM dim_session WHERE session_id = 'sess_a'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM dim_session WHERE session_id = 'sess_b'").fetchone()[0] == 1


def test_t1_upsert_zero_rows_soft_deletes_old_records(conn, temp_dir):
    """Test 7: Verify soft-delete upsert logic handles cases where a populator yields 0 rows.
    E.g. 0 tool errors on rerun. If errors were present in the first run but none in the second,
    the old errors must be soft-deleted (is_deleted = TRUE).
    """
    # 1. Run ETL with tool error -> populates fact_errors
    session_err = create_mock_session_file(temp_dir, "sess_err", make_basic_loglines("sess_err", tool_error=True))
    run_v15_etl(conn, session_err, project_name="test-project", parquet_lake_root=temp_dir / "lake")
    
    err_rows = conn.execute("SELECT COUNT(*) FROM fact_tool_calls WHERE is_error = TRUE AND session_id = 'sess_err' AND is_deleted = FALSE").fetchone()[0]
    assert err_rows > 0, "Expected active error row in fact_errors on first run"
    
    # 2. Run ETL on the same session but with NO errors -> populator yields 0 errors
    session_no_err = create_mock_session_file(temp_dir, "sess_err", make_basic_loglines("sess_err", tool_error=False))
    run_v15_etl(conn, session_no_err, project_name="test-project", parquet_lake_root=temp_dir / "lake")
    
    # Assert that the error record was soft-deleted (is_deleted = TRUE)
    active_errs = conn.execute("SELECT COUNT(*) FROM fact_tool_calls WHERE is_error = TRUE AND session_id = 'sess_err' AND is_deleted = FALSE").fetchone()[0]
    assert active_errs == 0, "Old error record was not soft-deleted on rerun with 0 errors"
    
    # The call itself is one row at tool_use_id grain: it is UPDATED in
    # place (is_error flips to FALSE), not soft-deleted -- the call still
    # happened. What must vanish is the failure, and it has.
    live_calls = conn.execute("SELECT COUNT(*) FROM fact_tool_calls WHERE session_id = 'sess_err' AND is_deleted = FALSE AND is_error IS NOT TRUE").fetchone()[0]
    assert live_calls > 0, "Expected the call to survive with is_error cleared"


# --- Feature 2: Staging & Summary Query Optimization ---

def test_t1_staging_unconditionally_cleared_log_entries(conn, temp_dir):
    """Test 8: Verify etl.log_entries is unconditionally cleared after run_v15_etl completes.
    Even with include_thinking=True (default).
    """
    session = create_mock_session_file(temp_dir, "sess_stg_1", make_basic_loglines("sess_stg_1"))
    run_v15_etl(conn, session, project_name="test-project", parquet_lake_root=temp_dir / "lake", include_thinking=True)
    
    # Verify staging is empty
    count = conn.execute("SELECT COUNT(*) FROM etl.log_entries").fetchone()[0]
    assert count == 0, "etl.log_entries was not cleared at the end of the ETL run"


def test_t1_heuristics_uses_dim_session_depth_level(conn, temp_dir):
    """Test 13: Verify populate_dim_session_heuristics passes the actual depth_level from dim_session.
    If depth_level > 0 (e.g. 1), and there are 9 user messages (msg_count > 8 gives score 1),
    the complexity should be classified as 'moderate' (score = 1 + 2 = 3).
    If depth_level is ignored (passed as 0), complexity would be 'simple' (score = 1).
    """
    # Create subagent-like session path so it gets picked up
    subagent_dir = temp_dir / "projects" / "p1" / "sess_parent" / "subagents"
    subagent_dir.mkdir(parents=True, exist_ok=True)
    
    # Write a .meta.json sidecar to mock agent details
    meta_path = subagent_dir / "agent-sub123.meta.json"
    meta_path.write_text(json.dumps({"agentType": "worker", "description": "subagent work"}), encoding="utf-8")
    
    # Construct 9 user messages to give msg_count = 9
    loglines = []
    for i in range(9):
        loglines.append({
            "type": "user",
            "uuid": f"u_{i}",
            "sessionId": "agent-sub123",
            "timestamp": f"2026-04-19T10:00:0{i}Z",
            "cwd": "/work",
            "gitBranch": "main",
            "version": "2.1.114",
            "message": {"role": "user", "content": "run task"},
        })
    
    jsonl_path = subagent_dir / "agent-sub123.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in loglines:
            f.write(json.dumps(line) + "\n")
            
    # Pre-insert parent session with depth_level = 0
    conn.execute("INSERT INTO dim_session (session_key, session_id, depth_level) VALUES (md5('sess_parent'), 'sess_parent', 0)")
    
    # Run E2E ETL on the subagent
    run_v15_etl(conn, jsonl_path, project_name="p1", parquet_lake_root=temp_dir / "lake")
    
    # Query depth_level and complexity for subagent
    row = conn.execute("SELECT depth_level, complexity FROM dim_session WHERE session_id = 'agent-sub123'").fetchone()
    assert row is not None
    assert row[0] == 1, "Expected depth_level to propagate to 1"
    assert row[1] == "moderate", f"Expected complexity to be 'moderate' (got {row[1]}) since depth_level was 1"


def test_t1_heuristics_js_mapped_to_web(conn, temp_dir):
    """Test 14: Verify .js is mapped to 'web' domain in the domain classifier."""
    # Ensure domain map contains JS
    from ccutils.etl.heuristics import _DOMAIN_MAP
    assert ".js" in _DOMAIN_MAP.get("web", set()) or ".js" in _DOMAIN_MAP.get("backend", set()), "JavaScript extension (.js) not found in _DOMAIN_MAP"
    
    # Classify domain functionally
    domain = classify_domain([".js"])
    assert domain in ("web", "backend"), f"Expected domain classification of ['.js'] to resolve to 'web' or 'backend' (got {domain})"


def test_t1_heuristics_ts_mapped_to_web_or_backend(conn, temp_dir):
    """Test 15: Verify .ts is mapped to 'web' or 'backend' in _DOMAIN_MAP."""
    from ccutils.etl.heuristics import _DOMAIN_MAP
    assert ".ts" in _DOMAIN_MAP.get("web", set()) or ".ts" in _DOMAIN_MAP.get("backend", set()), "TypeScript extension (.ts) not found in _DOMAIN_MAP"
    
    domain = classify_domain([".ts"])
    assert domain in ("web", "backend"), f"Expected domain classification of ['.ts'] to resolve to 'web' or 'backend' (got {domain})"


# --- Feature 4: HTML Export and Security Improvements ---


def test_t1_html_inline_blocks_only_in_csp_bearing_templates(temp_dir):
    """Test 19: A template may inject inline <style>/<script> only if it also
    emits its own CSP meta tag (which _build_csp fills with the blocks'
    sha256 hashes). An injection without a policy to pin it is the regression
    this test exists to catch.
    """
    templates_dir = Path(__file__).parent.parent / "src" / "ccutils" / "templates"
    for tpl in sorted(templates_dir.glob("*.html")):
        content = tpl.read_text(encoding="utf-8")
        has_injection = re.search(r"<(style|script)>\{\{", content) is not None
        if has_injection:
            assert 'http-equiv="Content-Security-Policy"' in content and "{{ csp" in content, (
                f"{tpl.name} injects inline blocks but renders no CSP to pin them"
            )


def test_t1_html_no_innerhtml_in_shipped_js(temp_dir):
    """Test 20: No shipped JS uses dynamic innerHTML assignment.

    Clearing like .innerHTML = '' is fine; assignment from a variable or
    concatenation is an XSS sink. Scans every JS source the exporter inlines
    (templates/*.js and static/*.js) so a renamed or added file stays covered.
    """
    src_root = Path(__file__).parent.parent / "src" / "ccutils"
    js_files = sorted((src_root / "templates").glob("*.js")) + sorted(
        (src_root / "static").glob("*.js"))
    assert js_files, "no JS sources found -- glob roots are stale"
    for js in js_files:
        content = js.read_text(encoding="utf-8")
        matches = re.findall(r"\w+\.innerHTML\s*=\s*[^';]+", content)
        for match in matches:
            assert "''" in match or '""' in match, (
                f"Unsafe innerHTML assignment found in {js.name}: {match}")


# =========================================================================
# TIER 2: Boundary & Corner Cases (Tests 21-40)
# =========================================================================

# --- Feature 1: Incremental Load & DDL Migration Fixes ---


def test_t2_incremental_load_duplicate_sessions_updated_in_place(conn, temp_dir):
    """Test 22: Verify running ETL on the same session file twice updates dim_session rows in-place.
    Does not produce duplicate session keys.
    """
    session = create_mock_session_file(temp_dir, "sess_dup", make_basic_loglines("sess_dup"))
    
    run_v15_etl(conn, session, project_name="test-project", parquet_lake_root=temp_dir / "lake")
    first_count = conn.execute("SELECT COUNT(*) FROM dim_session WHERE session_id = 'sess_dup'").fetchone()[0]
    assert first_count == 1
    
    run_v15_etl(conn, session, project_name="test-project", parquet_lake_root=temp_dir / "lake")
    second_count = conn.execute("SELECT COUNT(*) FROM dim_session WHERE session_id = 'sess_dup'").fetchone()[0]
    
    assert second_count == 1, "Duplicate session keys created on re-ETL of same session"


def test_t2_upsert_soft_delete_only_affects_matching_session(conn, temp_dir):
    """Test 24: Verify soft-delete only deletes records for the specific session_id being re-run,
    leaving other sessions' records untouched.
    """
    sess_a = create_mock_session_file(temp_dir, "sess_a", make_basic_loglines("sess_a", tool_error=True))
    run_v15_etl(conn, sess_a, project_name="p", parquet_lake_root=temp_dir / "lake")
    
    sess_b = create_mock_session_file(temp_dir, "sess_b", make_basic_loglines("sess_b", tool_error=True))
    run_v15_etl(conn, sess_b, project_name="p", parquet_lake_root=temp_dir / "lake")
    
    assert conn.execute("SELECT COUNT(*) FROM fact_tool_calls WHERE is_error = TRUE AND session_id = 'sess_a' AND is_deleted = FALSE").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM fact_tool_calls WHERE is_error = TRUE AND session_id = 'sess_b' AND is_deleted = FALSE").fetchone()[0] > 0
    
    sess_a_fixed = create_mock_session_file(temp_dir, "sess_a", make_basic_loglines("sess_a", tool_error=False))
    run_v15_etl(conn, sess_a_fixed, project_name="p", parquet_lake_root=temp_dir / "lake")
    
    assert conn.execute("SELECT COUNT(*) FROM fact_tool_calls WHERE is_error = TRUE AND session_id = 'sess_a' AND is_deleted = FALSE").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM fact_tool_calls WHERE is_error = TRUE AND session_id = 'sess_b' AND is_deleted = FALSE").fetchone()[0] > 0


# --- Feature 2: Staging & Summary Query Optimization ---

def test_t2_staging_cleared_on_exception(conn, temp_dir):
    """Test 26: Verify staging tables are cleared even if an exception occurs during the ETL run."""
    conn.execute(
        """
        INSERT INTO etl.log_entries (
            etl_run_id, parsed_at, parser_version, record_source, entry_id, source_path, sequence_num, type, session_id
        ) VALUES ('run123', current_timestamp, '1.0', 'test', 'entry123', '/path', 1, 'user', 'sess123')
        """
    )
    
    invalid_file = temp_dir / "invalid.jsonl"
    invalid_file.write_text("invalid json lines", encoding="utf-8")
    
    with pytest.raises(Exception):
        run_v15_etl(conn, invalid_file, project_name="p", parquet_lake_root=temp_dir / "lake")
        
    count = conn.execute("SELECT COUNT(*) FROM etl.log_entries").fetchone()[0]
    assert count == 0, "Staging was not cleared on ETL failure"


def test_t2_heuristics_missing_depth_defaults_to_zero(conn, temp_dir):
    """Test 31: Verify dim_session_heuristics handles missing/NULL depth_level by defaulting to 0."""
    conn.execute("INSERT INTO dim_session (session_key, session_id, depth_level) VALUES (md5('sess_null'), 'sess_null', NULL)")
    res = classify_complexity(tool_count=0, msg_count=1, agent_depth=None, error_count=0)
    assert res == "trivial", f"Expected complexity to default to 'trivial' on NULL depth (got {res})"


def test_t2_heuristics_extreme_depth_levels(conn):
    """Test 32: Verify classify_complexity handles extreme depth levels correctly (e.g. 100)."""
    res = classify_complexity(tool_count=0, msg_count=1, agent_depth=100, error_count=0)
    assert res == "simple", f"Expected complexity score with deep depth level to be 'simple' (got {res})"


def test_t2_heuristics_mixed_extensions_ranking(conn):
    """Test 33: Verify domain classifier returns 'mixed' when multiple domains have equal scores."""
    domain = classify_domain([".js", ".py"])
    assert domain == "mixed", f"Expected domain to resolve to 'mixed' for ['.js', '.py'] (got {domain})"


def test_t2_subagent_recursive_cte_handles_deep_tree(conn):
    """Test 35: Verify subagent depth propagation recursive CTE query logic handles arbitrary deep levels.

    Propagation is scoped to the staged sessions' tree: staging only the
    DEEPEST session must walk up to root A and recompute the whole chain.
    """
    conn.execute("INSERT INTO dim_session (session_key, session_id, is_agent, parent_session_key) VALUES (md5('A'), 'A', FALSE, NULL)")
    conn.execute("INSERT INTO dim_session (session_key, session_id, is_agent, parent_session_key) VALUES (md5('B'), 'B', TRUE, md5('A'))")
    conn.execute("INSERT INTO dim_session (session_key, session_id, is_agent, parent_session_key) VALUES (md5('C'), 'C', TRUE, md5('B'))")
    conn.execute("INSERT INTO dim_session (session_key, session_id, is_agent, parent_session_key) VALUES (md5('D'), 'D', TRUE, md5('C'))")
    conn.execute(
        """
        INSERT INTO etl.log_entries (
            etl_run_id, parsed_at, parser_version, record_source, entry_id, source_path, sequence_num, type, session_id
        ) VALUES ('run-t35', current_timestamp, '1.0', 'test', 'entry-t35', '/path', 1, 'user', 'D')
        """
    )

    from ccutils.etl.subagent_enrichment import _propagate_depth_level
    _propagate_depth_level(conn)

    depths = {r[0]: r[1] for r in conn.execute("SELECT session_id, depth_level FROM dim_session").fetchall()}
    assert depths["A"] == 0
    assert depths["B"] == 1
    assert depths["C"] == 2
    assert depths["D"] == 3


# --- Feature 4: HTML Export and Security Improvements ---


def test_t2_html_csp_contains_strict_default_src(temp_dir):
    """Test 37: The built CSP locks default-src to 'none'."""
    from ccutils.export.html import _build_csp

    csp = _build_csp("body { color: red }", "console.log('x')")
    assert "default-src 'none'" in csp, "CSP default-src is not set to 'none'"


# =========================================================================
# TIER 4: Real-World Application Scenarios (Tests 45-50)
# =========================================================================


def test_t4_scenario_4_deep_subagent_depth_verification(conn, temp_dir):
    """Test 48: Scenario 4 - Deep subagent session depth verification.
    Sets up a deep multi-level subagent hierarchy: A -> B -> C -> D.
    Verifies recursive CTE propagates correct depth levels.
    """
    conn.execute("INSERT INTO dim_session (session_key, session_id, is_agent, parent_session_key) VALUES (md5('A'), 'A', FALSE, NULL)")
    conn.execute("INSERT INTO dim_session (session_key, session_id, is_agent, parent_session_key) VALUES (md5('B'), 'B', TRUE, md5('A'))")
    conn.execute("INSERT INTO dim_session (session_key, session_id, is_agent, parent_session_key) VALUES (md5('C'), 'C', TRUE, md5('B'))")
    conn.execute("INSERT INTO dim_session (session_key, session_id, is_agent, parent_session_key) VALUES (md5('D'), 'D', TRUE, md5('C'))")
    # Scoped propagation: stage a mid-chain session; the up-walk finds
    # root A, the down-walk recomputes B/C/D.
    conn.execute(
        """
        INSERT INTO etl.log_entries (
            etl_run_id, parsed_at, parser_version, record_source, entry_id, source_path, sequence_num, type, session_id
        ) VALUES ('run-t48', current_timestamp, '1.0', 'test', 'entry-t48', '/path', 1, 'user', 'B')
        """
    )

    from ccutils.etl.subagent_enrichment import _propagate_depth_level
    _propagate_depth_level(conn)

    depths = {r[0]: r[1] for r in conn.execute("SELECT session_id, depth_level FROM dim_session").fetchall()}
    assert depths["A"] == 0
    assert depths["B"] == 1
    assert depths["C"] == 2
    assert depths["D"] == 3


