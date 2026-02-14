"""DuckDB archive generation.

This module provides functions for creating DuckDB database archives
from Claude Code session files. Supports both simple (4-table) and
star (25+ table dimensional) schemas.
"""

import os
import tempfile
import time
from pathlib import Path

import duckdb

from ..parsers import find_all_sessions
from ..schemas import (
    create_duckdb_schema,
    export_session_to_duckdb,
    create_star_schema,
    run_star_schema_etl,
    export_star_schema_to_json,
)
from ..schemas.star.utils import generate_dimension_key


def generate_duckdb_archive(
    source_folder,
    output_dir,
    schema_type="simple",
    include_agents=False,
    include_thinking=False,
    truncate_output=2000,
    progress_callback=None,
    max_workers=1,
    batch_size=10,
):
    """Generate DuckDB archive for all sessions.

    Supports both simple (4-table) and star (25+ dimensional tables) schemas.
    Uses a stage-and-load pattern for efficient batch processing:
    - Stage: Parse sessions (parallelizable with max_workers)
    - Load: Bulk insert in batches (batch_size sessions per transaction)

    Args:
        source_folder: Path to Claude projects folder
        output_dir: Path for output
        schema_type: "simple" (4 tables) or "star" (dimensional model)
        include_agents: Whether to include agent sessions
        include_thinking: Whether to include thinking blocks
        truncate_output: Max chars for tool output
        progress_callback: Optional callback with signature:
            callback(project_name, session_name, current, total, stats)
            where stats is a dict with 'rows_inserted', 'db_size_mb', 'rate'
        max_workers: Number of parallel workers for staging (default: 1)
        batch_size: Sessions per transaction batch (default: 10)

    Returns:
        dict with statistics including row counts
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "archive.duckdb"

    # Create appropriate schema
    if schema_type == "star":
        conn = create_star_schema(db_path)
        etl_func = run_star_schema_etl
    else:
        conn = create_duckdb_schema(db_path)
        etl_func = export_session_to_duckdb

    projects = find_all_sessions(source_folder, include_agents=include_agents)

    total_session_count = sum(len(p["sessions"]) for p in projects)
    processed_count = 0
    successful_sessions = 0
    failed_sessions = []

    # Stats tracking
    start_time = time.time()

    # Flatten sessions for processing
    session_tasks = []
    for project in projects:
        project_name = project["name"]
        for session in project["sessions"]:
            session_tasks.append((project_name, session["path"]))

    # Process sessions
    if max_workers > 1 and len(session_tasks) > 1:
        # Parallel processing - stage then load in batches
        _process_parallel(
            conn,
            session_tasks,
            etl_func,
            include_thinking,
            truncate_output,
            batch_size,
            progress_callback,
            db_path,
            start_time,
            failed_sessions,
            schema_type,
        )
        successful_sessions = len(session_tasks) - len(failed_sessions)
    else:
        # Sequential processing (original behavior)
        for project_name, session_path in session_tasks:
            try:
                etl_func(
                    conn,
                    session_path,
                    project_name,
                    include_thinking=include_thinking,
                    truncate_output=truncate_output,
                )
                successful_sessions += 1
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project_name,
                        "session": session_path.stem,
                        "error": str(e),
                    }
                )

            processed_count += 1
            if progress_callback:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                db_size = _get_db_size_mb(db_path)
                stats = {
                    "rows_inserted": _count_rows(conn, schema_type),
                    "db_size_mb": db_size,
                    "rate": rate,
                }
                progress_callback(
                    project_name,
                    session_path.stem,
                    processed_count,
                    total_session_count,
                    stats,
                )

    # Post-ETL batch processing for star schema
    if schema_type == "star":
        _calculate_session_depths(conn)
        _build_session_chains(conn)
        _link_agent_delegations(conn)
        _build_session_file_bridge(conn)

    # Get final row counts
    final_row_count = _count_rows(conn, schema_type)
    final_db_size = _get_db_size_mb(db_path)

    conn.close()

    return {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
        "db_path": db_path,
        "schema_type": schema_type,
        "rows_inserted": final_row_count,
        "db_size_mb": final_db_size,
    }


def _process_parallel(
    conn,
    session_tasks,
    etl_func,
    include_thinking,
    truncate_output,
    batch_size,
    progress_callback,
    db_path,
    start_time,
    failed_sessions,
    schema_type,
):
    """Process sessions in batches with progress reporting.

    Note: DuckDB connections are not thread-safe for writes, so we
    process in batches and serialize the actual DB writes.
    """
    total = len(session_tasks)
    processed = 0
    rows_total = 0

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = session_tasks[batch_start:batch_end]

        # Process batch - serialize DB writes
        for project_name, session_path in batch:
            try:
                etl_func(
                    conn,
                    session_path,
                    project_name,
                    include_thinking=include_thinking,
                    truncate_output=truncate_output,
                )
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project_name,
                        "session": session_path.stem,
                        "error": str(e),
                    }
                )

            processed += 1
            if progress_callback:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                db_size = _get_db_size_mb(db_path)
                # Count rows periodically (expensive, so estimate)
                if processed % 5 == 0:
                    rows_total = _count_rows(conn, schema_type)
                stats = {
                    "rows_inserted": rows_total,
                    "db_size_mb": db_size,
                    "rate": rate,
                }
                progress_callback(
                    project_name,
                    session_path.stem,
                    processed,
                    total,
                    stats,
                )


def _count_rows(conn, schema_type):
    """Count total rows across relevant tables."""
    if schema_type == "star":
        tables = [
            "fact_messages",
            "fact_tool_calls",
            "fact_content_blocks",
            "fact_session_summary",
        ]
    else:
        tables = ["messages", "tool_calls", "sessions"]

    total = 0
    for table in tables:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            total += result[0] if result else 0
        except Exception:
            pass
    return total


def _get_db_size_mb(db_path):
    """Get database file size in MB."""
    try:
        size_bytes = os.path.getsize(db_path)
        return round(size_bytes / (1024 * 1024), 2)
    except Exception:
        return 0.0


def generate_star_json_archive(
    source_folder,
    output_dir,
    include_agents=False,
    include_thinking=False,
    truncate_output=2000,
    progress_callback=None,
    max_workers=1,
    batch_size=10,
):
    """Generate star schema JSON archive for all sessions.

    Creates a JSON directory structure with dimensions/ and facts/ subdirs.

    Args:
        source_folder: Path to Claude projects folder
        output_dir: Path for output
        include_agents: Whether to include agent sessions
        include_thinking: Whether to include thinking blocks
        truncate_output: Max chars for tool output
        progress_callback: Optional progress callback
        max_workers: Number of parallel workers (default: 1)
        batch_size: Sessions per batch (default: 10)

    Returns:
        dict with statistics
    """
    import tempfile

    source_folder = Path(source_folder)
    output_dir = Path(output_dir)

    # First build the DuckDB, then export to JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Generate DuckDB with star schema
        stats = generate_duckdb_archive(
            source_folder,
            tmp_path,
            schema_type="star",
            include_agents=include_agents,
            include_thinking=include_thinking,
            truncate_output=truncate_output,
            progress_callback=progress_callback,
            max_workers=max_workers,
            batch_size=batch_size,
        )

        # Export to JSON
        db_path = tmp_path / "archive.duckdb"
        conn = duckdb.connect(str(db_path))
        export_star_schema_to_json(conn, output_dir)
        conn.close()

    stats["output_dir"] = output_dir
    stats["db_path"] = None  # No DuckDB file for JSON export
    return stats


def _calculate_session_depths(conn):
    """Calculate depth_level for all sessions based on parent-child relationships.

    Iteratively sets depth levels:
    - Root sessions (no parent) get depth 0
    - Children get parent's depth + 1
    - Caps at 100 iterations for safety
    - Orphans (unresolvable parent) stay at 0
    """
    # Root sessions already default to 0
    # Iteratively resolve children
    for _ in range(100):
        updated = conn.execute(
            """
            UPDATE dim_session child
            SET depth_level = parent.depth_level + 1
            FROM dim_session parent
            WHERE child.parent_session_key = parent.session_key
              AND child.parent_session_key IS NOT NULL
              AND child.is_agent = TRUE
              AND parent.depth_level IS NOT NULL
              AND child.depth_level = 0
              AND parent.depth_level >= 0
              AND child.session_key != parent.session_key
            """
        )
        if updated.fetchone() is None:
            break
        # Check if any rows were actually changed
        remaining = conn.execute(
            """
            SELECT COUNT(*) FROM dim_session child
            JOIN dim_session parent ON child.parent_session_key = parent.session_key
            WHERE child.is_agent = TRUE
              AND child.depth_level = 0
              AND parent.depth_level > 0
            """
        ).fetchone()
        if remaining is None or remaining[0] == 0:
            break


def _build_session_chains(conn):
    """Build session chain records from sessions sharing the same slug.

    Groups sessions by slug, creates a dim_session_chain record for each group,
    and updates dim_session.chain_key for all sessions in the chain.
    """
    # Find all distinct slugs with their sessions
    slug_groups = conn.execute(
        """
        SELECT slug, COUNT(*) as cnt,
               MIN(first_timestamp) as first_ts,
               MAX(last_timestamp) as last_ts,
               MIN(session_key) as first_sk,
               MAX(session_key) as last_sk,
               MIN(project_key) as proj_key
        FROM dim_session
        WHERE slug IS NOT NULL AND slug != ''
        GROUP BY slug
        """
    ).fetchall()

    for row in slug_groups:
        slug = row[0]
        session_count = row[1]
        first_ts = row[2]
        last_ts = row[3]
        first_sk = row[4]
        last_sk = row[5]
        proj_key = row[6]

        chain_key = generate_dimension_key(slug)

        # Calculate total duration
        total_duration = 0
        if first_ts and last_ts:
            try:
                total_duration = int((last_ts - first_ts).total_seconds())
            except (TypeError, AttributeError):
                pass

        # Insert chain record
        if not conn.execute(
            "SELECT 1 FROM dim_session_chain WHERE chain_key = ?", [chain_key]
        ).fetchone():
            conn.execute(
                """INSERT INTO dim_session_chain
                   (chain_key, slug, project_key, first_session_key,
                    last_session_key, session_count, first_timestamp,
                    last_timestamp, total_duration_seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    chain_key,
                    slug,
                    proj_key,
                    first_sk,
                    last_sk,
                    session_count,
                    first_ts,
                    last_ts,
                    total_duration,
                ],
            )

        # Update all sessions in this chain
        conn.execute(
            "UPDATE dim_session SET chain_key = ? WHERE slug = ?",
            [chain_key, slug],
        )


def _link_agent_delegations(conn):
    """Link agent sessions to their parent's Task tool_use calls.

    For each agent session (is_agent=TRUE with parent_session_key):
    1. Find Task tool_use blocks in the parent session's fact_tool_calls
    2. Match by timestamp proximity (agent's first_timestamp closest to Task call)
    3. Extract description/prompt/subagent_type from input_json
    4. Set match_confidence based on match quality
    """
    import json

    # Find all agent sessions with parents
    agents = conn.execute(
        """
        SELECT a.session_key, a.parent_session_key, a.first_timestamp, a.last_timestamp
        FROM dim_session a
        WHERE a.is_agent = TRUE AND a.parent_session_key IS NOT NULL
        """
    ).fetchall()

    for agent_row in agents:
        agent_session_key = agent_row[0]
        parent_session_key = agent_row[1]
        agent_first_ts = agent_row[2]
        agent_last_ts = agent_row[3]

        # Skip if delegation already exists
        if conn.execute(
            "SELECT 1 FROM fact_agent_delegations WHERE agent_session_key = ?",
            [agent_session_key],
        ).fetchone():
            continue

        # Find Task tool calls in the parent session
        task_calls = conn.execute(
            """
            SELECT ftc.tool_call_id, ftc.timestamp, ftc.input_json,
                   ftc.is_error, ftc.output_text, ftc.date_key, ftc.time_key
            FROM fact_tool_calls ftc
            JOIN dim_tool dt ON ftc.tool_key = dt.tool_key
            WHERE ftc.session_key = ? AND dt.tool_name = 'Task'
            ORDER BY ftc.timestamp
            """,
            [parent_session_key],
        ).fetchall()

        if not task_calls:
            continue

        # Find the best matching Task call by timestamp proximity
        best_match = None
        best_distance = float("inf")

        for tc in task_calls:
            tc_timestamp = tc[1]
            if tc_timestamp and agent_first_ts:
                try:
                    distance = abs((agent_first_ts - tc_timestamp).total_seconds())
                    if distance < best_distance:
                        best_distance = distance
                        best_match = tc
                except (TypeError, AttributeError):
                    continue

        if best_match is None:
            continue

        # Parse Task input
        tool_call_id = best_match[0]
        tc_timestamp = best_match[1]
        input_json_str = best_match[2]
        is_error = best_match[3]
        output_text = best_match[4]
        date_key = best_match[5]
        time_key = best_match[6]

        task_description = None
        task_prompt = None
        subagent_type = None

        if input_json_str:
            try:
                input_data = json.loads(input_json_str)
                task_description = input_data.get("description")
                task_prompt = input_data.get("prompt")
                subagent_type = input_data.get("subagent_type")
            except (json.JSONDecodeError, TypeError):
                pass

        # Determine match confidence
        # 1.0 if only one Task call matches this agent's timeframe
        # Lower for ambiguous matches
        match_confidence = 1.0
        if len(task_calls) > 1:
            # Check how many Task calls are close to this agent
            close_calls = sum(
                1
                for tc in task_calls
                if tc[1]
                and agent_first_ts
                and abs((agent_first_ts - tc[1]).total_seconds()) < 60
            )
            if close_calls > 1:
                match_confidence = 0.5
            else:
                match_confidence = 0.8

        # Determine completion status
        completion_status = "unknown"
        if is_error:
            completion_status = "error"
        elif output_text:
            completion_status = "completed"

        delegation_key = generate_dimension_key(parent_session_key, agent_session_key)

        conn.execute(
            """INSERT INTO fact_agent_delegations
               (delegation_key, parent_session_key, agent_session_key,
                task_tool_call_id, date_key, time_key,
                task_description, task_prompt, subagent_type,
                agent_output, completion_status,
                delegation_timestamp, completion_timestamp, match_confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                delegation_key,
                parent_session_key,
                agent_session_key,
                tool_call_id,
                date_key,
                time_key,
                task_description,
                task_prompt,
                subagent_type,
                output_text[:2000] if output_text else None,
                completion_status,
                tc_timestamp,
                agent_last_ts,
                match_confidence,
            ],
        )


def _build_session_file_bridge(conn):
    """Build bridge table aggregating file operations by session and file.

    Aggregates fact_file_operations into per-(session, file) summaries
    with operation counts broken down by type.
    """
    conn.execute(
        """
        INSERT INTO bridge_session_file
        SELECT
            md5(ffo.session_key || '|' || ffo.file_key) AS session_file_key,
            ffo.session_key,
            ffo.file_key,
            MIN(ffo.timestamp) AS first_operation_timestamp,
            MAX(ffo.timestamp) AS last_operation_timestamp,
            COUNT(*) AS operation_count,
            SUM(CASE WHEN ffo.operation_type = 'read' THEN 1 ELSE 0 END) AS read_count,
            SUM(CASE WHEN ffo.operation_type = 'write' THEN 1 ELSE 0 END) AS write_count,
            SUM(CASE WHEN ffo.operation_type = 'edit' THEN 1 ELSE 0 END) AS edit_count,
            SUM(COALESCE(ffo.file_size_chars, 0)) AS total_chars_written
        FROM fact_file_operations ffo
        WHERE ffo.file_key IS NOT NULL
        GROUP BY ffo.session_key, ffo.file_key
        """
    )
