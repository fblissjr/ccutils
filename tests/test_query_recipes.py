"""Every SQL recipe in the query-warehouse skill runs against the current schema.

Claim: the recipes are instructions an agent copies, and nothing connected
them to the DDL. At 1.0.0 `semantic_agent_delegations` changed its columns
and the delegation recipe went on naming the old ones; the first agent to
use it would have got a binder error, or worse, quietly rewritten the query
around a column that no longer meant what the comment said.

An empty warehouse is enough. DuckDB binds every table and column name
before it looks at a row, so a recipe naming something that does not exist
fails here whatever the data holds. What this cannot see is a recipe that
still binds and now means something else; that stays a review job.

Delete this and the skill drifts from the schema one rename at a time.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pytest

from ccutils import create_star_schema

SKILL_DIR = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "query-warehouse"
RECIPE_FILES = sorted(SKILL_DIR.rglob("*.md"))

_SQL_BLOCK = re.compile(r"^```sql\n(.*?)^```", re.M | re.S)


def sql_statements(markdown: str) -> list[str]:
    """Each statement of each ```sql block, comments dropped.

    Comments go before the split because a recipe's trailing comment may
    hold a semicolon. No recipe carries `--` inside a string literal; if one
    ever does, it fails loudly here as a parse error rather than slipping by.
    """
    statements = []
    for block in _SQL_BLOCK.findall(markdown):
        code = "\n".join(line.split("--", 1)[0] for line in block.splitlines())
        statements += [s.strip() for s in code.split(";") if s.strip()]
    return statements


def _recipes():
    for path in RECIPE_FILES:
        for i, statement in enumerate(sql_statements(path.read_text())):
            yield pytest.param(statement, id=f"{path.name}:{i}")


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    conn = create_star_schema(tmp_path_factory.mktemp("recipes") / "w.duckdb")
    yield conn
    conn.close()


def test_the_skill_has_recipes_to_check():
    """Non-vacuity: a moved file or a changed fence would leave nothing to run."""
    assert RECIPE_FILES, f"no markdown under {SKILL_DIR}"
    assert sum(len(sql_statements(p.read_text())) for p in RECIPE_FILES) >= 10


@pytest.mark.parametrize("statement", list(_recipes()))
def test_recipe_binds_against_the_current_schema(warehouse, statement):
    try:
        warehouse.execute(statement).fetchall()
    except duckdb.Error as exc:
        pytest.fail(f"{type(exc).__name__}: {exc}\n\n{statement}")


def test_the_extractor_sees_a_broken_recipe(warehouse):
    """The oracle can fail: a recipe naming a column that is gone is caught."""
    (statement,) = sql_statements(
        "```sql\n-- old name\nSELECT agent_derived_io_tokens FROM semantic_agent_delegations;\n```\n"
    )
    with pytest.raises(duckdb.BinderException):
        warehouse.execute(statement)
