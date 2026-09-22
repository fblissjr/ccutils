"""The record-source allow-list lives outside the warehouse layers.

Tier 1 writers (parsers/) must validate their provenance labels without
importing etl/ or schemas/: the Claude Code lake writer never validated its
label at all, and the first attempt to fix that pulled the whole star-schema
package into the parser import graph. Deleting these tests lets the
allow-list drift back into etl/ and the label check quietly disappear from
Tier 1 again.
"""

import ast
from pathlib import Path

import pytest

from ccutils import provenance
from ccutils.etl import lineage

SRC = Path(provenance.__file__)


def test_lineage_reexports_the_same_allow_list():
    # Two copies of the allow-list is the drift this module exists to prevent.
    assert lineage._RECORD_SOURCES is provenance._RECORD_SOURCES
    assert lineage.record_source_label is provenance.record_source_label


def test_provenance_imports_nothing_from_the_warehouse_layers():
    tree = ast.parse(SRC.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    offenders = {m for m in imported if "etl" in m or "schemas" in m}
    assert not offenders, offenders


@pytest.mark.parametrize("label", [
    "antigravity_conversation_db",
    "antigravity_summaries_db",
    "antigravity_conversation_file",
    "antigravity_store_file",
    "antigravity_encrypted",
    "antigravity_config_projects",
])
def test_antigravity_labels_are_allowed(label):
    assert provenance.record_source_label(label) == label


def test_unknown_label_names_the_module_to_edit():
    with pytest.raises(ValueError, match="provenance.py"):
        provenance.record_source_label("antigravity_typo")
