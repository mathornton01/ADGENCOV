"""Group-map / annotation table handling for the symmetry partitions.

Four of the built-in symmetries are not derivable from the expression matrix
alone and need an external two-column table:

===================  ==================  ==============================
Grouping             Table columns       Passed to build_group_labels as
===================  ==================  ==============================
``chromosome``       gene, chromosome    ``annotation``
``reactome``         gene, group         ``group_map``
``go_process``       gene, group         ``group_map``
``custom_group_map`` gene, group         ``group_map``
``hierarchical_wreath`` gene, group      ``group_map``
===================  ==================  ==============================

The C++ reader takes a path, so callers that hold the table as text (the HTTP
service, which receives it in the request body) go through :func:`parse_table`,
which writes a temp file and parses it with the same delimiter-sniffing reader
the CLI uses.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

from ._core import read_table

#: Groupings that require an external table, mapped to the ``build_group_labels``
#: keyword the table is passed as and the column the table must carry.
GROUPS_NEEDING_TABLE: Dict[str, Dict[str, str]] = {
    "chromosome": {"kwarg": "annotation", "column": "chromosome"},
    "reactome": {"kwarg": "group_map", "column": "group"},
    "go_process": {"kwarg": "group_map", "column": "group"},
    "custom_group_map": {"kwarg": "group_map", "column": "group"},
    "hierarchical_wreath": {"kwarg": "group_map", "column": "group"},
}

#: Groupings derivable from the expression matrix alone.
SELF_CONTAINED_GROUPS = ("none", "gene_family", "correlation_blocks", "auto")


def needs_table(group: str) -> bool:
    """Does *group* require an external annotation / group-map table?"""
    return group in GROUPS_NEEDING_TABLE


def required_column(group: str) -> Optional[str]:
    """The non-gene column *group*'s table must carry, or None."""
    spec = GROUPS_NEEDING_TABLE.get(group)
    return spec["column"] if spec else None


def parse_table(text: str, group: str) -> Any:
    """Parse group-map *text* into a C++ ``Table``, validated for *group*.

    Raises ``ValueError`` with an actionable message when the table is empty,
    lacks a ``gene`` column, or lacks the column the grouping needs — these are
    surfaced to the user as a 422 rather than a mid-analysis crash.
    """
    spec = GROUPS_NEEDING_TABLE.get(group)
    if spec is None:
        raise ValueError(f"grouping {group!r} does not take a group map")
    if not text or not text.strip():
        raise ValueError(
            f"grouping {group!r} needs a group map with columns "
            f"gene,{spec['column']}, but the supplied table is empty"
        )
    with tempfile.TemporaryDirectory(prefix="adgencov_gmap_") as td:
        path = os.path.join(td, "group_map.tsv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
        table = read_table(path)

    if table.col_index("gene") < 0:
        raise ValueError(
            f"group map must have a 'gene' column; found {list(table.headers)}"
        )
    if table.col_index(spec["column"]) < 0:
        raise ValueError(
            f"grouping {group!r} needs a {spec['column']!r} column; "
            f"found {list(table.headers)}"
        )
    if table.nrow == 0:
        raise ValueError("group map has a header but no rows")
    return table


def build_kwargs(group: str, text: Optional[str]) -> Dict[str, Any]:
    """Keyword arguments for ``build_group_labels`` given raw group-map *text*.

    Returns an empty dict for the self-contained groupings, so callers can
    always splat the result without branching.
    """
    if not needs_table(group):
        return {}
    spec = GROUPS_NEEDING_TABLE[group]
    return {spec["kwarg"]: parse_table(text or "", group)}
