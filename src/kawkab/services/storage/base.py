"""Shared storage helpers used by both storage backends."""

from __future__ import annotations

import json


def parse_metadata_json(row: dict) -> dict:
    """Move a row's raw ``metadata_json`` string into ``metadata`` (dict).

    Used by every tracking-import read on both backends; malformed JSON
    degrades to an empty dict rather than raising (an unreadable provenance
    annotation must not kill the record it describes).
    """
    if isinstance(row.get("metadata_json"), str):
        try:
            row["metadata"] = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            row["metadata"] = {}
    return row
