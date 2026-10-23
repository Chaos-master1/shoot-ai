"""Player position vocabulary shared by the importer, cohort and benchmark.

StatsBomb names positions in a long form ("Right Center Midfield") while
Kawkab stores short codes ("CM"). The player benchmark compares players
*within a position group* (GK/DEF/MID/FWD), so the mapping must be one
owner rather than three hand-kept copies.

The importer previously fell back to "MID" for any position it did not
recognise. That is a silent lie: a right wing back became a midfielder and
could be compared with the wrong population. This module returns "" for an
unknown name — the benchmark then reports that the position cannot be
measured instead of inventing one.

Pure data. No I/O.
"""

from __future__ import annotations

#: StatsBomb position name -> Kawkab position code.
STATSBOMB_POSITION_CODES: dict[str, str] = {
    "Goalkeeper": "GK",
    "Right Center Back": "CB",
    "Left Center Back": "CB",
    "Center Back": "CB",
    "Right Back": "RB",
    "Left Back": "LB",
    "Right Wing Back": "RWB",
    "Left Wing Back": "LWB",
    "Defensive Midfield": "DM",
    "Right Defensive Midfield": "DM",
    "Left Defensive Midfield": "DM",
    "Center Defensive Midfield": "DM",
    "Right Center Midfield": "CM",
    "Left Center Midfield": "CM",
    "Center Midfield": "CM",
    "Right Midfield": "RM",
    "Left Midfield": "LM",
    "Center Attacking Midfield": "AM",
    "Right Attacking Midfield": "AM",
    "Left Attacking Midfield": "AM",
    "Right Wing": "RW",
    "Left Wing": "LW",
    "Center Forward": "CF",
    "Left Center Forward": "CF",
    "Right Center Forward": "CF",
    "Striker": "ST",
    "Secondary Striker": "SS",
}

#: Kawkab code -> the group a player is benchmarked against. Broad groups
#: are deliberate: a 297-match cohort has enough centre-backs for a stable
#: percentile, but not enough "left centre-backs" for one.
POSITION_GROUPS: dict[str, str] = {
    "GK": "GK",
    "CB": "DEF",
    "RB": "DEF",
    "LB": "DEF",
    "RWB": "DEF",
    "LWB": "DEF",
    "DM": "MID",
    "CM": "MID",
    "RM": "MID",
    "LM": "MID",
    "AM": "MID",
    "RW": "FWD",
    "LW": "FWD",
    "CF": "FWD",
    "ST": "FWD",
    "SS": "FWD",
}

POSITION_GROUP_ORDER: tuple[str, ...] = ("GK", "DEF", "MID", "FWD")

POSITION_GROUP_LABELS: dict[str, str] = {
    "GK": "goalkeepers",
    "DEF": "defenders",
    "MID": "midfielders",
    "FWD": "forwards",
}


def position_code(name: str | None) -> str:
    """StatsBomb position name -> Kawkab code, or "" when unrecognised."""
    return STATSBOMB_POSITION_CODES.get((name or "").strip(), "")


def position_group(code: str | None) -> str | None:
    """Kawkab position code -> GK/DEF/MID/FWD, or None when unknown."""
    return POSITION_GROUPS.get((code or "").strip().upper())


def position_group_label(group: str | None) -> str:
    """Human label for a group (falls back to the raw group key)."""
    return POSITION_GROUP_LABELS.get(group or "", group or "")
