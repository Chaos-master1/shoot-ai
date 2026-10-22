from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class TagTemplate:
    name: str
    category: str
    color: str = "#666666"
    shortcut: str = ""
    description: str = ""


DEFAULT_TAG_TEMPLATES: list[TagTemplate] = [
    TagTemplate("Shot", "attack", "#e74c3c", "1", "Shot on goal"),
    TagTemplate("Pass", "attack", "#2ecc71", "2", "Pass completion"),
    TagTemplate("Key Pass", "attack", "#f39c12", "3", "Chance-creating pass"),
    TagTemplate("Cross", "attack", "#9b59b6", "4", "Cross into box"),
    TagTemplate("Dribble", "attack", "#1abc9c", "5", "Successful dribble"),
    TagTemplate("Through Ball", "attack", "#3498db", "6", "Line-breaking pass"),
    TagTemplate("Run", "attack", "#e67e22", "7", "Off-ball run"),
    TagTemplate("First Touch", "attack", "#2c3e50", "8", "First touch control"),
    TagTemplate("Tackle", "defense", "#c0392b", "q", "Tackle won"),
    TagTemplate("Interception", "defense", "#27ae60", "w", "Pass intercepted"),
    TagTemplate("Clearance", "defense", "#8e44ad", "e", "Defensive clearance"),
    TagTemplate("Block", "defense", "#d35400", "r", "Shot blocked"),
    TagTemplate("Pressure", "defense", "#2980b9", "t", "Pressing opponent"),
    TagTemplate("Cover", "defense", "#16a085", "z", "Covering teammate"),
    TagTemplate("Mistake", "mistake", "#e74c3c", "x", "Error leading to chance"),
    TagTemplate("Bad Pass", "mistake", "#c0392b", "c", "Pass turnover"),
    TagTemplate("Lost Duel", "mistake", "#8e44ad", "v", "Lost 1v1 duel"),
    TagTemplate("Positional Error", "mistake", "#d35400", "b", "Out of position"),
    TagTemplate("Foul", "mistake", "#e67e22", "n", "Foul committed"),
    TagTemplate("Handball", "mistake", "#2c3e50", "m", "Handball"),
    TagTemplate("Corner", "set_piece", "#3498db", ",", "Corner kick"),
    TagTemplate("Free Kick", "set_piece", "#2ecc71", ".", "Free kick taken"),
    TagTemplate("Throw In", "set_piece", "#f39c12", "/", "Throw-in"),
    TagTemplate("Penalty", "set_piece", "#e74c3c", "p", "Penalty kick"),
    TagTemplate("Goal Kick", "set_piece", "#1abc9c", "s", "Goal kick restart"),
]


@dataclass
class TagAnalytics:
    total_tags: int = 0
    by_category: Counter = field(default_factory=Counter)
    by_type: Counter = field(default_factory=Counter)
    by_player: Counter = field(default_factory=Counter)
    by_period: Counter = field(default_factory=Counter)
    co_occurrence: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    timeline: list[dict] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_tags": self.total_tags,
            "by_category": dict(self.by_category),
            "by_type": dict(self.by_type),
            "by_player": dict(self.by_player),
            "by_period": dict(self.by_period),
            "co_occurrence": {k: dict(v) for k, v in self.co_occurrence.items()},
            "timeline": self.timeline[:50],
            "patterns": self.patterns,
        }


def compute_tag_analytics(
    tags: list[dict],
    window_size_seconds: float = 30.0,
    min_pattern_support: int = 2,
) -> TagAnalytics:
    analytics = TagAnalytics()
    if not tags:
        return analytics

    analytics.total_tags = len(tags)
    seen_pairs: set[tuple[str, str]] = set()

    sorted_tags = sorted(tags, key=lambda t: t.get("timestamp", 0))

    for t in sorted_tags:
        analytics.by_category[t.get("category", "unknown")] += 1
        analytics.by_type[t.get("type", "unknown")] += 1
        player = t.get("player_name", t.get("track_id", "unknown"))
        analytics.by_player[str(player)] += 1
        period = t.get("period", "unknown")
        analytics.by_period[str(period)] += 1
        analytics.timeline.append(
            {
                "timestamp": t.get("timestamp", 0),
                "type": t.get("type", ""),
                "category": t.get("category", ""),
                "player": str(player),
            }
        )

    # co-occurrence within time window
    for i, a in enumerate(sorted_tags):
        ts_a = a.get("timestamp", 0)
        type_a = a.get("type", "unknown")
        for j in range(i + 1, len(sorted_tags)):
            b = sorted_tags[j]
            ts_b = b.get("timestamp", 0)
            if ts_b - ts_a > window_size_seconds:
                break
            type_b = b.get("type", "unknown")
            if type_a == type_b:
                continue
            pair = (type_a, type_b) if type_a < type_b else (type_b, type_a)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            analytics.co_occurrence[type_a][type_b] += 1

    # pattern mining: find common sequences of 2-3 tags
    for i in range(len(sorted_tags) - 1):
        ts_i = sorted_tags[i].get("timestamp", 0)
        type_i = sorted_tags[i].get("type", "unknown")
        for j in range(i + 1, min(i + 6, len(sorted_tags))):
            ts_j = sorted_tags[j].get("timestamp", 0)
            if ts_j - ts_i > window_size_seconds:
                break
            type_j = sorted_tags[j].get("type", "unknown")
            count = 0
            for k, _ in enumerate(range(j + 1, min(j + 6, len(sorted_tags))), start=1):
                ts_k = sorted_tags[k].get("timestamp", 0)
                if ts_k - ts_j > window_size_seconds:
                    break
                type_k = sorted_tags[k].get("type", "unknown")
                pattern_str = f"{type_i} → {type_j} → {type_k}"
                idx = _find_pattern(analytics.patterns, pattern_str)
                if idx >= 0:
                    analytics.patterns[idx]["count"] += 1
                else:
                    analytics.patterns.append(
                        {
                            "pattern": pattern_str,
                            "count": 1,
                            "tags": [type_i, type_j, type_k],
                        }
                    )
                if count >= min_pattern_support:
                    break

    analytics.patterns.sort(key=lambda p: p["count"], reverse=True)
    analytics.patterns = analytics.patterns[:20]
    return analytics


def _find_pattern(patterns: list[dict], pattern_str: str) -> int:
    for i, p in enumerate(patterns):
        if p["pattern"] == pattern_str:
            return i
    return -1


def tags_to_csv(tags: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "type", "category", "player", "period", "notes"])
    for t in sorted(tags, key=lambda x: x.get("timestamp", 0)):
        writer.writerow(
            [
                t.get("timestamp", ""),
                t.get("type", ""),
                t.get("category", ""),
                t.get("player_name", t.get("track_id", "")),
                t.get("period", ""),
                t.get("notes", ""),
            ]
        )
    return output.getvalue()


def tags_from_csv(csv_text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(csv_text))
    tags = []
    for row in reader:
        tags.append(
            {
                "timestamp": float(row.get("timestamp", 0)),
                "type": row.get("type", ""),
                "category": row.get("category", ""),
                "player_name": row.get("player", ""),
                "period": row.get("period", ""),
                "notes": row.get("notes", ""),
            }
        )
    return tags


def export_tags_sportscode(tags: list[dict]) -> str:
    """Export tags in Sportscode CSV format (code, time, notes)."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Code", "Time", "Notes", "Period"])
    for t in sorted(tags, key=lambda x: x.get("timestamp", 0)):
        writer.writerow(
            [
                t.get("type", ""),
                _format_timecode(t.get("timestamp", 0)),
                t.get("notes", ""),
                t.get("period", ""),
            ]
        )
    return output.getvalue()


def import_tags_sportscode(csv_text: str) -> list[dict]:
    """Import tags from Sportscode CSV format (code, time, notes)."""
    reader = csv.DictReader(io.StringIO(csv_text))
    tags = []
    for row in reader:
        tc = row.get("Time", "00:00:00.000")
        tags.append(
            {
                "timestamp": _parse_timecode(tc),
                "type": row.get("Code", ""),
                "notes": row.get("Notes", ""),
                "period": row.get("Period", ""),
                "category": _guess_category(row.get("Code", "")),
            }
        )
    return tags


def _format_timecode(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _parse_timecode(tc: str) -> float:
    parts = tc.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0


def _guess_category(code: str) -> str:
    attack = {"shot", "pass", "cross", "dribble", "run", "key pass", "through ball", "first touch"}
    defense = {"tackle", "interception", "clearance", "block", "pressure", "cover"}
    mistake = {"mistake", "bad pass", "lost duel", "positional error", "foul", "handball"}
    code_lower = code.lower()
    if code_lower in attack:
        return "attack"
    if code_lower in defense:
        return "defense"
    if code_lower in mistake:
        return "mistake"
    return "set_piece"
