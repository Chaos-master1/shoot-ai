"""Recurring Pattern Detection.

Detects recurring tactical sequences, identifies signature patterns
within matches, and finds cross-match patterns. All numpy-only.
"""

from __future__ import annotations

from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M
SIMILARITY_THRESHOLD = 0.7


def _zone_index(x: float, y: float) -> int:
    zx = min(int(x / (PITCH_LENGTH / 3)), 2)
    zy = min(int(y / (PITCH_WIDTH / 3)), 2)
    return zx * 3 + zy


def _sequence_zones(seq: list[dict[str, Any]]) -> list[int]:
    zones: list[int] = []
    for e in seq:
        sx = float(e.get("start_x", PITCH_LENGTH / 2))
        sy = float(e.get("start_y", PITCH_WIDTH / 2))
        zones.append(_zone_index(sx, sy))
        ex = float(e.get("end_x", PITCH_LENGTH / 2))
        ey = float(e.get("end_y", PITCH_WIDTH / 2))
        zones.append(_zone_index(ex, ey))
    return zones


def _zone_similarity(seq_a: list[dict[str, Any]], seq_b: list[dict[str, Any]]) -> float:
    za = _sequence_zones(seq_a)
    zb = _sequence_zones(seq_b)
    if not za or not zb:
        return 0.0
    if len(za) != len(zb):
        longer = max(len(za), len(zb))
        shorter = min(len(za), len(zb))
        ratio = shorter / longer
        overlap = sum(1 for i in range(min(len(za), len(zb))) if za[i] == zb[i])
        return (overlap / longer) * ratio
    matches = sum(1 for i in range(len(za)) if za[i] == zb[i])
    return matches / len(za)


class TacticalPatternDetector:
    def detect_recurring_sequences(
        self, events: list[dict[str, Any]], team: str, min_occurrences: int = 2
    ) -> list[dict[str, Any]]:
        from kawkab.core.pass_patterns import PassPatternAnalyzer

        ppa = PassPatternAnalyzer()
        team_events = [e for e in events if e.get("team") == team and e.get("type") == "pass"]
        if not team_events:
            return []
        sequences = ppa.extract_pass_sequences(team_events, max_length=4)
        if len(sequences) < 2:
            return []
        clusters: list[list[list[dict[str, Any]]]] = []
        assigned = [False] * len(sequences)
        for i in range(len(sequences)):
            if assigned[i]:
                continue
            cluster = [sequences[i]]
            assigned[i] = True
            for j in range(i + 1, len(sequences)):
                if assigned[j]:
                    continue
                if _zone_similarity(sequences[i], sequences[j]) >= SIMILARITY_THRESHOLD:
                    cluster.append(sequences[j])
                    assigned[j] = True
            clusters.append(cluster)
        result: list[dict[str, Any]] = []
        for cluster in clusters:
            if len(cluster) < min_occurrences:
                continue
            rep = cluster[0]
            shots = 0
            goals = 0
            last_ts = rep[-1].get("timestamp", 0)
            for ev in events:
                if (
                    ev.get("team") == team
                    and ev.get("type") == "shot"
                    and 0 < ev.get("timestamp", 0) - last_ts <= 15
                ):
                    shots += 1
                    if ev.get("is_goal"):
                        goals += 1
            result.append(
                {
                    "count": len(cluster),
                    "representative_seq": [
                        {
                            "from": e.get("from_track_id"),
                            "to": e.get("to_track_id"),
                            "start_x": e.get("start_x"),
                            "start_y": e.get("start_y"),
                            "end_x": e.get("end_x"),
                            "end_y": e.get("end_y"),
                        }
                        for e in rep
                    ],
                    "shot_rate": round(shots / len(cluster), 2) if cluster else 0,
                    "goal_rate": round(goals / len(cluster), 2) if cluster else 0,
                }
            )
        return sorted(result, key=lambda x: -x["count"])

    def identify_signature_patterns(
        self, events: list[dict[str, Any]], team: str, min_frequency: int = 3
    ) -> list[dict[str, Any]]:
        recurring = self.detect_recurring_sequences(events, team, min_occurrences=min_frequency)
        return sorted(
            [r for r in recurring if r["count"] >= min_frequency],
            key=lambda x: (x["count"], x["shot_rate"]),
            reverse=True,
        )

    def compare_patterns_across_matches(
        self, match_events_list: list[list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        all_patterns: list[dict[str, Any]] = []
        for midx, events in enumerate(match_events_list):
            for team in {e.get("team", "") for e in events if e.get("type") == "pass"}:
                patterns = self.detect_recurring_sequences(events, team, min_occurrences=1)
                for p in patterns:
                    all_patterns.append(
                        {
                            "match_idx": midx,
                            "team": team,
                            "count": p["count"],
                            "shot_rate": p["shot_rate"],
                            "goal_rate": p["goal_rate"],
                            "representative_seq": p["representative_seq"],
                        }
                    )
        if not all_patterns:
            return []
        cross_match: list[dict[str, Any]] = []
        handled = [False] * len(all_patterns)
        for i in range(len(all_patterns)):
            if handled[i]:
                continue
            group = [all_patterns[i]]
            handled[i] = True
            for j in range(i + 1, len(all_patterns)):
                if handled[j]:
                    continue
                pi_seq = [
                    {
                        "start_x": e["start_x"],
                        "start_y": e["start_y"],
                        "end_x": e["end_x"],
                        "end_y": e["end_y"],
                    }
                    for e in all_patterns[i].get("representative_seq", [])
                ]
                pj_seq = [
                    {
                        "start_x": e["start_x"],
                        "start_y": e["start_y"],
                        "end_x": e["end_x"],
                        "end_y": e["end_y"],
                    }
                    for e in all_patterns[j].get("representative_seq", [])
                ]
                if _zone_similarity(pi_seq, pj_seq) >= SIMILARITY_THRESHOLD:
                    group.append(all_patterns[j])
                    handled[j] = True
            if len(group) >= 2:
                matches = list({p["match_idx"] for p in group})
                if len(matches) >= 2:
                    avg_shot_rate = sum(p["shot_rate"] for p in group) / len(group)
                    cross_match.append(
                        {
                            "match_count": len(matches),
                            "matches": sorted(matches),
                            "total_occurrences": sum(p["count"] for p in group),
                            "avg_shot_rate": round(avg_shot_rate, 2),
                            "teams_involved": list({p["team"] for p in group}),
                            "representative_seq": group[0]["representative_seq"],
                        }
                    )
        return sorted(cross_match, key=lambda x: -x["total_occurrences"])
