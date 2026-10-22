"""Pass Pattern Clustering — detect and classify pass sequence patterns.

All numpy-only, no pandas/scipy/sklearn.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


def _zone_key(x: float, y: float) -> tuple[int, int]:
    zx = min(int(x / (PITCH_LENGTH / 3)), 2)
    zy = min(int(y / (PITCH_WIDTH / 3)), 2)
    return (zx, zy)


def _lateral_direction(seq: list[dict[str, Any]]) -> str:
    ys = [e.get("start_y", PITCH_WIDTH / 2) for e in seq] + [seq[-1].get("end_y", PITCH_WIDTH / 2)]
    total_swing = max(ys) - min(ys)
    if total_swing < PITCH_WIDTH * 0.15:
        return "center"
    avg_y = sum(ys) / len(ys)
    return "left" if avg_y < PITCH_WIDTH / 2 else "right"


def _net_forward_progress(seq: list[dict[str, Any]]) -> float:
    start_x: float = seq[0].get("start_x", PITCH_LENGTH / 2)
    end_x: float = seq[-1].get("end_x", PITCH_LENGTH / 2)
    return end_x - start_x


@dataclass
class PassPatternReport:
    team_patterns: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    build_up: dict[str, Any] = field(default_factory=dict)
    combination_zones: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_patterns": self.team_patterns,
            "build_up": self.build_up,
            "combination_zones": self.combination_zones,
        }


class PassPatternAnalyzer:
    """Analyze pass sequences to classify team build-up and combination patterns."""

    def extract_pass_sequences(
        self,
        events: list[dict[str, Any]],
        max_length: int = 4,
    ) -> list[list[dict[str, Any]]]:
        if not events:
            return []
        sorted_ev = sorted(events, key=lambda e: e.get("timestamp", 0))
        passes = [e for e in sorted_ev if e.get("type") == "pass"]
        if not passes:
            return []
        sequences: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = [passes[0]]
        current_team = passes[0].get("team", "home")

        for p in passes[1:]:
            team = p.get("team", current_team)
            if team != current_team or len(current) >= max_length:
                if len(current) >= 2:
                    sequences.append(current)
                current = [p]
                current_team = team
            else:
                current.append(p)

        if len(current) >= 2:
            sequences.append(current)
        return sequences

    def classify_sequence_pattern(
        self,
        sequence: list[dict[str, Any]],
    ) -> str:
        if len(sequence) < 2:
            return "possession_maintenance"

        start_x = sequence[0].get("start_x", PITCH_LENGTH / 2)
        _ = sequence[-1].get("end_x", PITCH_LENGTH / 2)
        side = _lateral_direction(sequence)
        forward_progress = _net_forward_progress(sequence)

        # Detect cross sequence: ends near goal line in wide areas
        last_pass = sequence[-1]
        lx = last_pass.get("end_x", PITCH_LENGTH / 2)
        ly = last_pass.get("end_y", PITCH_WIDTH / 2)
        is_cross = lx >= PITCH_LENGTH * 0.8 and (
            ly <= PITCH_WIDTH * 0.25 or ly >= PITCH_WIDTH * 0.75
        )
        if is_cross:
            return "cross_sequence"

        # Detect switch of play: large lateral movement across midfield
        ys = [e.get("start_y", PITCH_WIDTH / 2) for e in sequence] + [
            sequence[-1].get("end_y", PITCH_WIDTH / 2)
        ]
        if max(ys) - min(ys) >= PITCH_WIDTH * 0.55:
            return "switch_of_play"

        # Detect combination play (1-2 passing): pass to A, immediate return
        if len(sequence) >= 2:
            for i in range(len(sequence) - 1):
                p1_from = sequence[i].get("from_track_id")
                p1_to = sequence[i].get("to_track_id")
                p2_from = sequence[i + 1].get("from_track_id")
                p2_to = sequence[i + 1].get("to_track_id")
                if p1_from == p2_to and p1_to == p2_from:
                    return "combination_play"

        # Detect progressive patterns
        if forward_progress >= PITCH_LENGTH * 0.2:
            return f"progressive_{side}"

        # Detect build-up (starts in defensive third)
        if start_x <= PITCH_LENGTH * 0.33:
            return f"build_up_{side}"

        return "possession_maintenance"

    def cluster_sequences_by_pattern(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, dict[str, dict[str, float]]]:
        sequences = self.extract_pass_sequences(events)
        team_counts: dict[str, defaultdict[str, int]] = {}
        for seq in sequences:
            team = seq[0].get("team", "home")
            if team not in team_counts:
                team_counts[team] = defaultdict(int)
            pattern = self.classify_sequence_pattern(seq)
            team_counts[team][pattern] += 1

        result: dict[str, dict[str, dict[str, float]]] = {}
        for team, counts in team_counts.items():
            total = sum(counts.values())
            result[team] = {
                k: {"count": float(v), "pct_of_total": round(v / total * 100, 1) if total else 0.0}
                for k, v in counts.items()
            }
        return result

    def detect_build_up_pattern(
        self,
        events: list[dict[str, Any]],
        team: str,
    ) -> dict[str, Any]:
        team_events = [e for e in events if e.get("team") == team and e.get("type") == "pass"]
        sequences = self.extract_pass_sequences(team_events)
        build_up_seqs = [
            s for s in sequences if self.classify_sequence_pattern(s).startswith("build_up")
        ]

        if not build_up_seqs:
            return {
                "primary_side": "center",
                "build_up_pct_per_side": {"left": 0, "center": 0, "right": 0},
                "avg_passes_per_build_up": 0,
                "build_up_to_shot_rate": 0,
            }

        side_counts: dict[str, int] = {"left": 0, "center": 0, "right": 0}
        total_passes = 0
        shot_after = 0

        for seq in build_up_seqs:
            pattern = self.classify_sequence_pattern(seq)
            for s in ("left", "center", "right"):
                if pattern.endswith(s):
                    side_counts[s] += 1
                    break
            total_passes += len(seq)

        total = sum(side_counts.values()) or 1
        pct_per_side = {k: round(v / total * 100, 1) for k, v in side_counts.items()}
        primary_side = max(side_counts, key=lambda k: side_counts[k])

        # Count how many build-up sequences ended with a shot within 5 events
        sorted_ev = sorted(team_events, key=lambda e: e.get("timestamp", 0))
        _ = 0
        for seq in build_up_seqs:
            last_ts = seq[-1].get("timestamp", 0)
            for ev in sorted_ev:
                if ev.get("timestamp", 0) > last_ts and ev.get("type") == "shot":
                    if ev.get("timestamp", 0) - last_ts <= 15:
                        shot_after += 1
                    break

        return {
            "primary_side": primary_side,
            "build_up_pct_per_side": pct_per_side,
            "avg_passes_per_build_up": round(total_passes / len(build_up_seqs), 1),
            "build_up_to_shot_rate": round(shot_after / len(build_up_seqs), 2),
        }

    def compute_combination_play_frequency(
        self,
        events: list[dict[str, Any]],
        team: str,
    ) -> dict[str, Any]:
        team_passes = [e for e in events if e.get("team") == team and e.get("type") == "pass"]
        sorted_passes = sorted(team_passes, key=lambda e: e.get("timestamp", 0))

        combos: list[dict[str, Any]] = []
        for i in range(len(sorted_passes) - 1):
            p1 = sorted_passes[i]
            p2 = sorted_passes[i + 1]
            p1_from = p1.get("from_track_id")
            p1_to = p1.get("to_track_id")
            p2_from = p2.get("from_track_id")
            p2_to = p2.get("to_track_id")
            if p1_from == p2_to and p1_to == p2_from:
                zone_x = (
                    p1.get("start_x", PITCH_LENGTH / 2) + p1.get("end_x", PITCH_LENGTH / 2)
                ) / 2
                zone_y = (p1.get("start_y", PITCH_WIDTH / 2) + p1.get("end_y", PITCH_WIDTH / 2)) / 2
                combos.append(
                    {
                        "zone_x": round(zone_x, 1),
                        "zone_y": round(zone_y, 1),
                        "player_1_track_id": p1_from,
                        "player_2_track_id": p1_to,
                        "timestamp": p1.get("timestamp", 0),
                    }
                )

        zones: list[dict[str, Any]] = []
        for c in combos:
            zx = min(int(c["zone_x"] / (PITCH_LENGTH / 5)), 4)
            zy = min(int(c["zone_y"] / (PITCH_WIDTH / 5)), 4)
            zones.append({"zone": f"{zx}_{zy}", "x": c["zone_x"], "y": c["zone_y"]})

        zone_counts: dict[str, int] = defaultdict(int)
        for z in zones:
            zone_counts[z["zone"]] += 1

        return {
            "team": team,
            "total_combinations": len(combos),
            "combinations_per_90": round(len(combos) / 90 * 90, 1),
            "zones": [
                {"zone": k, "count": v} for k, v in sorted(zone_counts.items(), key=lambda x: -x[1])
            ],
        }
