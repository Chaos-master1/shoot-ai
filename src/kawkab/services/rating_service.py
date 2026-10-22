"""Rating service — compute per-player ratings from match events."""

from __future__ import annotations


class RatingService:
    def compute_ratings(self, events: list[dict], players: list[dict]) -> list[dict]:
        from collections import defaultdict

        player_stats: dict[int, dict] = defaultdict(
            lambda: {
                "passes": 0,
                "passes_completed": 0,
                "shots": 0,
                "goals": 0,
                "tackles": 0,
                "carries": 0,
                "dribbles": 0,
                "track_id": 0,
            }
        )

        player_names: dict[int, str] = {}
        player_teams: dict[int, str] = {}
        for p in players:
            tid = p.get("track_id", 0)
            player_names[tid] = p.get("name", f"Player {tid}")
            player_teams[tid] = p.get("team", "")

        for e in events:
            tid = e.get("from_track_id", 0)
            if tid == 0:
                continue
            stats = player_stats[tid]
            stats["track_id"] = tid
            etype = e.get("type", "")
            if etype == "pass":
                stats["passes"] += 1
                if e.get("completed", False):
                    stats["passes_completed"] += 1
            elif etype == "shot":
                stats["shots"] += 1
                if e.get("is_goal", False):
                    stats["goals"] += 1
            elif etype == "tackle":
                stats["tackles"] += 1
            elif etype == "carry":
                stats["carries"] += 1
            elif etype == "dribble":
                stats["dribbles"] += 1

        results = []
        for tid, stats in player_stats.items():
            pass_acc = stats["passes_completed"] / max(stats["passes"], 1)
            shot_rating = min(stats["goals"] * 20 + stats["shots"] * 2, 100)
            tackle_rating = stats["tackles"] * 5
            carry_rating = stats["carries"] * 2 + stats["dribbles"] * 3
            volume = 1 + stats["passes"] + stats["shots"] + stats["tackles"]
            raw = (
                pass_acc * 30
                + shot_rating * 25
                + tackle_rating * 15
                + carry_rating * 20
                + min(volume, 50)
            ) / 100.0
            rating = min(max(raw, 0.0), 100.0)
            results.append(
                {
                    "track_id": tid,
                    "name": player_names.get(tid, f"Player {tid}"),
                    "team": player_teams.get(tid, ""),
                    "rating": round(rating, 1),
                    "pass_accuracy": round(pass_acc, 3),
                    "shot_impact": round(shot_rating / 100.0, 3),
                    "tackles": stats["tackles"],
                }
            )

        return results
