"""Scoreline Probability Distribution.

Computes scoreline probabilities using Poisson simulation,
match outcome aggregation, and Shannon entropy. All numpy-only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


class ScorelineDistribution:
    def compute_scoreline_probabilities(
        self, events: list[dict[str, Any]], n_sims: int = 50000
    ) -> dict[str, Any]:
        if not events:
            return {"scorelines": {}, "n_sims": 0}
        match_duration = max(e.get("timestamp", 0) for e in events)
        remaining_minutes = max(90.0 - match_duration / 60.0, 0)
        if remaining_minutes <= 0:
            goals_home = sum(
                1
                for e in events
                if e.get("type") == "shot" and e.get("is_goal") and e.get("team") == "home"
            )
            goals_away = sum(
                1
                for e in events
                if e.get("type") == "shot" and e.get("is_goal") and e.get("team") == "away"
            )
            key = f"{goals_home}-{goals_away}"
            return {"scorelines": {key: 1.0}, "n_sims": n_sims, "remaining_minutes": 0}
        # xg may be present-but-NULL (json_extract) — treat as 0, never
        # let None into the Poisson rate (TypeError killed real reports).
        xg_total_home = sum(
            (e.get("xg") or 0.0)
            for e in events
            if e.get("team") == "home" and e.get("type") == "shot"
        )
        xg_total_away = sum(
            (e.get("xg") or 0.0)
            for e in events
            if e.get("team") == "away" and e.get("type") == "shot"
        )
        elapsed_minutes = match_duration / 60.0
        xg_rate_home = xg_total_home / max(elapsed_minutes, 1)
        xg_rate_away = xg_total_away / max(elapsed_minutes, 1)
        sim_goals_home = np.random.poisson(xg_rate_home * remaining_minutes, n_sims)
        sim_goals_away = np.random.poisson(xg_rate_away * remaining_minutes, n_sims)
        goals_home_sofar = sum(
            1
            for e in events
            if e.get("type") == "shot" and e.get("is_goal") and e.get("team") == "home"
        )
        goals_away_sofar = sum(
            1
            for e in events
            if e.get("type") == "shot" and e.get("is_goal") and e.get("team") == "away"
        )
        final_home = sim_goals_home + goals_home_sofar
        final_away = sim_goals_away + goals_away_sofar
        counts: dict[str, int] = defaultdict(int)
        for h, a in zip(final_home, final_away, strict=False):
            counts[f"{int(h)}-{int(a)}"] += 1
        scorelines: dict[str, float] = {}
        for key, count in counts.items():
            prob = count / n_sims
            if prob >= 0.001:
                scorelines[key] = round(prob, 4)
        total_prob = sum(scorelines.values())
        if total_prob > 0:
            scorelines = {k: round(v / total_prob, 4) for k, v in scorelines.items()}
        return {
            "scorelines": dict(sorted(scorelines.items(), key=lambda x: -x[1])),
            "n_sims": n_sims,
            "remaining_minutes": round(remaining_minutes, 1),
            "xg_rate_home": round(xg_rate_home, 3),
            "xg_rate_away": round(xg_rate_away, 3),
            "goals_current_home": goals_home_sofar,
            "goals_current_away": goals_away_sofar,
        }

    def compute_match_outcome_probs(self, scoreline_probs: dict[str, float]) -> dict[str, float]:
        win_h = 0.0
        draw = 0.0
        win_a = 0.0
        for key, prob in scoreline_probs.items():
            parts = key.split("-")
            if len(parts) != 2:
                continue
            try:
                h = int(parts[0])
                a = int(parts[1])
            except ValueError:
                continue
            if h > a:
                win_h += prob
            elif h < a:
                win_a += prob
            else:
                draw += prob
        return {"win_home": round(win_h, 4), "draw": round(draw, 4), "win_away": round(win_a, 4)}

    def compute_scoreline_entropy(self, scoreline_probs: dict[str, float]) -> float:
        entropy = 0.0
        for prob in scoreline_probs.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return round(entropy, 4)
