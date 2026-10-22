"""xG and xT computation mixin."""

from __future__ import annotations

from kawkab.core.xg_model import compute_xg_trained_from_dict


class XgXtMixin:
    def compute_xg_simple(self, events, pitch_length_m=105.0, pitch_width_m=68.0):
        """Match-level xG from raw event dicts via the ACTIVE xG model.

        This replaces the old inline exp/cos heuristic with the trained
        StatsBomb model (same weights the Pro/xA paths use), so every xG
        number in the app comes from one calibrated model.

        Angle convention: ``angle_to_goal_deg`` from the CV pipeline is
        DEVIATION-from-central (0° = central) — the trained model's input
        convention. StatsBomb-imported events carry OPENING angle under
        ``angle_deg`` and are converted at import time.

        Honest-absent: shots with no spatial metadata get xg 0.0 and an
        xg_available=False marker instead of the old fabricated
        distance=18/angle=30 estimate (the old defaults produced a
        plausible-looking but invented number).
        """
        home_xg = 0.0
        away_xg = 0.0
        shot_details = []

        for event in events:
            if event.get("type") != "shot":
                continue

            timestamp = event.get("timestamp", 0)
            team = event.get("team", "home")
            metadata = event.get("metadata", {})

            distance_m = metadata.get("distance_to_goal_m")
            angle_deg = metadata.get("angle_to_goal_deg")
            if distance_m is not None and angle_deg is not None:
                xg = compute_xg_trained_from_dict(
                    {
                        "type": "shot",
                        "distance_m": float(distance_m),
                        "angle_deg": float(angle_deg),
                    }
                )
                xg = max(0.0, min(1.0, xg))
                xg_available = True
            else:
                xg = 0.0
                xg_available = False

            if team == "home":
                home_xg += xg
            else:
                away_xg += xg

            shot_details.append(
                {
                    "timestamp": timestamp,
                    "team": team,
                    "track_id": event.get("track_id") or event.get("player_id", 0),
                    "start_x": event.get("start_x", 0.0),
                    "start_y": event.get("start_y", 34.0),
                    "distance_m": distance_m,
                    "angle_deg": angle_deg,
                    "xg": round(xg, 3),
                    "xg_available": xg_available,
                    "on_target": event.get("on_target", False),
                    "is_goal": event.get("is_goal", False),
                }
            )

        return {
            "home": round(home_xg, 3),
            "away": round(away_xg, 3),
            "shot_details": shot_details,
        }

    def compute_xt_simple(self, events, pitch_length_m=105.0, pitch_width_m=68.0):
        xt_zones = [
            [0.01, 0.02, 0.03, 0.04],
            [0.02, 0.05, 0.08, 0.12],
            [0.03, 0.08, 0.15, 0.25],
            [0.04, 0.12, 0.25, 0.50],
        ]

        def get_xt_value(x_pct, y_pct):
            col = min(3, int(x_pct * 4))
            row = min(3, int(y_pct * 4))
            return xt_zones[row][col]

        home_xt = 0.0
        away_xt = 0.0

        for event in events:
            if event.get("type") != "pass":
                continue
            if not event.get("completed"):
                continue

            team = event.get("team", "home")
            metadata = event.get("metadata", {})
            start_x = metadata.get("start_x_pct", 0.5)
            end_x = metadata.get("end_x_pct", 0.6)
            start_xt = get_xt_value(start_x, 0.5)
            end_xt = get_xt_value(end_x, 0.5)
            xt_delta = max(0.0, end_xt - start_xt)

            if team == "home":
                home_xt += xt_delta
            else:
                away_xt += xt_delta

        return {
            "home": round(home_xt, 3),
            "away": round(away_xt, 3),
        }
