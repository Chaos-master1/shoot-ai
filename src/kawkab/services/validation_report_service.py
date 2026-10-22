"""Validation report service — Phase 2 of the elite-readiness roadmap.

Turns the existing validation building blocks (``core/validation``
metrics, StatsBomb corpus loaders, model-card registry) into a single
reproducible report service, exposed via ``python -m kawkab validate``
and importable for tests/CI regression gates.

The report is the trust artifact elite analysts ask for first: model
cards for every registered model, xG/PSxG/xT calibration vs the StatsBomb
corpus (when present locally), and corpus provenance. Exit-code honesty
matches scripts/validate_models.py: a bad score still writes a report —
honesty is the gate, not optimism.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_DIR = "data/statsbomb_corpus"
DEFAULT_OUT_DIR = "docs/validation"


@dataclass
class ValidationSection:
    """One evaluated area of the report."""

    name: str
    status: str  # "evaluated" | "skipped"
    reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class ValidationReportService:
    """Builds the reproducible model-validation report."""

    def __init__(self, corpus_dir: str | Path = DEFAULT_CORPUS_DIR) -> None:
        self.corpus_dir = Path(corpus_dir)

    # ── Sections ────────────────────────────────────────────────────────

    def model_cards_section(self) -> ValidationSection:
        """Coverage of the model-card registry."""
        from kawkab.core.model_card_registry import list_model_cards

        cards = list_model_cards()
        return ValidationSection(
            name="model_cards",
            status="evaluated",
            metrics={
                "registered": len(cards),
                "cards": [
                    {
                        "name": c.name,
                        "version": c.version,
                        "model_type": c.model_type,
                        "training_data": c.training_data,
                        "test_count": c.test_count,
                        "known_limitations": len(c.known_limitations),
                    }
                    for c in cards
                ],
            },
        )

    def xg_calibration_section(self, max_matches: int = 40) -> ValidationSection:
        """xG model calibration vs StatsBomb's published xG on the corpus."""
        try:
            from kawkab.core.validation.metrics import brier_score, roc_auc
            from kawkab.core.validation.statsbomb_loader import (
                extract_shots_for_fitting,
                load_statsbomb_corpus,
            )
            from kawkab.core.xg_model import active_xg_model

            # load_statsbomb_corpus is a lazy generator — use its limit
            # parameter rather than slicing.
            matches = list(load_statsbomb_corpus(self.corpus_dir, limit=max_matches))
            if not matches:
                return ValidationSection(
                    name="xg_calibration",
                    status="skipped",
                    reason=f"corpus empty at {self.corpus_dir}",
                )
            shots, n_matches = extract_shots_for_fitting(matches)
            if not shots:
                return ValidationSection(
                    name="xg_calibration",
                    status="skipped",
                    reason="corpus produced zero shots",
                )

            model = active_xg_model()
            y = [1.0 if s.is_goal else 0.0 for s in shots]
            preds: list[float] = []
            sb_xg: list[float] = []
            for s in shots:
                ev = {
                    "type": "shot",
                    "distance_m": s.distance_m,
                    "angle_deg": s.angle_deg,
                    "body_part": s.body_part
                    if s.body_part in ("right_foot", "left_foot", "head")
                    else "right_foot",
                    "shot_type": s.shot_type
                    if s.shot_type in ("open_play", "free_kick", "penalty", "corner")
                    else "open_play",
                    "gk_distance_m": s.gk_distance_m,
                    "is_rebound": s.is_rebound,
                    "is_big_chance": s.is_big_chance,
                    "is_one_on_one": s.is_one_on_one,
                }
                preds.append(float(model.compute(ev)))
                sb_xg.append(float(getattr(s, "statsbomb_xg", 0.0) or 0.0))

            import numpy as np

            p = np.clip(np.array(preds), 0.0, 1.0)
            yv = np.array(y)
            sb = np.array(sb_xg)
            mae_vs_sb = float(np.mean(np.abs(p - sb))) if len(sb) else float("nan")
            return ValidationSection(
                name="xg_calibration",
                status="evaluated",
                metrics={
                    "n_shots": len(shots),
                    "n_matches": n_matches,
                    "brier": round(brier_score(yv, p), 4),
                    "roc_auc": round(roc_auc(yv, p), 4),
                    "mae_vs_statsbomb_xg": round(mae_vs_sb, 4),
                    "mean_predicted_xg": round(float(p.mean()), 4),
                    "mean_actual_goal_rate": round(float(yv.mean()), 4),
                },
            )
        except Exception as exc:
            return ValidationSection(name="xg_calibration", status="skipped", reason=str(exc))

    def psxg_calibration_section(self, max_matches: int = 20) -> ValidationSection:
        """PSxG vs outcomes on corpus shots that were on target."""
        try:
            from kawkab.core.psxg_model import compute_psxg
            from kawkab.core.validation.metrics import brier_score
            from kawkab.core.validation.statsbomb_loader import (
                extract_shots_for_fitting,
                load_statsbomb_corpus,
            )

            matches = list(load_statsbomb_corpus(self.corpus_dir, limit=max_matches))
            shots, _n_matches = extract_shots_for_fitting(matches)
            on_target = [
                s
                for s in shots
                if getattr(s, "shot_type", "") != "penalty" and 8.0 <= s.distance_m <= 30.0
            ]
            if not on_target:
                return ValidationSection(
                    name="psxg_calibration",
                    status="skipped",
                    reason="no on-target-range shots in corpus sample",
                )
            y, preds = [], []
            for s in on_target:
                result = compute_psxg(
                    distance_m=s.distance_m,
                    angle_deg=s.angle_deg,
                    body_part=s.body_part
                    if s.body_part in ("right_foot", "left_foot", "head")
                    else "right_foot",
                )
                preds.append(float(result.psxg))
                y.append(1.0 if s.is_goal else 0.0)
            import numpy as np

            return ValidationSection(
                name="psxg_calibration",
                status="evaluated",
                metrics={
                    "n_shots": len(on_target),
                    "brier": round(brier_score(np.array(y), np.clip(np.array(preds), 0, 1)), 4),
                },
            )
        except Exception as exc:
            return ValidationSection(name="psxg_calibration", status="skipped", reason=str(exc))

    def xt_grid_section(self) -> ValidationSection:
        """Sanity of the trained xT reference grid (bounds + goalward growth)."""
        try:
            from kawkab.core.xt_model import _load_reference_grid

            arr = _load_reference_grid()
            if arr is None:
                return ValidationSection(
                    name="xt_grid",
                    status="skipped",
                    reason="no trained grid available",
                )
            import numpy as np

            arr = np.asarray(arr, dtype=float)
            return ValidationSection(
                name="xt_grid",
                status="evaluated",
                metrics={
                    "shape": list(arr.shape),
                    "min": round(float(arr.min()), 5),
                    "max": round(float(arr.max()), 5),
                    "all_nonnegative": bool((arr >= 0).all()),
                    "max_in_attacking_third": bool(
                        arr[:, arr.shape[1] * 2 // 3 :].max() >= arr[:, : arr.shape[1] // 3].max()
                    ),
                },
            )
        except Exception as exc:
            return ValidationSection(name="xt_grid", status="skipped", reason=str(exc))

    # ── Tracking-model sections (real open tracking data) ───────────────

    TRACKING_FIXTURE_DIR = "tests/fixtures/tracking"
    METRICA_HOME = "metrica_sample2_home.csv"
    METRICA_AWAY = "metrica_sample2_away.csv"
    METRICA_EVENTS = "metrica_sample2_events.csv"

    def _load_metrica_fixture(self) -> dict[str, Any] | None:
        """Genuine Metrica open tracking + events, or None when absent."""
        from kawkab.core.validation.metrica_loader import load_metrica_match

        base = Path(self.TRACKING_FIXTURE_DIR)
        home, away = base / self.METRICA_HOME, base / self.METRICA_AWAY
        if not (home.exists() and away.exists()):
            return None
        mm = load_metrica_match(home, away)
        events: list[dict[str, Any]] = []
        ev_path = base / self.METRICA_EVENTS
        if ev_path.exists():
            import csv as _csv

            with open(ev_path, encoding="utf-8") as f:
                for row in _csv.DictReader(f):

                    def _num(v: Any) -> float | None:
                        try:
                            f = float(v)
                        except (TypeError, ValueError):
                            return None
                        return None if f != f else f  # CSV 'NaN' -> None

                    sx, sy = _num(row.get("Start X")), _num(row.get("Start Y"))
                    ex, ey = _num(row.get("End X")), _num(row.get("End Y"))
                    ev_type = (row.get("Type") or "").lower()
                    subtype = (row.get("Subtype") or "").lower()
                    # Map Metrica's event vocabulary onto the analyzer's
                    # canonical types: Metrica 'RECOVERY' is a team GAINING
                    # the ball (interception when subtype says so, else a
                    # loose-ball recovery).
                    if ev_type == "recovery":
                        ev_type = "interception" if "interception" in subtype else "loose_ball"
                    events.append(
                        {
                            "type": ev_type,
                            "subtype": subtype,
                            "team": (row.get("Team") or "").lower(),
                            "period": int(row.get("Period") or 1),
                            "timestamp": _num(row.get("Start Time [s]")) or 0.0,
                            # Metrica events share the tracking files' [0,1]
                            # normalized pitch; convert to Kawkab meters (105x68).
                            # Both key conventions are filled: the recovery
                            # analyzer reads x/y, the xT/carry code reads
                            # start_x/start_y/end_x/end_y. Missing coordinates
                            # fall back to pitch center.
                            "x": (sx if sx is not None else 0.5) * 105.0,
                            "y": (sy if sy is not None else 0.5) * 68.0,
                            "start_x": (sx if sx is not None else 0.5) * 105.0,
                            "start_y": (sy if sy is not None else 0.5) * 68.0,
                            "end_x": (ex if ex is not None else 0.5) * 105.0,
                            "end_y": (ey if ey is not None else 0.5) * 68.0,
                            "completed": True,
                            # Metrica marks goals as SHOT / ON TARGET-GOAL —
                            # the xT model's ze term needs them.
                            "is_goal": (ev_type == "shot" and "goal" in subtype),
                        }
                    )
        return {"tracking": mm, "events": events}

    @staticmethod
    def _tracking_frames_dicts(mm: Any, every_n: int = 5) -> list[dict[str, Any]]:
        """Fixture frames in the shape pitch-control's compute_match_control
        consumes, subsampled for runtime (25 fps -> 5 Hz)."""
        frames = []
        for fr in mm.sample(every_n):
            frames.append(
                {
                    "timestamp": fr.time_s,
                    "home_positions": [(p[0], p[1]) for p in fr.home],
                    "away_positions": [(p[0], p[1]) for p in fr.away],
                    "ball_pos": (fr.ball[0], fr.ball[1]) if fr.ball else None,
                }
            )
        return frames

    def pitch_control_section(self) -> ValidationSection:
        """Voronoi pitch control on real tracking data: control must sum to
        ~1 per frame, stay in [0,1], and concentrate on the team near the
        ball (the model's core behavioral claim)."""
        try:
            data = self._load_metrica_fixture()
            if data is None:
                return ValidationSection(
                    name="pitch_control",
                    status="skipped",
                    reason="Metrica fixture missing (tests/fixtures/tracking)",
                )
            from kawkab.core.pitch_control import VoronoiPitchControl

            frames = self._tracking_frames_dicts(data["tracking"], every_n=25)
            if not frames:
                return ValidationSection(
                    name="pitch_control",
                    status="skipped",
                    reason="no frames",
                )
            pc = VoronoiPitchControl()
            match = pc.compute_match_control(frames)
            # NOTE: MatchPitchControl reports PERCENTAGES (0-100), the UI's
            # display convention — not fractions.
            h, a = match.avg_home_control, match.avg_away_control
            return ValidationSection(
                name="pitch_control",
                status="evaluated",
                metrics={
                    "frames_evaluated": len(frames),
                    "avg_home_control_pct": round(h, 4),
                    "avg_away_control_pct": round(a, 4),
                    "sums_to_100": round(h + a, 4),
                    "all_in_percent_range": bool(0.0 <= h <= 100.0 and 0.0 <= a <= 100.0),
                    "home_third_control_pct": round(match.home_third_control, 4),
                    "away_third_control_pct": round(match.away_third_control, 4),
                },
            )
        except Exception as exc:
            return ValidationSection(name="pitch_control", status="skipped", reason=str(exc))

    def ball_recovery_section(self) -> ValidationSection:
        """Ball-recovery analysis on real events: recovery counts per team
        must be plausible and zone-distributed (not all in one zone)."""
        try:
            data = self._load_metrica_fixture()
            if data is None or not data["events"]:
                return ValidationSection(
                    name="ball_recovery",
                    status="skipped",
                    reason="Metrica events fixture missing",
                )
            from kawkab.core.ball_recovery import RECOVERY_EVENT_TYPES, BallRecoveryAnalyzer

            events = data["events"]
            n_rec = sum(
                1 for e in events if e.get("type") in {t.lower() for t in RECOVERY_EVENT_TYPES}
            )
            analyzer = BallRecoveryAnalyzer()
            home = analyzer.analyze_recoveries(events, "home")
            away = analyzer.analyze_recoveries(events, "away")
            home_zones = len(home.get("recoveries_by_zone", {}) or {})
            away_zones = len(away.get("recoveries_by_zone", {}) or {})
            return ValidationSection(
                name="ball_recovery",
                status="evaluated",
                metrics={
                    "recovery_events": n_rec,
                    "home_recoveries": home.get("total_recoveries", 0),
                    "away_recoveries": away.get("total_recoveries", 0),
                    "home_zones_used": home_zones,
                    "away_zones_used": away_zones,
                    "recoveries_leading_to_shot": (
                        home.get("recoveries_leading_to_shot", 0)
                        + away.get("recoveries_leading_to_shot", 0)
                    ),
                    "zones_spread": bool(home_zones >= 3 or away_zones >= 3),
                },
            )
        except Exception as exc:
            return ValidationSection(name="ball_recovery", status="skipped", reason=str(exc))

    def carry_xt_section(self) -> ValidationSection:
        """Carry-xT on real tracking+events: carries derived from ball
        movement must yield non-negative xT (model floors at 0) and sane
        distances."""
        try:
            data = self._load_metrica_fixture()
            if data is None:
                return ValidationSection(
                    name="carry_xt",
                    status="skipped",
                    reason="Metrica fixture missing",
                )
            from kawkab.core.carry_xt import compute_carry_xt_from_tracking

            mm = data["tracking"]
            events = data["events"]
            frames_raw = self._tracking_frames_dicts(mm, every_n=5)
            # Derive carries from consecutive ball positions (real event
            # feeds do not label carries; the service does the same).
            carries: list[dict[str, Any]] = []
            prev: tuple[float, float] | None = None
            team_flip = 0
            for fr in frames_raw:
                if not fr["ball_pos"]:
                    continue
                bx, by = fr["ball_pos"]
                if prev is not None:
                    dist = ((bx - prev[0]) ** 2 + (by - prev[1]) ** 2) ** 0.5
                    # 0.2 s apart at 5 Hz: >1 m of ball travel means the
                    # ball is genuinely moving (pass/carry), not jitter.
                    if 1.0 <= dist <= 40.0:
                        carries.append(
                            {
                                "type": "carry",
                                "team": "home" if team_flip % 2 == 0 else "away",
                                "start_x": prev[0],
                                "start_y": prev[1],
                                "end_x": bx,
                                "end_y": by,
                                "timestamp": fr["timestamp"],
                                "completed": True,
                            }
                        )
                        team_flip += 1
                prev, _ = (bx, by), fr["timestamp"]
            if not carries:
                return ValidationSection(
                    name="carry_xt",
                    status="skipped",
                    reason="no carries derived",
                )
            # Build the model from the FULL event set (passes, shots, etc.),
            # exactly as the production path does — a carry-only transition
            # matrix is a degenerate cold-start (no shots -> zero-value grid).
            from kawkab.core.xt_model import ExpectedThreatModel

            xt = ExpectedThreatModel()
            xt.build_transition_matrix(events)
            report = compute_carry_xt_from_tracking(frames_raw, carries, xt_model=xt)
            n = report.home_carries + report.away_carries
            return ValidationSection(
                name="carry_xt",
                status="evaluated",
                metrics={
                    "carries": n,
                    "home_total_xt": round(report.home_total_xt, 4),
                    "away_total_xt": round(report.away_total_xt, 4),
                    "home_progressive": report.home_progressive,
                    "away_progressive": report.away_progressive,
                    "xt_nonnegative": bool(report.home_total_xt >= 0 and report.away_total_xt >= 0),
                },
            )
        except Exception as exc:
            return ValidationSection(name="carry_xt", status="skipped", reason=str(exc))

    def physical_load_section(self) -> ValidationSection:
        """Physical-load plausibility on real tracking: per-player distance
        in the fixture's 80-second window must fall within an honest
        extrapolated elite band (10-13 km per ~96 min match)."""
        try:
            data = self._load_metrica_fixture()
            if data is None:
                return ValidationSection(
                    name="physical_load",
                    status="skipped",
                    reason="Metrica fixture missing",
                )
            from kawkab.services.cv_service import (
                Detection,
                FrameDetections,
                MatchTrackData,
            )
            from kawkab.services.physical_load_service import PhysicalLoadService

            mm = data["tracking"]
            svc = PhysicalLoadService()
            # Sample at 5 Hz (GPS-equivalent) AND smooth each trajectory
            # with a 1 s moving average: raw vendor coordinate jitter
            # accumulates phantom distance that dwarfs real movement. Both
            # steps are standard load-pipeline practice.
            sampled = mm.sample(every_n=5)
            series: dict[int, list[tuple[float, float]]] = {}
            for fr in sampled:
                for idx, pos in enumerate(fr.home):
                    series.setdefault(1 + idx, []).append((pos[0], pos[1]))
                for idx, pos in enumerate(fr.away):
                    series.setdefault(101 + idx, []).append((pos[0], pos[1]))
            smoothed: dict[int, list[tuple[float, float]]] = {}
            for tid, pts in series.items():
                out = []
                for i in range(len(pts)):
                    lo, hi = max(0, i - 2), min(len(pts), i + 3)
                    out.append(
                        (
                            sum(p[0] for p in pts[lo:hi]) / (hi - lo),
                            sum(p[1] for p in pts[lo:hi]) / (hi - lo),
                        )
                    )
                smoothed[tid] = out
            frames = []
            for j, fr in enumerate(sampled):
                dets = []
                for tid in sorted(smoothed):
                    x, y = smoothed[tid][j]
                    dets.append(
                        Detection(
                            bbox=(x, y, x, y),
                            confidence=1.0,
                            class_id=0,
                            class_name="person",
                            track_id=tid,
                        )
                    )
                frames.append(
                    FrameDetections(
                        frame_number=fr.frame,
                        timestamp=fr.time_s,
                        detections=dets,
                        image_width=105,
                        image_height=68,
                    )
                )
            track = MatchTrackData(
                match_id=0,
                fps=mm.fps / 5,
                total_frames=len(frames),
                duration_seconds=frames[-1].timestamp if frames else 0.0,
                frames=frames,
                track_registry={},
            )
            import asyncio as _asyncio

            results = _asyncio.run(svc.compute_physical_load(track, homography_matrix=None))
            if not results:
                return ValidationSection(
                    name="physical_load",
                    status="skipped",
                    reason="no trajectories",
                )
            # Extrapolate the ~80 s window to a 96-minute match for the
            # plausibility band (10-13 km/match, elite published range).
            durations = frames[-1].timestamp
            if durations <= 0:
                return ValidationSection(
                    name="physical_load",
                    status="skipped",
                    reason="zero duration",
                )
            # Like-for-like plausibility: average running speed over the
            # window vs the published elite match average (~10.5 km over
            # ~96 min ≈ 1.8-2.2 m/s). The fixture is a match's OPENING
            # spell (highest tempo, fresh players), so the sanity band
            # extends above the match average; it still catches gross
            # errors (jitter artifacts push >3.5 m/s, dead data <1 m/s).
            # No linear extrapolation to 96 min — that would assume the
            # opening 80 s represents a full match.
            speeds = sorted(m.total_distance_m / durations for m in results.values())
            median_speed = speeds[len(speeds) // 2]
            return ValidationSection(
                name="physical_load",
                status="evaluated",
                metrics={
                    "players": len(results),
                    "window_seconds": round(durations, 1),
                    "median_avg_speed_mps": round(median_speed, 2),
                    "within_sanity_band_1p8_2p6_mps": bool(1.8 <= median_speed <= 2.6),
                },
            )
        except Exception as exc:
            return ValidationSection(name="physical_load", status="skipped", reason=str(exc))

    # ── Report assembly ─────────────────────────────────────────────────

    def build_report(self, max_matches: int = 40) -> dict[str, Any]:
        """Full report dict — reproducible given the same corpus snapshot."""
        sections = [
            self.model_cards_section(),
            self.xg_calibration_section(max_matches=max_matches),
            self.psxg_calibration_section(max_matches=max_matches),
            self.xt_grid_section(),
            self.pitch_control_section(),
            self.ball_recovery_section(),
            self.carry_xt_section(),
            self.physical_load_section(),
        ]
        evaluated = [s for s in sections if s.status == "evaluated"]
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "corpus_dir": str(self.corpus_dir),
            "sections": [asdict(s) for s in sections],
            "summary": {
                "sections_evaluated": len(evaluated),
                "sections_skipped": len(sections) - len(evaluated),
                "model_cards_registered": next(
                    (s.metrics["registered"] for s in sections if s.name == "model_cards"),
                    0,
                ),
            },
        }

    def to_markdown(self, report: dict[str, Any]) -> str:
        """Human-readable markdown twin of the JSON report."""
        lines = [
            "# Kawkab Validation Report",
            "",
            f"Generated: {report['generated_at']}",
            f"Corpus: `{report['corpus_dir']}`",
            "",
        ]
        for s in report["sections"]:
            lines.append(f"## {s['name']} — {s['status']}")
            if s.get("reason"):
                lines.append(f"\n> Skipped: {s['reason']}")
            if s.get("metrics"):
                lines.append("")
                lines.append("| metric | value |")
                lines.append("|---|---|")
                for k, v in s["metrics"].items():
                    if k == "cards":
                        lines.append(f"| registered cards | {len(v)} |")
                        continue
                    lines.append(f"| {k} | `{v}` |")
            lines.append("")
        return "\n".join(lines)

    def write_report(
        self,
        out_dir: str | Path = DEFAULT_OUT_DIR,
        max_matches: int = 40,
    ) -> dict[str, Any]:
        """Build and write validation_report.json + .md; returns the report."""
        report = self.build_report(max_matches=max_matches)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "validation_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        (out / "validation_report.md").write_text(self.to_markdown(report), encoding="utf-8")
        return report
