"""Independent accuracy audit framework for football analytics models.

Usage:
    python scripts/accuracy_audit.py --match-id 42 --ground-truth data/gt/match_42.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from kawkab.core.mot_metrics import compute_mot_metrics

logger.remove()
logger.add(sys.stderr, format="<level>{level:8s}</level> | <message>", level="INFO")


@dataclass
class AuditResult:
    overall_grade: str = "F"
    overall_score: float = 0.0
    event_accuracy: dict[str, Any] = field(default_factory=dict)
    possession_error: dict[str, Any] | None = None
    tracking_metrics: dict[str, Any] | None = None
    xg_metrics: dict[str, Any] | None = None
    improvement_suggestions: list[str] = field(default_factory=list)
    per_category_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_grade": self.overall_grade,
            "overall_score": self.overall_score,
            "event_accuracy": self.event_accuracy,
            "possession_error": self.possession_error,
            "tracking_metrics": self.tracking_metrics,
            "xg_metrics": self.xg_metrics,
            "improvement_suggestions": self.improvement_suggestions,
            "per_category_scores": self.per_category_scores,
        }


def _grade_from_score(score: float) -> str:
    if score >= 0.95:
        return "A"
    if score >= 0.85:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"


def _brier_score(probs: list[float], outcomes: list[bool]) -> float:
    if not probs or len(probs) != len(outcomes):
        return 0.0
    return float(np.mean([(p - o) ** 2 for p, o in zip(probs, outcomes, strict=False)]))


def _log_loss(probs: list[float], outcomes: list[bool], eps: float = 1e-15) -> float:
    if not probs or len(probs) != len(outcomes):
        return 0.0
    probs = np.clip(probs, eps, 1 - eps)
    return float(
        -np.mean(
            [
                o * math.log(p) + (1 - o) * math.log(1 - p)
                for o, p in zip(outcomes, probs, strict=False)
            ]
        )
    )


def _calibration_error(probs: list[float], outcomes: list[bool], n_bins: int = 10) -> float:
    if not probs or len(probs) != len(outcomes):
        return 0.0
    probs = np.array(probs)
    outcomes = np.array(outcomes, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    error = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_conf = float(np.mean(probs[mask]))
            bin_acc = float(np.mean(outcomes[mask]))
            error += abs(bin_conf - bin_acc) * np.sum(mask)
    return error / max(len(probs), 1)


class AccuracyAudit:
    """Independent accuracy audit for football analytics models.

    Compares computed analytics against ground truth data and
    produces a scored report with per-category grades.

    Args:
        match_id: Match identifier.
        ground_truth_path: Optional path to load ground truth from.
    """

    def __init__(self, match_id: int | str, ground_truth_path: str | None = None):
        self.match_id = match_id
        self.ground_truth: dict[str, Any] = {}
        self.computed: dict[str, Any] = {}
        self._loaded = False

        if ground_truth_path:
            self.load_ground_truth(ground_truth_path)

    def load_ground_truth(self, path: str) -> bool:
        """Load ground truth data from a StatsBomb JSON, Metrica CSV, or manual JSON.

        Supported formats (auto-detected by extension and structure):
        - .json with "events" key:          StatsBomb-like or manual
        - .json with "tracks"/"tracking":   tracking ground truth

        Args:
            path: Path to ground truth file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        p = Path(path)
        if not p.exists():
            logger.error("Ground truth path does not exist: {}", path)
            return False

        try:
            raw: Any = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load ground truth: {}", exc)
            return False

        if isinstance(raw, dict):
            self.ground_truth = raw
        elif isinstance(raw, list):
            self.ground_truth = {"events": raw}
        else:
            logger.error("Unsupported ground truth format in {}", path)
            return False

        events = self.ground_truth.get("events", [])
        tracks = self.ground_truth.get("tracks", self.ground_truth.get("tracking", []))
        logger.info(
            "Loaded ground truth: {} events, {} tracking frames",
            len(events),
            len(tracks) if isinstance(tracks, list) else 0,
        )
        self._loaded = True
        return True

    def compare_events(
        self,
        computed_events: list[dict[str, Any]],
        tolerance_s: float = 2.0,
    ) -> dict[str, Any]:
        """Compare computed events against ground truth events.

        Computes precision, recall, and F1 per event type within a
        temporal tolerance window.

        Args:
            computed_events: List of computed event dicts (must have "type" and "timestamp").
            tolerance_s: Time tolerance in seconds for matching.

        Returns:
            Dict with per-type and overall metrics.
        """
        gt_events = self.ground_truth.get("events", [])
        if not gt_events:
            logger.warning("No ground truth events to compare against")
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "per_type": {},
                "total_gt": 0,
                "total_computed": 0,
            }

        gt_by_type: dict[str, list[float]] = {}
        for ev in gt_events:
            t = ev.get("type", "unknown")
            ts = ev.get("timestamp", 0.0)
            gt_by_type.setdefault(t, []).append(ts)

        comp_by_type: dict[str, list[float]] = {}
        for ev in computed_events:
            t = ev.get("type", "unknown")
            ts = ev.get("timestamp", 0.0)
            comp_by_type.setdefault(t, []).append(ts)

        all_types = set(gt_by_type.keys()) | set(comp_by_type.keys())
        per_type: dict[str, dict[str, float]] = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for etype in sorted(all_types):
            gt_times = sorted(gt_by_type.get(etype, []))
            comp_times = sorted(comp_by_type.get(etype, []))

            matched_gt = set()
            tp = 0
            for _ci, ct in enumerate(comp_times):
                best_d = tolerance_s
                best_gi = None
                for gi, gt_t in enumerate(gt_times):
                    if gi in matched_gt:
                        continue
                    d = abs(ct - gt_t)
                    if d < best_d:
                        best_d = d
                        best_gi = gi
                if best_gi is not None:
                    tp += 1
                    matched_gt.add(best_gi)

            fp = len(comp_times) - tp
            fn = len(gt_times) - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)

            per_type[etype] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
            total_tp += tp
            total_fp += fp
            total_fn += fn

        overall_precision = total_tp / max(total_tp + total_fp, 1)
        overall_recall = total_tp / max(total_tp + total_fn, 1)
        overall_f1 = (
            2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-9)
        )

        result = {
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1": round(overall_f1, 4),
            "per_type": per_type,
            "total_gt": len(gt_events),
            "total_computed": len(computed_events),
        }
        logger.info(
            "Event comparison: P={:.3f} R={:.3f} F1={:.3f} ({}/{} events)",
            overall_precision,
            overall_recall,
            overall_f1,
            total_tp,
            len(gt_events),
        )
        return result

    def compare_possession(
        self,
        computed_pct: float,
        ground_truth_pct: float,
    ) -> dict[str, Any]:
        """Compare computed possession percentage against ground truth.

        Args:
            computed_pct: Computed home possession percentage.
            ground_truth_pct: Ground truth home possession percentage.

        Returns:
            Dict with absolute error and deviation category.
        """
        error = abs(computed_pct - ground_truth_pct)
        if error <= 2.0:
            category = "excellent"
        elif error <= 5.0:
            category = "acceptable"
        else:
            category = "poor"

        result = {
            "computed": round(computed_pct, 1),
            "ground_truth": round(ground_truth_pct, 1),
            "absolute_error": round(error, 1),
            "category": category,
        }
        logger.info(
            "Possession comparison: |{} - {}| = {} ({})",
            computed_pct,
            ground_truth_pct,
            error,
            category,
        )
        return result

    def compare_tracking(
        self,
        computed_tracks: dict[int, list[tuple[int, float, float]]],
        gt_tracks: dict[int, list[tuple[int, float, float]]] | None = None,
        fp_threshold: float = 20.0,
    ) -> dict[str, Any]:
        """Compare computed tracking against ground truth using CLEAR MOT metrics.

        Args:
            computed_tracks: Computed tracks {track_id: [(frame, x, y), ...]}.
            gt_tracks: Ground truth tracks. If None, uses tracks from loaded ground truth.
            fp_threshold: Distance threshold in metres/pixels for a match.

        Returns:
            Dict with MOTA, MOTP, IDF1, etc.
        """
        if gt_tracks is None:
            gt_tracks = self.ground_truth.get("tracks", self.ground_truth.get("tracking", {}))

        if not gt_tracks or not computed_tracks:
            logger.warning("Insufficient tracking data for comparison")
            return {}

        metrics = compute_mot_metrics(computed_tracks, gt_tracks, fp_threshold=fp_threshold)
        logger.info(
            "Tracking comparison: MOTA={:.4f} MOTP={:.2f} IDF1={:.4f}",
            metrics.get("mota", 0),
            metrics.get("motp", 0),
            metrics.get("idf1", 0),
        )
        return metrics

    def compare_xg(
        self,
        computed_shots: list[dict[str, Any]],
        gt_shots: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compare computed xG values against ground truth shot data.

        Computes log-loss, Brier score, and calibration error.

        Args:
            computed_shots: List of computed shot dicts with keys "xg" and "is_goal".
            gt_shots: Ground truth shots. If None, uses events from loaded ground truth.

        Returns:
            Dict with xG accuracy metrics.
        """
        if gt_shots is None:
            gt_shots = [e for e in self.ground_truth.get("events", []) if e.get("type") == "shot"]

        if not gt_shots or not computed_shots:
            logger.warning("Insufficient shot data for xG comparison")
            return {}

        comp_map: dict[float, float] = {}
        for s in computed_shots:
            key = (s.get("timestamp", 0), s.get("team", ""))
            comp_map[hash(str(key))] = s.get("xg", 0.0)

        paired_probs: list[float] = []
        paired_outcomes: list[bool] = []
        for gs in gt_shots:
            key = (gs.get("timestamp", 0), gs.get("team", ""))
            hk = hash(str(key))
            if hk in comp_map:
                paired_probs.append(comp_map[hk])
                paired_outcomes.append(gs.get("is_goal", False))
            else:
                paired_probs.append(gs.get("xg", 0.0))
                paired_outcomes.append(gs.get("is_goal", False))

        if not paired_probs:
            return {}

        n_shots = len(paired_probs)
        n_goals = sum(paired_outcomes)
        avg_xg = float(np.mean(paired_probs))
        brier = _brier_score(paired_probs, paired_outcomes)
        logloss = _log_loss(paired_probs, paired_outcomes)
        cal_error = _calibration_error(paired_probs, paired_outcomes)

        result = {
            "n_shots": n_shots,
            "n_goals": n_goals,
            "avg_xg": round(avg_xg, 4),
            "avg_goal_rate": round(n_goals / max(n_shots, 1), 4),
            "brier_score": round(brier, 4),
            "log_loss": round(logloss, 4),
            "calibration_error": round(cal_error, 4),
        }
        logger.info(
            "xG comparison: Brier={:.4f} LogLoss={:.4f} CalErr={:.4f} ({} shots)",
            brier,
            logloss,
            cal_error,
            n_shots,
        )
        return result

    def generate_report(self) -> dict[str, Any]:
        """Generate a full audit report.

        Uses data stored in self.computed and self.ground_truth.
        Returns a dict with per-category scores, overall grade (A-F),
        and improvement suggestions.

        Returns:
            Audit report dict.
        """
        if not self._loaded and not self.ground_truth:
            logger.warning("No ground truth loaded — report will be empty")
            report = AuditResult(overall_grade="N/A")
            return report.to_dict()

        result = AuditResult()
        category_scores: dict[str, float] = {}
        suggestions: list[str] = []

        # Event accuracy
        computed_events = self.computed.get("events", [])
        if computed_events:
            ev_result = self.compare_events(computed_events)
            result.event_accuracy = ev_result
            f1 = ev_result.get("f1", 0.0)
            category_scores["event_accuracy"] = f1
            if f1 < 0.7:
                suggestions.append(
                    f"Low event detection F1 ({f1:.2f}) — review event type classification and temporal tolerance"
                )
            if ev_result.get("per_type"):
                worst_type = min(
                    ((t, d["f1"]) for t, d in ev_result["per_type"].items()),
                    key=lambda x: x[1],
                )
                if worst_type[1] < 0.6:
                    suggestions.append(
                        f"Event type '{worst_type[0]}' has F1={worst_type[1]:.2f} — consider feature engineering for this type"
                    )

        # Possession error
        if "possession_home" in self.ground_truth and "possession_home" in self.computed:
            pos_result = self.compare_possession(
                self.computed["possession_home"],
                self.ground_truth["possession_home"],
            )
            result.possession_error = pos_result
            err = pos_result.get("absolute_error", 100)
            score = max(0.0, 1.0 - err / 50.0)
            category_scores["possession"] = score
            if err > 5:
                suggestions.append(
                    f"Possession error is {err:.1f}% — check event filtering and possession assignment logic"
                )

        # Tracking accuracy
        computed_tracks = self.computed.get("tracks", self.computed.get("tracking", {}))
        if computed_tracks:
            track_result = self.compare_tracking(computed_tracks)
            result.tracking_metrics = track_result
            if track_result:
                mota = track_result.get("mota", 0.0)
                idf1 = track_result.get("idf1", 0.0)
                score = (mota + idf1) / 2.0
                category_scores["tracking"] = max(0.0, score)
                if mota < 0.5:
                    suggestions.append(
                        f"Tracking MOTA={mota:.3f} — high false positive/negative rate, check tracker thresholds and association logic"
                    )
                if idf1 < 0.5:
                    suggestions.append(
                        f"Tracking IDF1={idf1:.3f} — high ID switch rate, check ReID feature quality and matching threshold"
                    )

        # xG accuracy
        computed_shots = [e for e in self.computed.get("events", []) if e.get("type") == "shot"]
        if computed_shots:
            xg_result = self.compare_xg(computed_shots)
            result.xg_metrics = xg_result
            if xg_result:
                brier = xg_result.get("brier_score", 1.0)
                cal_error = xg_result.get("calibration_error", 1.0)
                brier_score = max(0.0, 1.0 - brier * 4.0)
                cal_score = max(0.0, 1.0 - cal_error * 5.0)
                xg_score = (brier_score + cal_score) / 2.0
                category_scores["xg"] = xg_score
                if brier > 0.25:
                    suggestions.append(
                        f"xG Brier score is {brier:.4f} (expected <0.20) — model probabilities may be poorly calibrated"
                    )
                if cal_error > 0.15:
                    suggestions.append(
                        f"xG calibration error is {cal_error:.4f} — consider isotonic regression or Platt scaling"
                    )

        # Overall
        if category_scores:
            overall = float(np.mean(list(category_scores.values())))
            result.overall_score = round(overall, 4)
            result.overall_grade = _grade_from_score(overall)
        else:
            result.overall_score = 0.0
            result.overall_grade = "N/A"
            suggestions.append(
                "No comparison data provided — load ground truth and computed data first"
            )

        result.per_category_scores = {k: round(v, 4) for k, v in category_scores.items()}
        result.improvement_suggestions = suggestions

        logger.info(
            "Audit complete: grade={} score={:.3f} ({} categories)",
            result.overall_grade,
            result.overall_score,
            len(category_scores),
        )

        return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy audit for football analytics models")
    parser.add_argument("--match-id", required=True, help="Match identifier")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--computed", default=None, help="Optional path to computed data JSON file")
    parser.add_argument(
        "--tolerance", type=float, default=2.0, help="Event match tolerance in seconds"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write audit report JSON (prints to stdout if omitted)",
    )
    args = parser.parse_args()

    auditor = AccuracyAudit(match_id=args.match_id, ground_truth_path=args.ground_truth)
    gt_data = auditor.ground_truth

    if args.computed:
        try:
            computed_path = Path(args.computed)
            if computed_path.exists():
                auditor.computed = json.loads(computed_path.read_text(encoding="utf-8"))
                logger.info("Loaded computed data from {}", args.computed)
            else:
                logger.warning("Computed data path not found: {}", args.computed)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load computed data: {}", exc)

    if not auditor.computed:
        computed_events = gt_data.get("events", [])
        auditor.computed = {"events": computed_events}
        logger.info("No computed data provided — using ground truth as computed (baseline check)")

    if auditor.computed:
        ev = auditor.computed.get("events", [])
        if ev:
            auditor.compare_events(ev, tolerance_s=args.tolerance)

        if "possession_home" in auditor.computed:
            gt_poss = gt_data.get("possession_home", 50.0)
            auditor.compare_possession(auditor.computed["possession_home"], gt_poss)

        computed_tracks = auditor.computed.get("tracks", auditor.computed.get("tracking", {}))
        if computed_tracks:
            auditor.compare_tracking(computed_tracks)

        computed_shots = [e for e in ev if e.get("type") == "shot"]
        if computed_shots:
            auditor.compare_xg(computed_shots)

    report = auditor.generate_report()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Report written to {}", args.output)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
