"""Train the Kawkab PSxG model on StatsBomb open-data on-target shots.

PSxG = P(goal | shot on target) — the goalkeeper metric (goals conceded
vs PSxG faced). Trained from every StatsBomb corpus shot whose outcome
is on-target (Saved / Goal / Post / Saved to Post) and that carries a
3D ``end_location`` [x, y, z] within the goal frame.

Features (all derivable from Kawkab's own pipeline or honest fallbacks):
    - distance to goal (m), distance²
    - opening angle (deg)
    - |placement distance from goal mouth center| (m)
    - placement height z (m)
    - |placement lateral offset| (m)
    - is_header, is_free_kick
    - angle × lateral interaction

The three hand-tuned implementations (psxg_model.py, psxg_improved.py,
psxg_model_trained.py — see CLAUDE.md "three independent PSxG
implementations") are superseded by these fitted weights; the old
signature on psxg_model.compute_psxg is kept (goalkeeper_analytics.py
depends on it) but the coefficients become data-driven.

Usage:
    PYTHONPATH=src python -m kawkab.core.validation.train_psxg \
        --corpus data/statsbomb_corpus --force
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from kawkab.core.validation.metrics import (
    brier_score,
    calibration_error,
    log_loss,
    reliability_curve,
    roc_auc,
)
from kawkab.core.validation.statsbomb_loader import (
    load_statsbomb_corpus,
    sb_to_meters,
    shot_distance_angle,
)
from kawkab.core.xg_trainer import batch_gradient_descent

# Goal geometry (Kawkab meters — matches statsbomb_loader constants)
GOAL_CENTER_Y = 34.0
GOAL_LINE_X = 105.0
HALF_GOAL_W = 3.66
GOAL_HEIGHT_M = 2.44

# SB goal center is y=40 in their 80-wide grid; lateral = y - 40
SB_GOAL_CENTER_Y = 40.0
SB_TO_M_Y = 68.0 / 80.0

ON_TARGET_OUTCOMES = {"Saved", "Goal", "Post", "Saved to Post"}

PSXG_FEATURE_NAMES = [
    "intercept",
    "distance_m",
    "distance_m_sq",
    "angle_opening_deg",
    "placement_height",
    "placement_lateral_abs",
    "lateral_height_interaction",
    "is_header",
    "is_free_kick",
]


@dataclass
class OnTargetShot:
    """One on-target shot with goal-mouth placement, for PSxG fitting."""

    distance_m: float
    angle_opening_deg: float
    placement_x_m: float  # lateral, goal-center-relative (−3.66..3.66)
    placement_z_m: float  # height above ground (0..2.44)
    is_header: bool
    is_free_kick: bool
    is_goal: bool
    match_id: str
    body_part: str


def extract_on_target_shots(corpus_dir: str | Path) -> list[OnTargetShot]:
    """Pull every on-target shot with a 3D end_location from the corpus."""
    corpus_dir = Path(corpus_dir)
    shots: list[OnTargetShot] = []
    for m in load_statsbomb_corpus(corpus_dir):
        raw_path = corpus_dir / f"{m.match_id}.json"
        with open(raw_path) as f:
            raw = json.load(f)
        for ev in raw:
            if ev.get("type", {}).get("name") != "Shot":
                continue
            shot = ev.get("shot") or {}
            if shot.get("outcome", {}).get("name") not in ON_TARGET_OUTCOMES:
                continue
            end_loc = shot.get("end_location") or []
            start_loc = ev.get("location") or []
            if len(end_loc) < 3 or len(start_loc) < 2:
                continue
            lateral_m = (end_loc[1] - SB_GOAL_CENTER_Y) * SB_TO_M_Y
            # SB open-data z is already meters (goal height ~2.44)
            height_m = end_loc[2]
            if not (0.0 <= height_m <= 3.2):
                height_m = max(0.0, min(GOAL_HEIGHT_M, height_m * 0.85))
            x_m, y_m = sb_to_meters(start_loc)
            distance_m, angle_deg = shot_distance_angle(x_m, y_m)
            body_part_name = (shot.get("body_part") or {}).get("name", "Right Foot")
            shot_type_name = (shot.get("type") or {}).get("name", "Open Play")
            shots.append(
                OnTargetShot(
                    distance_m=distance_m,
                    angle_opening_deg=angle_deg,
                    placement_x_m=max(-HALF_GOAL_W, min(HALF_GOAL_W, lateral_m)),
                    placement_z_m=max(0.0, min(GOAL_HEIGHT_M, height_m)),
                    is_header=(body_part_name == "Head"),
                    is_free_kick=(shot_type_name == "Free Kick"),
                    is_goal=(shot.get("outcome", {}).get("name") == "Goal"),
                    match_id=m.match_id,
                    body_part={
                        "Right Foot": "right_foot",
                        "Left Foot": "left_foot",
                        "Head": "head",
                    }.get(body_part_name, "other"),
                )
            )
    return shots


def split_train_val(
    shots: list[OnTargetShot],
    *,
    val_fraction: float = 0.2,
    seed: int = 13,
) -> tuple[list[OnTargetShot], list[OnTargetShot]]:
    """Match-level split — no leakage between train and val."""
    match_ids = sorted({s.match_id for s in shots})
    rng = np.random.default_rng(seed)
    idx = np.arange(len(match_ids))
    rng.shuffle(idx)
    n_val = max(1, int(len(match_ids) * val_fraction))
    val_ids = {match_ids[i] for i in idx[:n_val]}
    train = [s for s in shots if s.match_id not in val_ids]
    val = [s for s in shots if s.match_id in val_ids]
    return train, val


def build_feature_matrix(shots: list[OnTargetShot]) -> tuple[np.ndarray, np.ndarray]:
    n = len(shots)
    X = np.zeros((n, len(PSXG_FEATURE_NAMES)), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(shots):
        # Interaction: corners score most; the empirical zone table shows
        # top-center is the EASIEST on-target placement to save (0.156),
        # so lateral and height must interact positively, not blend into
        # a single center-distance radius (which conflated the two and
        # learned the wrong sign for top corners).
        X[i] = [
            1.0,
            s.distance_m,
            s.distance_m**2,
            s.angle_opening_deg,
            s.placement_z_m,
            abs(s.placement_x_m),
            abs(s.placement_x_m) * s.placement_z_m,
            1.0 if s.is_header else 0.0,
            1.0 if s.is_free_kick else 0.0,
        ]
        y[i] = 1.0 if s.is_goal else 0.0
    return X, y


def evaluate_psxg(name: str, p: np.ndarray, shots: list[OnTargetShot]) -> dict[str, Any]:
    y = np.array([1.0 if s.is_goal else 0.0 for s in shots])
    p = np.clip(p, 0.0, 1.0)
    try:
        auc = roc_auc(y, p)
    except ValueError:
        auc = float("nan")
    return {
        "model": name,
        "n_on_target": len(shots),
        "n_goals": int(y.sum()),
        "goal_rate": float(y.mean()),
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "auc": auc,
        "ece_10bin": calibration_error(y, p, n_bins=10),
        "reliability_curve_10bin": reliability_curve(y, p, n_bins=10),
        "mean_prediction": float(p.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/statsbomb_corpus")
    parser.add_argument("--out", default="src/kawkab/core/trained_psxg_coefficients.json")
    parser.add_argument("--report", default="docs/validation/psxg_training_report.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"[psxg-train] refusing to overwrite {out_path} (use --force)")
        return 1

    print(f"[psxg-train] loading corpus {args.corpus}")
    shots = extract_on_target_shots(args.corpus)
    if len(shots) < 100:
        print(f"[psxg-train] too few on-target shots ({len(shots)}); need >= 100")
        return 1
    print(f"[psxg-train] {len(shots)} on-target shots with 3D end_location")

    train, val = split_train_val(shots, seed=args.seed)
    print(f"[psxg-train] split: {len(train)} train / {len(val)} val (match-level)")

    X_train, y_train = build_feature_matrix(train)
    theta, _ = batch_gradient_descent(X_train, y_train, lr=args.lr, epochs=args.epochs, l2=args.l2)
    coeffs = dict(zip(PSXG_FEATURE_NAMES, theta.tolist(), strict=True))

    X_val, _ = build_feature_matrix(val)
    z_val = X_val @ theta
    p_val = 1.0 / (1.0 + np.exp(-np.clip(z_val, -30, 30)))

    evals = [evaluate_psxg("kawkab_psxg_trained", p_val, val)]

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": str(Path(args.corpus).resolve()),
        "n_on_target_shots": len(shots),
        "split": {"seed": args.seed, "n_train": len(train), "n_val": len(val)},
        "coefficients": coeffs,
        "evaluation": evals,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                **coeffs,
                "_model_name": "kawkab_psxg_trained_statsbomb",
                "_n_train_shots": len(train),
                "_trained_at": report["generated_at"],
                "_feature_names": PSXG_FEATURE_NAMES,
            },
            f,
            indent=2,
        )
    print(f"[psxg-train] coefficients -> {out_path}")

    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[psxg-train] report -> {rp}")

    for ev in evals:
        print(
            f"  {ev['model']:>24}  Brier {ev['brier']:.4f}  AUC {ev['auc']:.4f}  "
            f"LogLoss {ev['log_loss']:.4f}  ECE {ev['ece_10bin']:.4f}  "
            f"mean {ev['mean_prediction']:.4f} (true rate {ev['goal_rate']:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
