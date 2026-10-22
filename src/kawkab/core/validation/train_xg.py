"""Train the Kawkab xG model on the StatsBomb open-data corpus.

Usage:
    PYTHONPATH=src python -m kawkab.core.validation.train_xg \
        --corpus data/statsbomb_corpus \
        --out src/kawkab/core/trained_xg_coefficients.json \
        --report docs/validation/xg_training_report.json

Pipeline:
    1. Load every match in the corpus, extract non-penalty shots.
    2. Split train/val by MATCH (no leakage across halves of the same
       match), seeded and reproducible.
    3. Fit logistic regression (numpy-only ``xg_trainer``) on train.
    4. Evaluate on val: Brier, AUC, log loss, ECE, reliability curve —
       for the trained model, the heuristic fallback, and StatsBomb's
       own published xG as the reference bar.
    5. Save coefficients + full evaluation JSON (provenance included).

Run as a module (not a script) so kawkab imports resolve.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from kawkab.core.validation.metrics import (
    brier_score,
    brier_skill_score,
    calibration_error,
    log_loss,
    reliability_curve,
    roc_auc,
)
from kawkab.core.validation.statsbomb_loader import (
    StatsBombMatch,
    StatsBombShot,
    extract_shots_for_fitting,
    load_statsbomb_corpus,
)
from kawkab.core.xg_model import ENHANCED_COEFFICIENTS, PENALTY_XG, EnhancedXgModel
from kawkab.core.xg_trainer import FEATURE_NAMES, FitShot, batch_gradient_descent


def shots_to_fit_shots(shots: list[StatsBombShot]) -> list[FitShot]:
    """StatsBombShot list -> FitShot list (the trainer's input type)."""
    out: list[FitShot] = []
    for s in shots:
        out.append(
            FitShot(
                distance_m=s.distance_m,
                # The trainer's angle feature is (1 - cos(deviation)):
                # it must receive the DEVIATION-from-central angle, not
                # the goal-opening angle (which would train the angle
                # terms backwards — see statsbomb_loader.shot_deviation_angle).
                angle_deg=s.angle_deviation_deg,
                is_header=(s.body_part == "head"),
                is_one_on_one=s.is_one_on_one,
                is_pressed=s.is_pressed,
                is_volley=False,
                is_free_kick=(s.shot_type == "free_kick"),
                gk_distance_m=s.gk_distance_m,
                is_rebound=s.is_rebound,
                is_big_chance=s.is_big_chance,
                is_goal=s.is_goal,
            )
        )
    return out


def split_train_val(
    matches: list[StatsBombMatch],
    *,
    val_fraction: float = 0.2,
    seed: int = 13,
) -> tuple[list[StatsBombShot], list[StatsBombShot]]:
    """Match-level split — every shot from one match stays on one side."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(matches))
    rng.shuffle(idx)
    n_val = max(1, int(len(matches) * val_fraction))
    val_ids = {matches[i].match_id for i in idx[:n_val]}
    train: list[StatsBombShot] = []
    val: list[StatsBombShot] = []
    for m in matches:
        target = val if m.match_id in val_ids else train
        for s in m.shots:
            if s.shot_type != "penalty":
                target.append(s)
    return train, val


def fit_with_holdout(
    train_shots: list[StatsBombShot],
    *,
    epochs: int = 8000,
    lr: float = 0.5,
    l2: float = 0.001,
) -> dict[str, float]:
    """Fit coefficients on the training split. <10 shots falls back to heuristic."""
    fit_shots = shots_to_fit_shots(train_shots)
    if len(fit_shots) < 10:
        return dict(ENHANCED_COEFFICIENTS)
    x_y = _build_xy(fit_shots)
    theta, _ = batch_gradient_descent(x_y[0], x_y[1], lr=lr, epochs=epochs, l2=l2)
    coeffs = dict(zip(FEATURE_NAMES, theta.tolist(), strict=True))
    return coeffs


def _build_xy(shots: list[FitShot]) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix build — mirrors xg_trainer._build_feature_matrix.

    Duplicated here (rather than importing the private helper) so the
    training path survives refactors of the trainer's internals; kept
    byte-for-byte equivalent semantics.
    """
    from kawkab.core.xg_trainer import _build_feature_matrix

    return _build_feature_matrix(shots)


def predict_xg(
    coeffs: dict[str, float],
    shots: list[StatsBombShot],
) -> np.ndarray:
    """Apply fitted coefficients to a shot list (penalties fixed at 0.76)."""
    model = EnhancedXgModel(coefficients=coeffs)
    preds: list[float] = []
    for s in shots:
        if s.shot_type == "penalty":
            preds.append(PENALTY_XG)
            continue
        ev = {
            "type": "shot",
            "distance_m": s.distance_m,
            "angle_deg": s.angle_deviation_deg,
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
        # extract_features reads body_part/shot_type via ShotEvent.from_dict
        preds.append(model.compute(ev))
    return np.array(preds, dtype=np.float64)


def evaluate(
    name: str,
    preds: np.ndarray,
    shots: list[StatsBombShot],
) -> dict[str, Any]:
    """Full metric suite for one set of predictions against outcomes."""
    y = np.array([1.0 if s.is_goal else 0.0 for s in shots], dtype=np.float64)
    p = np.clip(preds, 0.0, 1.0)
    try:
        auc = roc_auc(y, p)
    except ValueError:
        auc = float("nan")
    return {
        "model": name,
        "n_shots": len(shots),
        "n_goals": int(np.sum(y)),
        "goal_rate": float(np.mean(y)),
        "brier": brier_score(y, p),
        "brier_skill_vs_climatology": brier_skill_score(y, p),
        "log_loss": log_loss(y, p),
        "auc": auc,
        "ece_10bin": calibration_error(y, p, n_bins=10),
        "reliability_curve_10bin": reliability_curve(y, p, n_bins=10),
        "mean_prediction": float(np.mean(p)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/statsbomb_corpus")
    parser.add_argument("--out", default="src/kawkab/core/trained_xg_coefficients.json")
    parser.add_argument("--report", default=None, help="also write evaluation JSON here")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument(
        "--force",
        action="store_true",
        help="train even when coefficients already exist (default: refuse)",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"[xg-train] refusing to overwrite {out_path} (use --force)")
        return 1

    print(f"[xg-train] loading corpus {args.corpus}")
    matches = list(load_statsbomb_corpus(args.corpus))
    if len(matches) < 5:
        print(f"[xg-train] too few matches ({len(matches)}); need >= 5")
        return 1
    all_shots, _ = extract_shots_for_fitting(matches)
    print(f"[xg-train] {len(matches)} matches, {len(all_shots)} non-penalty shots")

    train, val = split_train_val(matches, val_fraction=args.val_fraction, seed=args.seed)
    print(f"[xg-train] split: {len(train)} train / {len(val)} val shots")

    coeffs = fit_with_holdout(train, epochs=args.epochs, lr=args.lr, l2=args.l2)
    n_train = len(train)

    # Evaluate on the held-out validation split
    p_trained = predict_xg(coeffs, val)
    p_heuristic = predict_xg(dict(ENHANCED_COEFFICIENTS), val)
    p_sb = np.array([s.statsbomb_xg for s in val], dtype=np.float64)

    evals = [
        evaluate("kawkab_trained", p_trained, val),
        evaluate("kawkab_heuristic", p_heuristic, val),
        evaluate("statsbomb_xg_reference", p_sb, val),
    ]

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": str(Path(args.corpus).resolve()),
        "n_matches": len(matches),
        "n_shots_total": len(all_shots),
        "split": {"val_fraction": args.val_fraction, "seed": args.seed},
        "train_shots": n_train,
        "val_shots": len(val),
        "coefficients": coeffs,
        "evaluation": evals,
    }

    serializable_coeffs: dict[str, Any] = {
        k: v for k, v in coeffs.items() if isinstance(v, (int, float))
    }
    serializable_coeffs["_model_name"] = "kawkab_trained_statsbomb"
    serializable_coeffs["_n_train_shots"] = n_train
    serializable_coeffs["_trained_at"] = report["generated_at"]
    serializable_coeffs["_feature_names"] = FEATURE_NAMES

    with open(out_path, "w") as f:
        json.dump(serializable_coeffs, f, indent=2)
    print(f"[xg-train] coefficients -> {out_path}")

    if args.report:
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[xg-train] report -> {rp}")

    for ev in evals:
        print(
            f"  {ev['model']:>26}  Brier {ev['brier']:.4f}  AUC {ev['auc']:.4f}  "
            f"LogLoss {ev['log_loss']:.4f}  ECE {ev['ece_10bin']:.4f}  mean_xG {ev['mean_prediction']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
