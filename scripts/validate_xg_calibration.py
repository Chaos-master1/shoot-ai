"""xG calibration harness — score the app's xG models against real goals.

Loads the StatsBomb open-data corpus (data/statsbomb_corpus/, 297 matches),
computes predictions from the app's trained and heuristic xG models plus
StatsBomb's own xG as the reference ceiling, and reports Brier score,
log loss and a 10-bin reliability curve.

Reference numbers (2026-09-18, boxmot/tracking validation round):
    297 matches, 7354 shots, 696 goals
    trained:   Brier=0.0789 LogLoss=0.2790 ECE=0.0277
    heuristic: Brier=0.0789 LogLoss=0.2819
    SB xG:     Brier=0.0692 LogLoss=0.2499

Interpretation: the app's trained model is within ~0.01 Brier of
StatsBomb's own published xG on the same shots -- properly calibrated,
not a placeholder. A regression that pushes the trained model's Brier
materially above the heuristic's (or >0.02 above the SB ceiling) is a
bug, and the guards below fail loudly when that happens.

Usage:
    PYTHONPATH=src .venv-linux/bin/python scripts/validate_xg_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from kawkab.core.validation.statsbomb_loader import (  # noqa: E402
    extract_shots_for_fitting,
    load_statsbomb_corpus,
)
from kawkab.core.xg_calibration import (  # noqa: E402
    compute_brier_score,
    compute_calibration_curve,
    compute_log_loss,
)
from kawkab.core.xg_model import (  # noqa: E402
    compute_xg_from_dict,
    compute_xg_trained_from_dict,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "statsbomb_corpus"

# Regression guards -- loose enough to be noise-immune, tight enough to
# catch catastrophic model regressions (pathological refits, broken
# feature plumbing).
MAX_TRAINED_BRIER = 0.10
MAX_TRAINED_MINUS_HEURISTIC = 0.005
MAX_TRAINED_MINUS_SB = 0.02


def main() -> int:
    if not CORPUS_DIR.exists():
        print(f"corpus not found: {CORPUS_DIR}")
        return 2

    matches = list(load_statsbomb_corpus(CORPUS_DIR))
    shots, n_matches = extract_shots_for_fitting(matches)
    if len(shots) < 1000:
        print(f"corpus too small: {len(shots)} shots from {n_matches} matches")
        return 2

    outcomes = [1 if s.is_goal else 0 for s in shots]
    trained = [compute_xg_trained_from_dict(s.to_fit_dict()) for s in shots]
    heuristic = [compute_xg_from_dict(s.to_fit_dict()) for s in shots]
    sb = [float(s.statsbomb_xg) for s in shots]

    b_t = compute_brier_score(trained, outcomes)
    b_h = compute_brier_score(heuristic, outcomes)
    b_sb = compute_brier_score(sb, outcomes)
    ll_t = compute_log_loss(trained, outcomes)
    ll_h = compute_log_loss(heuristic, outcomes)
    ll_sb = compute_log_loss(sb, outcomes)

    print(
        f"corpus: {n_matches} matches, {len(shots)} shots, {sum(outcomes)} goals\n"
    )
    print(f"{'model':<12} {'Brier':>8} {'LogLoss':>9}")
    print(f"{'trained':<12} {b_t:>8.4f} {ll_t:>9.4f}")
    print(f"{'heuristic':<12} {b_h:>8.4f} {ll_h:>9.4f}")
    print(f"{'SB xG':<12} {b_sb:>8.4f} {ll_sb:>9.4f}")

    curve = compute_calibration_curve(trained, outcomes, n_bins=10)
    print(f"\ntrained reliability: ECE={curve.ece:.4f}")
    print("bin_mid  mean_pred  observed")
    for mid, pred, obs in zip(curve.bins, curve.predicted, curve.observed, strict=True):
        print(f"  {mid:.2f}     {pred:.4f}    {obs:.4f}")

    failures: list[str] = []
    if b_t > MAX_TRAINED_BRIER:
        failures.append(
            f"trained Brier {b_t:.4f} > guard {MAX_TRAINED_BRIER} (pathological refit?)"
        )
    if b_t > b_h + MAX_TRAINED_MINUS_HEURISTIC:
        failures.append(
            f"trained Brier {b_t:.4f} exceeds heuristic {b_h:.4f} "
            f"by more than {MAX_TRAINED_MINUS_HEURISTIC} -- refit regressed"
        )
    if b_t > b_sb + MAX_TRAINED_MINUS_SB:
        failures.append(
            f"trained Brier {b_t:.4f} is more than {MAX_TRAINED_MINUS_SB} above "
            f"the StatsBomb ceiling {b_sb:.4f} -- model broken"
        )

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll calibration guards passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
