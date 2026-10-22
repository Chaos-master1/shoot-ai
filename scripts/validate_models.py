"""Validate Kawkab models against public ground truth.

Usage:
    PYTHONPATH=src python scripts/validate_models.py \
        [--corpus data/statsbomb_corpus] [--out docs/validation/validation_report.json]

Emits a reproducible JSON + markdown report:
    - xG (trained + heuristic) vs StatsBomb's published xG, with Brier,
      AUC, log loss, ECE, reliability curves, and bootstrap CIs.
    - Corpus provenance (n matches, n shots, split seed).

Exit code 0 = report written (even if numbers are bad — honesty is the
gate, not optimism).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Make 'kawkab' importable when run as a plain script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kawkab.core.validation.metrics import (  # noqa: E402
    bootstrap_ci,
    pearson_correlation,
)
from kawkab.core.validation.statsbomb_loader import (  # noqa: E402
    extract_shots_for_fitting,
    load_statsbomb_corpus,
)
from kawkab.core.validation.train_xg import (  # noqa: E402
    evaluate,
    predict_xg,
    split_train_val,
)
from kawkab.core.xg_model import (  # noqa: E402
    ENHANCED_COEFFICIENTS,
    EnhancedXgModel,
    trained_model_available,
)


def validate_xg(
    corpus_dir: Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 13,
) -> dict:
    """Full xG validation block for the report."""
    matches = list(load_statsbomb_corpus(corpus_dir))
    all_shots, _ = extract_shots_for_fitting(matches)
    train, val = split_train_val(matches, val_fraction=val_fraction, seed=seed)

    model = EnhancedXgModel()  # auto-loads trained coefficients if present
    p_trained = predict_xg(model.coef, val)
    p_heuristic = predict_xg(dict(ENHANCED_COEFFICIENTS), val)
    p_sb = [s.statsbomb_xg for s in val]

    evals = [
        evaluate("kawkab_active", p_trained, val),
        evaluate("kawkab_heuristic", p_heuristic, val),
        evaluate("statsbomb_xg_reference", p_sb, val),
    ]

    # Bootstrap CIs on headline numbers (per-shot Brier contributions)
    import numpy as np

    y = np.array([1.0 if s.is_goal else 0.0 for s in val])
    brier_ci = bootstrap_ci((p_trained - y) ** 2, statistic="mean", n_bootstrap=1000, seed=seed)
    brier_ci_sb = bootstrap_ci(
        (np.asarray(p_sb, dtype=np.float64) - y) ** 2,
        statistic="mean",
        n_bootstrap=1000,
        seed=seed,
    )

    # Agreement with StatsBomb per-shot (not skill, just agreement)
    mae_vs_sb = float(np.mean(np.abs(p_trained - np.asarray(p_sb))))
    corr_vs_sb = pearson_correlation(p_trained, p_sb)

    return {
        "active_model_provenance": model.coeffs_source,
        "trained_weights_found_on_disk": trained_model_available(),
        "n_matches": len(matches),
        "n_shots_non_penalty": len(all_shots),
        "split": {"val_fraction": val_fraction, "seed": seed, "n_val": len(val)},
        "evaluation": evals,
        "agreement_with_statsbomb": {
            "per_shot_mae": mae_vs_sb,
            "per_shot_pearson_r": corr_vs_sb,
        },
        "bootstrap_ci": {
            "kawkab_brier": brier_ci,
            "statsbomb_brier": brier_ci_sb,
        },
    }


def validate_psxg(corpus_dir: Path, *, seed: int = 13) -> dict | None:
    """PSxG validation block — trained vs on-target shots from the corpus."""
    import numpy as np

    from kawkab.core.validation.train_psxg import (
        build_feature_matrix,
        evaluate_psxg,
        extract_on_target_shots,
        split_train_val,
    )
    from kawkab.core.xg_trainer import batch_gradient_descent

    shots = extract_on_target_shots(corpus_dir)
    if len(shots) < 100:
        return None
    train, val = split_train_val(shots, seed=seed)
    X_train, y_train = build_feature_matrix(train)
    theta, _ = batch_gradient_descent(X_train, y_train, lr=0.5, epochs=8000, l2=0.001)
    X_val, _ = build_feature_matrix(val)
    p_val = 1.0 / (1.0 + np.exp(-np.clip(X_val @ theta, -30, 30)))
    return {
        "n_on_target_shots": len(shots),
        "split": {"n_train": len(train), "n_val": len(val), "seed": seed},
        "evaluation": [evaluate_psxg("kawkab_psxg_trained", p_val, val)],
    }


def validate_xt(corpus_dir: Path) -> dict | None:
    """xT reference-grid sanity block (monotonicity + provenance)."""
    from kawkab.core.validation.train_xt import train_reference_grid

    result = train_reference_grid(corpus_dir)
    grid = result["grid"]
    n_cols = result["cols"]
    own_half = max(max(row[: n_cols // 2]) for row in grid)
    final_third = max(max(row[2 * n_cols // 3 :]) for row in grid)
    return {
        "n_matches": result["n_matches"],
        "n_actions": result["n_actions"],
        "grid_shape": [result["rows"], result["cols"]],
        "sanity": {
            "max_own_half": own_half,
            "max_final_third": final_third,
            "monotonic": own_half < final_third,
        },
    }


def to_markdown(report: dict) -> str:
    """Render the report as a markdown page (docs/validation/REPORT.md)."""
    lines: list[str] = []
    lines.append("# Kawkab AI — Model Validation Report")
    lines.append("")
    lines.append(f"*Generated {report['generated_at']} — fully reproducible:*")
    lines.append("```")
    lines.append("PYTHONPATH=src python scripts/validate_models.py")
    lines.append("```")
    lines.append("")
    lines.append("## xG (Expected Goals)")
    lines.append("")
    xg = report["xg"]
    lines.append(f"- Active model: **{xg['active_model_provenance']}**")
    lines.append(
        f"- Corpus: **{xg['n_matches']} StatsBomb open-data matches**, "
        f"{xg['n_shots_non_penalty']} non-penalty shots"
    )
    lines.append(
        f"- Match-level train/val split: seed {xg['split']['seed']}, "
        f"{xg['split']['val_fraction']:.0%} val ({xg['split']['n_val']} shots)"
    )
    lines.append("")
    lines.append("| Model | Brier ↓ | AUC ↑ | LogLoss ↓ | ECE ↓ | mean xG |")
    lines.append("|---|---|---|---|---|---|")
    for ev in xg["evaluation"]:
        lines.append(
            f"| {ev['model']} | {ev['brier']:.4f} | {ev['auc']:.4f} | "
            f"{ev['log_loss']:.4f} | {ev['ece_10bin']:.4f} | {ev['mean_prediction']:.4f} |"
        )
    lines.append("")
    bc = xg["bootstrap_ci"]
    lines.append(
        f"- Brier 95% CI: Kawkab **{bc['kawkab_brier']['ci_lower']:.4f}–{bc['kawkab_brier']['ci_upper']:.4f}** "
        f"vs StatsBomb {bc['statsbomb_brier']['ci_lower']:.4f}–{bc['statsbomb_brier']['ci_upper']:.4f}"
    )
    ag = xg["agreement_with_statsbomb"]
    lines.append(
        f"- Per-shot agreement with StatsBomb xG: MAE {ag['per_shot_mae']:.4f}, r = {ag['per_shot_pearson_r']:.3f}"
    )
    lines.append("")
    lines.append(
        "*Penalties excluded (fixed 0.76 xG). Outcomes from StatsBomb "
        "open data (CC BY-NC-SA 4.0, non-commercial). Lower Brier = better; "
        "Brier 0.25 = coin-flip baseline.*"
    )
    lines.append("")

    if report.get("psxg"):
        ps = report["psxg"]
        lines.append("## PSxG (Post-Shot xG — goalkeeper metric)")
        lines.append("")
        lines.append(
            f"- Trained on **{ps['n_on_target_shots']} on-target shots** "
            f"({ps['split']['n_train']} train / {ps['split']['n_val']} held-out, "
            f"match-level split)"
        )
        for ev in ps["evaluation"]:
            lines.append(
                f"- {ev['model']}: Brier **{ev['brier']:.4f}**, AUC **{ev['auc']:.4f}**, "
                f"ECE {ev['ece_10bin']:.4f}, mean PSxG {ev['mean_prediction']:.4f} "
                f"(true goal rate {ev['goal_rate']:.4f})"
            )
        lines.append("")
    if report.get("xt"):
        xt = report["xt"]
        s = xt["sanity"]
        lines.append("## xT (Expected Threat reference grid)")
        lines.append("")
        lines.append(
            f"- League-wide grid trained from **{xt['n_matches']} matches / "
            f"{xt['n_actions']:,} actions** — shape {xt['grid_shape'][0]}x{xt['grid_shape'][1]}"
        )
        lines.append(
            f"- Monotonicity check: max own-half zone {s['max_own_half']:.4f} "
            f"< max final-third zone {s['max_final_third']:.4f} → **{s['monotonic']}**"
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/statsbomb_corpus")
    parser.add_argument("--out", default="docs/validation/validation_report.json")
    parser.add_argument("--markdown-out", default="docs/validation/REPORT.md")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    if not corpus.is_dir():
        print(f"validate_models: corpus dir not found: {corpus}")
        return 1

    xg_block = validate_xg(corpus, seed=args.seed)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus_dir": str(corpus.resolve()),
        "xg": xg_block,
        "psxg": validate_psxg(corpus, seed=args.seed),
        "xt": validate_xt(corpus),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[validate] JSON -> {out}")

    md = to_markdown(report)
    if args.markdown_out:
        md_out = Path(args.markdown_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        with open(md_out, "w") as f:
            f.write(md)
        print(f"[validate] markdown -> {md_out}")

    for ev in xg_block["evaluation"]:
        print(
            f"  {ev['model']:>26}  Brier {ev['brier']:.4f}  AUC {ev['auc']:.4f}  "
            f"LogLoss {ev['log_loss']:.4f}  ECE {ev['ece_10bin']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
