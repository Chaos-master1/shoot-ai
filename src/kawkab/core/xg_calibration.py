"""xG model calibration using reliability diagrams and Platt scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationCurve:
    bins: list[float]
    observed: list[float]
    predicted: list[float]
    mse: float
    ece: float


def compute_calibration_curve(
    predictions: list[float], outcomes: list[int], n_bins: int = 10
) -> CalibrationCurve:
    preds = np.array(predictions, dtype=np.float64)
    outc = np.array(outcomes, dtype=np.float64)
    if len(preds) == 0 or n_bins < 1:
        return CalibrationCurve(bins=[], observed=[], predicted=[], mse=0.0, ece=0.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.searchsorted(bin_edges[1:], preds, side="left"), 0, n_bins - 1)

    bin_midpoints: list[float] = []
    observed_rates: list[float] = []
    predicted_means: list[float] = []
    mse_sum = 0.0
    ece_sum = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        cnt = int(np.sum(mask))
        if cnt == 0:
            mid = (bin_edges[i] + bin_edges[i + 1]) / 2.0
            bin_midpoints.append(mid)
            observed_rates.append(0.0)
            predicted_means.append(mid)
            continue
        mean_pred = float(np.mean(preds[mask]))
        obs_rate = float(np.mean(outc[mask]))
        bin_midpoints.append(float((bin_edges[i] + bin_edges[i + 1]) / 2.0))
        observed_rates.append(obs_rate)
        predicted_means.append(mean_pred)
        diff = mean_pred - obs_rate
        mse_sum += diff * diff * cnt
        ece_sum += abs(diff) * cnt

    total = len(preds)
    return CalibrationCurve(
        bins=bin_midpoints,
        observed=observed_rates,
        predicted=predicted_means,
        mse=mse_sum / max(total, 1),
        ece=ece_sum / max(total, 1),
    )


def platt_scale(
    predictions: list[float], outcomes: list[int], max_iter: int = 100
) -> tuple[float, float]:
    preds = np.array(predictions, dtype=np.float64)
    outc = np.array(outcomes, dtype=np.float64)
    eps = 1e-15

    preds = np.clip(preds, eps, 1.0 - eps)
    logit_p = np.log(preds / (1.0 - preds))

    a, b = 0.0, 0.0
    prior0 = float(max(1, int(np.sum(outc == 0))))
    prior1 = float(max(1, int(np.sum(outc == 1))))
    T = np.array([prior1, prior0], dtype=np.float64)
    T /= np.sum(T)

    for _ in range(max_iter):
        exp_a_logit = np.exp(a * logit_p + b)
        p = exp_a_logit / (1.0 + exp_a_logit)
        p = np.clip(p, eps, 1.0 - eps)

        grad_a = np.sum(T[1] * (1.0 - outc) * p * logit_p - T[0] * outc * (1.0 - p) * logit_p)
        grad_b = np.sum(T[1] * (1.0 - outc) * p - T[0] * outc * (1.0 - p))

        w = p * (1.0 - p)
        hess_a = np.sum(T[1] * (1.0 - outc) * w * logit_p**2 + T[0] * outc * w * logit_p**2)
        hess_b = np.sum(T[1] * (1.0 - outc) * w + T[0] * outc * w)
        hess_ab = np.sum(T[1] * (1.0 - outc) * w * logit_p + T[0] * outc * w * logit_p)

        det = hess_a * hess_b - hess_ab * hess_ab
        if abs(det) < eps:
            break
        da = -(hess_b * grad_a - hess_ab * grad_b) / det
        db = -(-hess_ab * grad_a + hess_a * grad_b) / det

        a += da
        b += db
        if abs(da) < 1e-7 and abs(db) < 1e-7:
            break

    return a, b


def apply_platt_scale(predictions: list[float], a: float, b: float) -> list[float]:
    preds = np.array(predictions, dtype=np.float64)
    eps = 1e-15
    preds = np.clip(preds, eps, 1.0 - eps)
    logit_p = np.log(preds / (1.0 - preds))
    calibrated = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    return np.clip(calibrated, eps, 1.0 - eps).tolist()


def compute_brier_score(predictions: list[float], outcomes: list[int]) -> float:
    preds = np.array(predictions, dtype=np.float64)
    outc = np.array(outcomes, dtype=np.float64)
    if len(preds) == 0:
        return 0.0
    return float(np.mean((preds - outc) ** 2))


def compute_log_loss(predictions: list[float], outcomes: list[int], eps: float = 1e-15) -> float:
    preds = np.array(predictions, dtype=np.float64)
    outc = np.array(outcomes, dtype=np.float64)
    if len(preds) == 0:
        return 0.0
    preds = np.clip(preds, eps, 1.0 - eps)
    return float(-np.mean(outc * np.log(preds) + (1.0 - outc) * np.log(1.0 - preds)))


def compute_auc_roc(predictions: list[float], outcomes: list[int]) -> float:
    preds = np.array(predictions, dtype=np.float64)
    outc = np.array(outcomes, dtype=np.float64)
    if len(preds) == 0:
        return 0.5
    n_pos = int(np.sum(outc))
    n_neg = len(outc) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    pos_preds = preds[outc == 1]
    neg_preds = preds[outc == 0]
    concordant = 0.0
    for p in pos_preds:
        concordant += float(np.sum(neg_preds < p))
        concordant += 0.5 * float(np.sum(neg_preds == p))

    total_pairs = n_pos * max(n_neg, 1)
    if total_pairs == 0:
        return 0.5
    return concordant / total_pairs
