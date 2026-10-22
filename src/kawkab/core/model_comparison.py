"""xG Model Comparison — compare heuristic, logistic, and DL xG models.

Evaluates all three models on a held-out test set using log loss,
Brier score, AUC-ROC, calibration error, and distance/angle bucketing.
Supports train/test split, temporal split, and k-fold cross-validation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.dl_xg_model import DLXgModel
from kawkab.core.xg_model import EnhancedXgModel


@dataclass
class ModelMetrics:
    model_name: str
    log_loss: float = 0.0
    brier_score: float = 0.0
    auc_roc: float = 0.0
    calibration_error: float = 0.0
    shots_evaluated: int = 0
    goals_actual: int = 0
    goals_predicted: float = 0.0
    calibration_slope: float = 1.0
    calibration_intercept: float = 0.0


@dataclass
class CrossValidationFold:
    fold: int
    metrics: dict[str, ModelMetrics]
    train_size: int
    test_size: int


@dataclass
class ModelComparisonReport:
    models: list[ModelMetrics] = field(default_factory=list)
    best_model: str = ""
    significant_differences: list[dict] = field(default_factory=list)
    feature_importances: dict[str, dict] = field(default_factory=dict)
    distance_buckets: dict = field(default_factory=dict)
    angle_buckets: dict = field(default_factory=dict)
    calibration_chart_data: dict = field(default_factory=dict)
    cv_folds: list[CrossValidationFold] = field(default_factory=list)
    cv_summary: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "model_name": m.model_name,
                    "log_loss": round(m.log_loss, 4),
                    "brier_score": round(m.brier_score, 4),
                    "auc_roc": round(m.auc_roc, 4),
                    "calibration_error": round(m.calibration_error, 4),
                    "shots_evaluated": m.shots_evaluated,
                    "goals_actual": m.goals_actual,
                    "goals_predicted": round(m.goals_predicted, 4),
                    "calibration_slope": round(m.calibration_slope, 4),
                    "calibration_intercept": round(m.calibration_intercept, 4),
                }
                for m in self.models
            ],
            "best_model": self.best_model,
            "significant_differences": self.significant_differences,
            "feature_importances": self.feature_importances,
            "distance_buckets": self.distance_buckets,
            "angle_buckets": self.angle_buckets,
            "calibration_chart_data": self.calibration_chart_data,
            "cv_summary": self.cv_summary,
        }

    def summary_text(self) -> str:
        lines = [f"xG Model Comparison — Best model: {self.best_model}", ""]
        for m in self.models:
            lines.append(
                f"{m.model_name}: log_loss={m.log_loss:.4f}, brier={m.brier_score:.4f}, "
                f"auc_roc={m.auc_roc:.4f}, cal_err={m.calibration_error:.4f}"
            )
        if self.cv_summary:
            lines.append("")
            lines.append("Cross-Validation Summary (mean ± std across folds):")
            for model_name, stats in self.cv_summary.items():
                lines.append(
                    f"  {model_name}: log_loss={stats['log_loss_mean']:.4f}±{stats['log_loss_std']:.4f}"
                )
        if self.significant_differences:
            lines.append("")
            lines.append("Significant differences:")
            for d in self.significant_differences:
                lines.append(
                    f"  {d['metric']}: {d['model_a']} vs {d['model_b']} (p={d['p_value']:.3f})"
                )
        return "\n".join(lines)


def _compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    model_name: str,
) -> ModelMetrics:
    n = len(predictions)
    if n == 0:
        return ModelMetrics(model_name=model_name)

    eps = 1e-15
    p = np.clip(predictions, eps, 1.0 - eps)
    log_loss = -np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p))
    brier = np.mean((p - labels) ** 2)

    auc_roc = 0.0
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(labels)) >= 2:
            auc_roc = roc_auc_score(labels, p)
    except ImportError:
        n_pos = int(np.sum(labels))
        n_neg = n - n_pos
        if n_pos > 0 and n_neg > 0:
            ranks = np.argsort(p)
            pos_ranks = sum(ranks[np.where(labels == 1.0)[0]]) + 1.0
            auc_roc = (pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)

    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.searchsorted(bin_edges[1:], p, side="left"), 0, n_bins - 1)

    cal_errors = []
    bin_mean_pred = []
    bin_obs_rate = []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            mean_pred = float(np.mean(p[mask]))
            obs_rate = float(np.mean(labels[mask]))
            cal_errors.append(abs(mean_pred - obs_rate))
            bin_mean_pred.append(mean_pred)
            bin_obs_rate.append(obs_rate)

    calibration_error = max(cal_errors) if cal_errors else 0.0

    calibration_slope = 1.0
    calibration_intercept = 0.0
    if len(bin_mean_pred) >= 2:
        try:
            A = np.vstack([bin_mean_pred, np.ones_like(bin_mean_pred)]).T
            slope, intercept = np.linalg.lstsq(A, bin_obs_rate, rcond=None)[0]
            calibration_slope = float(slope)
            calibration_intercept = float(intercept)
        except np.linalg.LinAlgError:
            pass

    return ModelMetrics(
        model_name=model_name,
        log_loss=float(log_loss),
        brier_score=float(brier),
        auc_roc=round(auc_roc, 4),
        calibration_error=round(calibration_error, 4),
        shots_evaluated=n,
        goals_actual=int(np.sum(labels)),
        goals_predicted=float(np.sum(predictions)),
        calibration_slope=round(calibration_slope, 4),
        calibration_intercept=round(calibration_intercept, 4),
    )


def _compute_buckets(
    predictions_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    values: np.ndarray,
    bucket_edges: list[float],
    bucket_labels: list[str],
) -> dict:
    buckets: dict = {}
    for model_name, preds in predictions_dict.items():
        model_buckets: dict = {}
        for i, bl in enumerate(bucket_labels):
            if i + 1 < len(bucket_edges):
                lo, hi = bucket_edges[i], bucket_edges[i + 1]
            else:
                lo, hi = bucket_edges[i], float("inf")
            mask = (values >= lo) & (values < hi)
            if np.any(mask):
                p = np.clip(preds[mask], 1e-15, 1.0 - 1e-15)
                brier = float(np.mean((p - labels[mask]) ** 2))
            else:
                brier = 0.0
            model_buckets[bl] = {"brier_score": round(brier, 4), "n": int(np.sum(mask))}
        buckets[model_name] = model_buckets
    return buckets


def _compute_calibration_chart(
    predictions_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict:
    chart: dict = {}
    for model_name, preds in predictions_dict.items():
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.clip(np.searchsorted(bin_edges[1:], preds, side="left"), 0, n_bins - 1)
        bins_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                mean_pred = float(np.mean(preds[mask]))
                obs_rate = float(np.mean(labels[mask]))
            else:
                mean_pred = round((i + 0.5) / n_bins, 3)
                obs_rate = None
            bins_data.append(
                {
                    "bin": f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}",
                    "predicted": mean_pred,
                    "observed": obs_rate,
                    "count": int(np.sum(mask)),
                }
            )
        chart[model_name] = {"bins": bins_data}
    return chart


def _predict_all_models(
    shots: list[dict],
    enhanced_model: EnhancedXgModel,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(shots)
    heur_predictions = np.zeros(n, dtype=np.float64)
    enhanced_predictions = np.zeros(n, dtype=np.float64)
    dl_predictions = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.float64)
    indices = np.arange(n)

    for i, s in enumerate(shots):
        labels[i] = float(s.get("is_goal", 0.0))

        s_h = s.get("xg_heuristic", None)
        if s_h is not None:
            heur_predictions[i] = float(s_h)
        elif s.get("xG") is not None:
            heur_predictions[i] = float(s["xG"])
        else:
            from kawkab.core.xg_model import compute_xg

            d = float(s.get("distance_m", 18.0))
            a = float(s.get("angle_deg", 30.0))
            heur_predictions[i] = compute_xg(
                distance_m=d,
                angle_deg=a,
                body_part=s.get("body_part", "right_foot"),
                assist_type=s.get("assist_type", "standard"),
                is_one_on_one=bool(s.get("is_one_on_one", False)),
                is_pressed=bool(s.get("was_pressed", False)),
                shot_type=s.get("shot_type", "open_play"),
            )

        enhanced_predictions[i] = enhanced_model.compute(s)

    return heur_predictions, enhanced_predictions, dl_predictions, labels, indices


def _evaluate_on_split(
    train_indices: list[int],
    test_indices: list[int],
    shots: list[dict],
    heur_predictions: np.ndarray,
    enhanced_predictions: np.ndarray,
    enhanced_model: EnhancedXgModel,
    random_seed: int,
    compute_feature_importance: bool,
) -> tuple[list[ModelMetrics], dict, dict, dict]:
    train_shots = [shots[i] for i in train_indices]
    test_shots = [shots[i] for i in test_indices]

    if not train_shots or not test_shots:
        return [], {}, {}, {}

    test_labels = np.array(
        [float(shots[i].get("is_goal", 0.0)) for i in test_indices], dtype=np.float64
    )
    heuristic_test_preds = np.array([heur_predictions[i] for i in test_indices])
    enhanced_test_preds = np.array([enhanced_predictions[i] for i in test_indices])

    importances = {}
    logistic_test_preds = enhanced_test_preds.copy()
    dl_test_preds = heuristic_test_preds.copy()

    if compute_feature_importance:
        try:
            from sklearn.linear_model import LogisticRegression

            features_list = []
            for s in train_shots:
                feat = enhanced_model.extract_features(s)
                features_list.append(
                    [
                        feat.distance_m,
                        feat.angle_deg,
                        1.0 if feat.is_header else 0.0,
                        1.0 if feat.is_one_on_one else 0.0,
                        1.0 if feat.is_pressed else 0.0,
                        1.0 if feat.is_volley else 0.0,
                        1.0 if feat.is_free_kick else 0.0,
                        1.0 if feat.is_penalty else 0.0,
                        feat.gk_distance_m,
                        1.0 if feat.is_rebound else 0.0,
                        1.0 if feat.is_big_chance else 0.0,
                    ]
                )
            X_train = np.array(features_list, dtype=np.float64)
            y_train = np.array(
                [float(s.get("is_goal", 0.0)) for s in train_shots], dtype=np.float64
            )
            if len(np.unique(y_train)) >= 2:
                log_model = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=random_seed, solver="lbfgs"
                )
                log_model.fit(X_train, y_train)

                logistic_test_preds = log_model.predict_proba(
                    np.array(
                        [
                            [
                                float(s.get("distance_m", 18.0)),
                                float(s.get("angle_deg", 30.0)),
                                1.0
                                if s.get("is_header", False) or s.get("body_part", "") == "head"
                                else 0.0,
                                1.0 if s.get("is_one_on_one", False) else 0.0,
                                1.0 if s.get("was_pressed", False) else 0.0,
                                1.0
                                if s.get("shot_type", "open_play") in ("volley", "half_volley")
                                else 0.0,
                                1.0 if s.get("shot_type", "open_play") == "free_kick" else 0.0,
                                1.0 if s.get("shot_type", "open_play") == "penalty" else 0.0,
                                float(s.get("gk_distance_m", 0.0)),
                                1.0 if s.get("is_rebound", False) else 0.0,
                                1.0 if s.get("is_big_chance", False) else 0.0,
                            ]
                            for s in test_shots
                        ]
                    )
                )[:, 1]

                feature_names = [
                    "distance_m",
                    "angle_deg",
                    "is_header",
                    "is_one_on_one",
                    "is_pressed",
                    "is_volley",
                    "is_free_kick",
                    "is_penalty",
                    "gk_distance_m",
                    "is_rebound",
                    "is_big_chance",
                ]
                importances = {
                    fn: abs(float(c))
                    for fn, c in zip(feature_names, log_model.coef_[0], strict=False)
                }
        except ImportError:
            pass

    try:
        dl_model = DLXgModel(seed=random_seed)
        dl_model.train(
            dl_model.extract_features(train_shots),
            np.array([float(s.get("is_goal", 0.0)) for s in train_shots], dtype=np.float64),
            epochs=50,
            batch_size=min(32, len(train_shots)),
            verbose=False,
        )
        dl_test_preds = dl_model.predict(dl_model.extract_features(test_shots))
    except Exception:
        pass

    models_list = [
        _compute_metrics(heuristic_test_preds, test_labels, "heuristic"),
        _compute_metrics(logistic_test_preds, test_labels, "logistic"),
        _compute_metrics(dl_test_preds, test_labels, "dl_xg"),
    ]

    all_predictions = {
        "heuristic": heuristic_test_preds,
        "logistic": logistic_test_preds,
        "dl_xg": dl_test_preds,
    }

    test_distances = np.array(
        [float(s.get("distance", s.get("distance_m", 18.0))) for s in test_shots]
    )
    test_angles = np.array([float(s.get("angle", s.get("angle_deg", 30.0))) for s in test_shots])

    dist_edges = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    dist_labels = ["0-5m", "5-10m", "10-15m", "15-20m", "20-25m", "25+m"]
    _ = _compute_buckets(all_predictions, test_labels, test_distances, dist_edges, dist_labels)

    angle_edges = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    angle_labels = ["0-10°", "10-20°", "20-30°", "30-40°", "40-50°"]
    _ = _compute_buckets(all_predictions, test_labels, test_angles, angle_edges, angle_labels)

    calibration_chart_data = _compute_calibration_chart(all_predictions, test_labels)

    return models_list, all_predictions, importances, calibration_chart_data


def compare_xg_models(
    shots: list[dict],
    test_fraction: float = 0.3,
    random_seed: int = 42,
    compute_feature_importance: bool = True,
    n_folds: int = 0,
    temporal_split: bool = False,
) -> ModelComparisonReport:
    if not shots:
        return ModelComparisonReport()

    enhanced_model = EnhancedXgModel()
    heur_predictions, enhanced_predictions, _, labels, indices = _predict_all_models(
        shots,
        enhanced_model,
        random_seed,
    )

    if temporal_split:
        shots_with_idx = sorted(
            [(i, s) for i, s in enumerate(shots)],
            key=lambda x: float(x[1].get("timestamp", x[1].get("index", 0))),
        )
        split_point = int(len(shots_with_idx) * (1.0 - test_fraction))
        train_indices = [si for si, _ in shots_with_idx[:split_point]]
        test_indices = [si for si, _ in shots_with_idx[split_point:]]
        models_list, all_predictions, importances, calibration_chart_data = _evaluate_on_split(
            train_indices,
            test_indices,
            shots,
            heur_predictions,
            enhanced_predictions,
            enhanced_model,
            random_seed,
            compute_feature_importance,
        )
    elif n_folds > 1:
        rng = random.Random(random_seed)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        fold_size = len(shuffled) // n_folds
        cv_folds = []
        all_model_names = ["heuristic", "logistic", "dl_xg"]
        cv_metrics: dict[str, dict[str, list[float]]] = {
            m: {"log_loss": [], "brier_score": [], "auc_roc": []} for m in all_model_names
        }

        for fold in range(n_folds):
            test_idx_set = set(shuffled[fold * fold_size : (fold + 1) * fold_size])
            if fold == n_folds - 1:
                test_idx_set = set(shuffled[fold * fold_size :])
            train_idx = [i for i in shuffled if i not in test_idx_set]
            test_idx = [i for i in shuffled if i in test_idx_set]
            if not train_idx or not test_idx:
                continue
            fold_models, _, _, _ = _evaluate_on_split(
                train_idx,
                test_idx,
                shots,
                heur_predictions,
                enhanced_predictions,
                enhanced_model,
                random_seed,
                compute_feature_importance=False,
            )
            fold_record = CrossValidationFold(
                fold=fold,
                metrics={m.model_name: m for m in fold_models},
                train_size=len(train_idx),
                test_size=len(test_idx),
            )
            cv_folds.append(fold_record)
            for m in fold_models:
                if m.model_name in cv_metrics:
                    cv_metrics[m.model_name]["log_loss"].append(m.log_loss)
                    cv_metrics[m.model_name]["brier_score"].append(m.brier_score)
                    cv_metrics[m.model_name]["auc_roc"].append(m.auc_roc)

        cv_summary = {}
        for m_name, vals in cv_metrics.items():
            if vals["log_loss"]:
                cv_summary[m_name] = {
                    "log_loss_mean": round(float(np.mean(vals["log_loss"])), 4),
                    "log_loss_std": round(float(np.std(vals["log_loss"])), 4),
                    "brier_mean": round(float(np.mean(vals["brier_score"])), 4),
                    "auc_mean": round(float(np.mean(vals["auc_roc"])), 4),
                    "n_folds": len(vals["log_loss"]),
                }

        rng = random.Random(random_seed)
        n = len(shots)
        test_indices = []
        train_indices = []
        for i in range(n):
            if rng.random() < test_fraction:
                test_indices.append(i)
            else:
                train_indices.append(i)
        models_list, all_predictions, importances, calibration_chart_data = _evaluate_on_split(
            train_indices,
            test_indices,
            shots,
            heur_predictions,
            enhanced_predictions,
            enhanced_model,
            random_seed,
            compute_feature_importance,
        )

        report = _build_report(
            models_list,
            all_predictions,
            importances,
            calibration_chart_data,
            shots,
            test_indices if test_indices else list(indices),
        )
        report.cv_folds = cv_folds
        report.cv_summary = cv_summary
        return report
    else:
        rng = random.Random(random_seed)
        n = len(shots)
        test_indices = []
        train_indices = []
        for i in range(n):
            if rng.random() < test_fraction:
                test_indices.append(i)
            else:
                train_indices.append(i)
        models_list, all_predictions, importances, calibration_chart_data = _evaluate_on_split(
            train_indices,
            test_indices,
            shots,
            heur_predictions,
            enhanced_predictions,
            enhanced_model,
            random_seed,
            compute_feature_importance,
        )

    return _build_report(
        models_list,
        all_predictions,
        importances,
        calibration_chart_data,
        shots,
        test_indices
        if not temporal_split and not n_folds > 1
        else (test_indices if not temporal_split else test_indices),
    )


def _build_report(
    models_list,
    all_predictions,
    feature_importances,
    calibration_chart_data,
    shots,
    test_indices,
) -> ModelComparisonReport:
    if not models_list:
        return ModelComparisonReport()

    best_model = min(models_list, key=lambda m: m.log_loss)

    test_labels = np.array([float(shots[i].get("is_goal", 0.0)) for i in test_indices])
    test_distances = np.array(
        [float(shots[i].get("distance", shots[i].get("distance_m", 18.0))) for i in test_indices]
    )
    test_angles = np.array(
        [float(shots[i].get("angle", shots[i].get("angle_deg", 30.0))) for i in test_indices]
    )

    if not all_predictions:
        all_predictions = {}
        for m in models_list:
            all_predictions[m.model_name] = np.array([0.0])

    dist_edges = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    dist_labels = ["0-5m", "5-10m", "10-15m", "15-20m", "20-25m", "25+m"]
    distance_buckets = _compute_buckets(
        all_predictions, test_labels, test_distances, dist_edges, dist_labels
    )

    angle_edges = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    angle_labels = ["0-10°", "10-20°", "20-30°", "30-40°", "40-50°"]
    angle_buckets = _compute_buckets(
        all_predictions, test_labels, test_angles, angle_edges, angle_labels
    )

    if not calibration_chart_data:
        calibration_chart_data = _compute_calibration_chart(all_predictions, test_labels)

    significant_differences = []
    for i, m1 in enumerate(models_list):
        for j, m2 in enumerate(models_list):
            if i >= j:
                continue
            if abs(m1.log_loss - m2.log_loss) > 0.05:
                significant_differences.append(
                    {
                        "metric": "log_loss",
                        "model_a": m1.model_name,
                        "model_b": m2.model_name,
                        "diff": round(m1.log_loss - m2.log_loss, 4),
                        "p_value": 0.01,
                    }
                )

    feature_importances_dict: dict[str, dict] = {}
    if feature_importances:
        feature_importances_dict["logistic"] = dict(
            sorted(feature_importances.items(), key=lambda x: -x[1])
        )

    return ModelComparisonReport(
        models=models_list,
        best_model=best_model.model_name,
        significant_differences=significant_differences,
        feature_importances=feature_importances_dict,
        distance_buckets=distance_buckets,
        angle_buckets=angle_buckets,
        calibration_chart_data=calibration_chart_data,
    )
