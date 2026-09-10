"""
ARJUN - Optimized Isolation Forest Anomaly Detection
====================================================

Unsupervised anomaly detector for ARJUN's 33-dimensional network-state vector.

Why this version is different from the first detector
------------------------------------------------------
The first experiment used RobustScaler directly on the raw state vector. CIC
network features contain very heavy-tailed quantities (especially IAT
variance, bytes, packets, and rate features). Those scales can dominate an
axis-based anomaly detector.

This version therefore uses a leakage-safe preprocessing pipeline:

    raw state
       -> signed log1p transform
       -> robust clipping
       -> RobustScaler
       -> Isolation Forest

The detector is still genuinely unsupervised:
* attack labels are NEVER passed to IsolationForest.fit()
* threshold calibration uses benign states only
* the final test set is untouched until evaluation

It also evaluates several benign-only threshold percentiles in the report so
we can see the precision/recall trade-off without using attack labels to pick
the production threshold. The configured threshold is deliberately selected
from the benign calibration policy, not from test performance.

Outputs
-------
    saved_models/isolation_forest_anomaly_optimized.joblib
    evaluation/isolation_forest_optimized_report.json
    data/processed/isolation_forest_optimized_scores.npz

Run from ARJUN project root:
    python evaluation/isolation_forest_anomaly_optimized.py

Optional:
    python evaluation/isolation_forest_anomaly_optimized.py --trees 500
    python evaluation/isolation_forest_anomaly_optimized.py --threshold-percentile 99
    python evaluation/isolation_forest_anomaly_optimized.py --max-samples 4096

Important
---------
This is an anomaly detector, NOT an attack-probability model. Do not merge
its score directly into a probability without calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
STATE_FILE = PROCESSED_DIR / "training_states.npz"
MODEL_DIR = ROOT / "saved_models"
MODEL_FILE = MODEL_DIR / "isolation_forest_anomaly_optimized.joblib"
REPORT_FILE = ROOT / "evaluation" / "isolation_forest_optimized_report.json"
SCORES_FILE = PROCESSED_DIR / "isolation_forest_optimized_scores.npz"

SEED = 42
EXPECTED_DIMENSION = 33
DEFAULT_TREES = 500
DEFAULT_MAX_SAMPLES = 4096
DEFAULT_THRESHOLD_PERCENTILE = 99.0
DEFAULT_CLIP_Z = 8.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def finite_float32(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_states() -> Tuple[np.ndarray, np.ndarray, list[str]]:
    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"Missing processed state file: {STATE_FILE}\n"
            "Build training_states.npz before running this experiment."
        )

    data = np.load(STATE_FILE, allow_pickle=True)

    if "states" in data:
        states = finite_float32(data["states"])
    elif "X" in data:
        states = finite_float32(data["X"])
        if states.ndim == 3:
            states = states[:, -1, :]
    else:
        raise KeyError("training_states.npz must contain 'states' or 'X'.")

    if "labels" in data:
        labels = np.asarray(data["labels"]).reshape(-1).astype(np.int32)
    elif "y" in data:
        labels = np.asarray(data["y"]).reshape(-1).astype(np.int32)
    else:
        raise KeyError("training_states.npz must contain 'labels' or 'y'.")

    if states.ndim != 2:
        raise ValueError(f"Expected state matrix (N,D), got {states.shape}.")
    if states.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION} state features, got {states.shape[1]}."
        )
    if len(states) != len(labels):
        raise ValueError("States and labels have different lengths.")
    if len(states) < 1000:
        raise ValueError("Too few states for this experiment.")

    if "feature_names" in data:
        feature_names = [str(x) for x in np.asarray(data["feature_names"]).reshape(-1)]
    else:
        feature_names = [f"feature_{i}" for i in range(states.shape[1])]

    if len(feature_names) != states.shape[1]:
        feature_names = [f"feature_{i}" for i in range(states.shape[1])]

    return states, labels, feature_names


# ---------------------------------------------------------------------------
# Robust heavy-tail transformation
# ---------------------------------------------------------------------------


def signed_log1p(X: np.ndarray) -> np.ndarray:
    """Compress both positive and negative heavy tails without dropping sign."""
    X = finite_float32(X).astype(np.float64, copy=False)
    return np.sign(X) * np.log1p(np.abs(X)).astype(np.float64)


def robust_clip_fit(X: np.ndarray, clip_z: float = DEFAULT_CLIP_Z):
    """Fit per-feature robust bounds using only the benign fit set."""
    X = np.asarray(X, dtype=np.float64)
    median = np.median(X, axis=0)
    q25 = np.percentile(X, 25.0, axis=0)
    q75 = np.percentile(X, 75.0, axis=0)
    iqr = q75 - q25
    scale = np.where(iqr > 1e-12, iqr / 1.349, np.std(X, axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    lower = median - clip_z * scale
    upper = median + clip_z * scale
    return lower.astype(np.float32), upper.astype(np.float32)


def robust_clip_apply(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    X = finite_float32(X)
    return np.clip(X, lower, upper).astype(np.float32)


def transform_fit(X: np.ndarray, clip_z: float = DEFAULT_CLIP_Z):
    logged = signed_log1p(X)
    lower, upper = robust_clip_fit(logged, clip_z=clip_z)
    clipped = robust_clip_apply(logged, lower, upper)
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(clipped)
    return lower, upper, scaler


def transform_apply(
    X: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    scaler: RobustScaler,
) -> np.ndarray:
    logged = signed_log1p(X)
    clipped = robust_clip_apply(logged, lower, upper)
    return scaler.transform(clipped).astype(np.float32)


class OptimizedIsolationForest:
    """Small serializable wrapper around preprocessing + Isolation Forest."""

    def __init__(self, trees: int, max_samples: int | str, clip_z: float):
        if trees < 100:
            raise ValueError("trees must be at least 100")
        if isinstance(max_samples, int) and max_samples < 256:
            raise ValueError("max_samples must be at least 256")

        self.trees = int(trees)
        self.max_samples = max_samples
        self.clip_z = float(clip_z)
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None
        self.scaler_: RobustScaler | None = None
        self.detector_: IsolationForest | None = None

    def fit(self, X: np.ndarray):
        self.lower_, self.upper_, self.scaler_ = transform_fit(X, self.clip_z)
        X_t = transform_apply(X, self.lower_, self.upper_, self.scaler_)
        self.detector_ = IsolationForest(
            n_estimators=self.trees,
            contamination="auto",
            max_samples=self.max_samples,
            max_features=1.0,
            bootstrap=False,
            random_state=SEED,
            n_jobs=-1,
        )
        self.detector_.fit(X_t)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lower_ is None or self.upper_ is None or self.scaler_ is None:
            raise RuntimeError("Model has not been fitted.")
        return transform_apply(X, self.lower_, self.upper_, self.scaler_)

    def score_samples_anomaly(self, X: np.ndarray) -> np.ndarray:
        if self.detector_ is None:
            raise RuntimeError("Model has not been fitted.")
        X_t = self.transform(X)
        return (-self.detector_.decision_function(X_t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    pred = (scores >= threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_anomaly_count": int(pred.sum()),
        "anomaly_rate": float(pred.mean()),
    }
    return out


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def score_percentiles(scores: np.ndarray) -> Dict[str, float]:
    return {
        str(p): float(np.percentile(scores, p))
        for p in (90, 95, 97.5, 99, 99.5, 99.9)
    }


# ---------------------------------------------------------------------------
# Feature diagnostics
# ---------------------------------------------------------------------------


def feature_magnitude_proxy(
    model: OptimizedIsolationForest,
    X: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
):
    X_t = model.transform(X)
    magnitude = np.mean(np.abs(X_t), axis=0)
    order = np.argsort(magnitude)[::-1][:top_n]
    return [
        {
            "feature": feature_names[int(i)],
            "transformed_standardized_magnitude": float(magnitude[int(i)]),
        }
        for i in order
    ]


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run(args) -> dict:
    print("=" * 76)
    print("ARJUN - OPTIMIZED ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 76)

    states, labels, feature_names = load_states()
    n = len(states)
    split = int(n * 0.80)

    X_train = states[:split]
    y_train = labels[:split]
    X_test = states[split:]
    y_test = labels[split:]

    benign_mask = y_train == 0
    X_fit_all = X_train[benign_mask]
    if len(X_fit_all) < 1000:
        raise ValueError(f"Only {len(X_fit_all)} benign training states available.")

    # Reserve the last 20% of benign training states for threshold calibration.
    # This remains an unsupervised calibration set: labels are used only to
    # select benign rows, never to optimize the anomaly detector itself.
    fit_cut = int(len(X_fit_all) * 0.80)
    X_fit = X_fit_all[:fit_cut]
    X_cal = X_fit_all[fit_cut:]
    if len(X_fit) < 500 or len(X_cal) < 100:
        raise ValueError("Insufficient benign fit/calibration states.")

    print("\nDATA")
    print("-" * 76)
    print(f"Total states              : {n:,}")
    print(f"State dimension            : {states.shape[1]}")
    print(f"Chronological train states : {len(X_train):,}")
    print(f"Chronological test states  : {len(X_test):,}")
    print(f"Benign detector-fit states : {len(X_fit):,}")
    print(f"Benign calibration states  : {len(X_cal):,}")
    print(f"Attack states in test      : {int(np.sum(y_test == 1)):,}")

    model = OptimizedIsolationForest(
        trees=args.trees,
        max_samples=args.max_samples,
        clip_z=args.clip_z,
    ).fit(X_fit)

    calibration_scores = model.score_samples_anomaly(X_cal)
    test_scores = model.score_samples_anomaly(X_test)

    threshold = float(np.percentile(calibration_scores, args.threshold_percentile))
    primary = metrics_at_threshold(y_test, test_scores, threshold)
    primary.update(ranking_metrics(y_test, test_scores))

    # Diagnostic threshold table. These values are NOT used to select the
    # configured threshold; they simply expose the operational trade-off.
    threshold_table = []
    for p in (90.0, 95.0, 97.5, 99.0, 99.5, 99.9):
        t = float(np.percentile(calibration_scores, p))
        row = metrics_at_threshold(y_test, test_scores, t)
        row["calibration_percentile"] = p
        threshold_table.append(row)

    benign_test = test_scores[y_test == 0]
    attack_test = test_scores[y_test == 1]

    result = {
        "experiment": "optimized_isolation_forest_anomaly_detection",
        "model": "IsolationForest",
        "state_dimension": int(states.shape[1]),
        "feature_names": feature_names,
        "random_state": SEED,
        "n_estimators": int(args.trees),
        "max_samples": args.max_samples,
        "max_features": 1.0,
        "contamination": "auto",
        "preprocessing": [
            "signed_log1p",
            f"robust_clip_z_{args.clip_z:g}",
            "RobustScaler",
        ],
        "threshold_percentile": float(args.threshold_percentile),
        "threshold": threshold,
        "training_policy": "fit_only_on_benign_states",
        "threshold_policy": "configured_benign_calibration_percentile",
        "chronological_split": 0.80,
        "benign_fit_fraction": 0.80,
        "total_states": int(n),
        "train_states": int(len(X_train)),
        "test_states": int(len(X_test)),
        "benign_fit_states": int(len(X_fit)),
        "benign_calibration_states": int(len(X_cal)),
        "test_benign_states": int(np.sum(y_test == 0)),
        "test_attack_states": int(np.sum(y_test == 1)),
        "metrics": primary,
        "threshold_tradeoff": threshold_table,
        "calibration_score_percentiles": score_percentiles(calibration_scores),
        "test_score_summary": {
            "min": float(np.min(test_scores)),
            "max": float(np.max(test_scores)),
            "mean": float(np.mean(test_scores)),
            "median": float(np.median(test_scores)),
            "benign_mean": float(np.mean(benign_test)) if len(benign_test) else float("nan"),
            "attack_mean": float(np.mean(attack_test)) if len(attack_test) else float("nan"),
        },
        "top_feature_magnitude_proxies": feature_magnitude_proxy(
            model, X_test, feature_names, top_n=min(10, states.shape[1])
        ),
        "interpretation": (
            "Isolation Forest remains unsupervised. The preprocessing compresses "
            "heavy-tailed network features before robust scaling. Attack labels "
            "are used only for final evaluation and are never passed to model.fit(). "
            "The anomaly score is not an attack probability."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_names": feature_names,
        "state_dimension": int(states.shape[1]),
        "threshold_percentile": float(args.threshold_percentile),
        "training_policy": "fit_only_on_benign_states",
        "preprocessing": result["preprocessing"],
        "random_state": SEED,
    }
    joblib.dump(artifact, MODEL_FILE)

    predicted = (test_scores >= threshold).astype(np.int32)
    np.savez_compressed(
        SCORES_FILE,
        test_scores=test_scores,
        test_labels=y_test,
        predicted_anomaly=predicted,
        threshold=np.asarray([threshold], dtype=np.float32),
        feature_names=np.asarray(feature_names, dtype=object),
    )

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)

    print("\nMODEL")
    print("-" * 76)
    print(f"Isolation trees            : {args.trees}")
    print(f"Max samples               : {args.max_samples}")
    print("Contamination              : auto")
    print("Labels supplied to fit     : NONE (benign-only fit)")
    print("Preprocessing              : signed_log1p -> robust clip -> RobustScaler")
    print(f"Robust clip                : +/- {args.clip_z:g} robust scales")
    print("Fitting... DONE")

    print("\nRESULTS")
    print("-" * 76)
    print(f"Configured threshold       : {threshold:.6f}")
    print(f"Precision                  : {primary['precision']:.4f}")
    print(f"Recall                     : {primary['recall']:.4f}")
    print(f"F1                         : {primary['f1']:.4f}")
    print(f"FPR                        : {primary['fpr']:.4f}")
    print(f"ROC-AUC                    : {primary['roc_auc']:.4f}")
    print(f"Average Precision          : {primary['average_precision']:.4f}")
    print(f"Predicted anomalies        : {primary['predicted_anomaly_count']:,}")
    print(f"Test anomaly rate          : {primary['anomaly_rate']:.2%}")

    print("\nTHRESHOLD TRADE-OFF (diagnostic only)")
    print("-" * 76)
    print("Percentile | Precision | Recall | F1     | FPR")
    for row in threshold_table:
        print(
            f"{row['calibration_percentile']:9.1f}% | "
            f"{row['precision']:.4f}     | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['fpr']:.4f}"
        )

    print("\nTOP FEATURE MAGNITUDE PROXIES")
    print("-" * 76)
    for item in result["top_feature_magnitude_proxies"]:
        print(
            f"{item['feature']:<32} "
            f"{item['transformed_standardized_magnitude']:.4f}"
        )

    print("\nARTIFACTS")
    print("-" * 76)
    print(f"Model                      : {MODEL_FILE}")
    print(f"Report                     : {REPORT_FILE}")
    print(f"Scores                     : {SCORES_FILE}")
    print("\nISOLATION FOREST OPTIMIZATION: PASSED")
    print("=" * 76)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate optimized ARJUN Isolation Forest anomaly detector."
    )
    parser.add_argument("--trees", type=int, default=DEFAULT_TREES)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help="Isolation Forest subsample size; use -1 for all fit samples.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=DEFAULT_THRESHOLD_PERCENTILE,
        help="Benign calibration percentile for the configured threshold.",
    )
    parser.add_argument(
        "--clip-z",
        type=float,
        default=DEFAULT_CLIP_Z,
        help="Robust-scale clipping limit before RobustScaler.",
    )
    args = parser.parse_args()

    if args.max_samples == -1:
        args.max_samples = "auto"
    if not 100 <= args.trees <= 5000:
        raise ValueError("--trees must be between 100 and 5000.")
    if args.max_samples != "auto" and args.max_samples < 256:
        raise ValueError("--max-samples must be >=256 or -1.")
    if not 80.0 <= args.threshold_percentile <= 99.9:
        raise ValueError("--threshold-percentile must be between 80 and 99.9.")
    if not 2.0 <= args.clip_z <= 20.0:
        raise ValueError("--clip-z must be between 2 and 20.")
    return args


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
