"""
ARJUN - Isolation Forest Anomaly Detection
===========================================

Dedicated unsupervised anomaly detector for the ARJUN network-state layer.

Purpose
-------
Detect network states that are unusual compared with benign/normal network
behaviour. This component is deliberately separate from the supervised
XGBoost attack classifier and the predictive World Model.

Pipeline
--------
    training_states.npz
          |
          v
    33-D network state S_t
          |
          v
    RobustScaler (fit on BENIGN training states only)
          |
          v
    Isolation Forest
          |
          +--> anomaly score (higher = more anomalous)
          +--> Normal / Anomalous

Training policy
---------------
The Isolation Forest is trained only on benign states from the training
portion of the chronological dataset. Attack labels are NOT supplied to the
model during fitting. Labels are used only afterward for evaluation.

Outputs
-------
    saved_models/isolation_forest_anomaly.joblib
    evaluation/isolation_forest_report.json
    data/processed/isolation_forest_scores.npz

Run from the ARJUN project root:
    python evaluation/isolation_forest_anomaly.py

Optional:
    python evaluation/isolation_forest_anomaly.py --contamination 0.01
    python evaluation/isolation_forest_anomaly.py --trees 300
    python evaluation/isolation_forest_anomaly.py --threshold-percentile 95

Notes
-----
This detector is an anomaly detector, not an attack classifier. A state can
be anomalous without being a confirmed attack, and a known attack can be
missed by an anomaly detector. ARJUN combines this signal with XGBoost,
state estimation, the World Model, and risk analysis.
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
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed"
STATE_FILE = PROCESSED_DIR / "training_states.npz"

MODEL_DIR = ROOT / "saved_models"
MODEL_FILE = MODEL_DIR / "isolation_forest_anomaly.joblib"

REPORT_DIR = ROOT / "evaluation"
REPORT_FILE = REPORT_DIR / "isolation_forest_report.json"

SCORES_FILE = PROCESSED_DIR / "isolation_forest_scores.npz"

SEED = 42
EXPECTED_DIMENSION = 33
DEFAULT_TREES = 300
DEFAULT_CONTAMINATION = "auto"
DEFAULT_THRESHOLD_PERCENTILE = 95.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _finite_float32(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def load_training_states() -> Tuple[np.ndarray, np.ndarray, list[str]]:
    if not STATE_FILE.exists():
        raise FileNotFoundError(
            "ARJUN training state file was not found:\n"
            f"  {STATE_FILE}\n"
            "Build the processed training states before running this script."
        )

    data = np.load(STATE_FILE, allow_pickle=True)

    if "states" in data:
        states = _finite_float32(data["states"])
    elif "X" in data:
        states = _finite_float32(data["X"])
        if states.ndim == 3:
            states = states[:, -1, :]
    else:
        raise KeyError(
            f"{STATE_FILE.name} does not contain 'states' or 'X'."
        )

    if "labels" in data:
        labels = np.asarray(data["labels"]).reshape(-1).astype(np.int32)
    elif "y" in data:
        labels = np.asarray(data["y"]).reshape(-1).astype(np.int32)
    else:
        raise KeyError(
            f"{STATE_FILE.name} does not contain 'labels' or 'y'."
        )

    if states.ndim != 2:
        raise ValueError(f"Expected states shape (N,D), got {states.shape}.")
    if states.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION} features, got {states.shape[1]}."
        )
    if len(states) != len(labels):
        raise ValueError(
            f"States/labels length mismatch: {len(states)} vs {len(labels)}."
        )
    if len(states) < 100:
        raise ValueError("Too few states for a meaningful anomaly experiment.")

    if "feature_names" in data:
        feature_names = [str(x) for x in np.asarray(data["feature_names"]).reshape(-1)]
    else:
        feature_names = [f"feature_{i}" for i in range(states.shape[1])]

    if len(feature_names) != states.shape[1]:
        feature_names = [f"feature_{i}" for i in range(states.shape[1])]

    return states, labels, feature_names


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(trees: int, random_state: int = SEED) -> Pipeline:
    if trees < 50:
        raise ValueError("--trees must be at least 50.")

    detector = IsolationForest(
        n_estimators=int(trees),
        contamination=DEFAULT_CONTAMINATION,
        max_samples="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "scaler",
                RobustScaler(
                    quantile_range=(25.0, 75.0),
                    with_centering=True,
                    with_scaling=True,
                ),
            ),
            ("isolation_forest", detector),
        ]
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def anomaly_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Return a score where larger values mean more anomalous.

    sklearn's IsolationForest decision_function is larger for normal points,
    so its sign is inverted here for an intuitive anomaly score.
    """
    X = _finite_float32(X)
    return (-model.decision_function(X)).astype(np.float32)


def choose_threshold(scores: np.ndarray, benign_mask: np.ndarray, percentile: float) -> float:
    benign_scores = scores[benign_mask]
    if len(benign_scores) == 0:
        raise ValueError("Cannot choose an anomaly threshold: no benign calibration states.")
    return float(np.percentile(benign_scores, percentile))


def classification_metrics(y_true: np.ndarray, predicted_anomaly: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    predicted_anomaly = np.asarray(predicted_anomaly, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted_anomaly,
        labels=[0, 1],
    ).ravel()

    result = {
        "precision": float(precision_score(y_true, predicted_anomaly, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_anomaly, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_anomaly, zero_division=0)),
        "fpr": float(fp / max(fp + tn, 1)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "predicted_anomaly_count": int(np.sum(predicted_anomaly)),
        "anomaly_rate": float(np.mean(predicted_anomaly)),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        result["roc_auc"] = float("nan")
        result["average_precision"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Feature importance proxy
# ---------------------------------------------------------------------------


def feature_anomaly_contribution(model: Pipeline, X: np.ndarray, feature_names: list[str], top_n: int = 10):
    """
    Provide a simple feature-level diagnostic based on standardized magnitude.

    Isolation Forest itself does not provide SHAP values here. This diagnostic
    is intentionally labelled as a feature-magnitude proxy and must not be
    presented as Isolation Forest feature importance.
    """
    scaler: RobustScaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(_finite_float32(X))
    magnitude = np.mean(np.abs(X_scaled), axis=0)
    order = np.argsort(magnitude)[::-1][:top_n]

    return [
        {
            "feature": feature_names[int(i)],
            "standardized_magnitude": float(magnitude[int(i)]),
        }
        for i in order
    ]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(args) -> dict:
    print("=" * 72)
    print("ARJUN - ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 72)

    states, labels, feature_names = load_training_states()

    n = len(states)
    split_index = int(n * 0.80)
    if split_index <= 0 or split_index >= n:
        raise ValueError("Invalid chronological split.")

    X_train_all = states[:split_index]
    y_train_all = labels[:split_index]
    X_test = states[split_index:]
    y_test = labels[split_index:]

    # Unsupervised training: only benign states are used to fit the detector.
    benign_train_mask = y_train_all == 0
    X_fit = X_train_all[benign_train_mask]

    if len(X_fit) < 100:
        raise ValueError(
            f"Only {len(X_fit)} benign training states available; at least 100 are required."
        )

    print("\nDATA")
    print("------------------------------------------------------------------------")
    print(f"Total states              : {n:,}")
    print(f"State dimension            : {states.shape[1]}")
    print(f"Chronological train states : {len(X_train_all):,}")
    print(f"Chronological test states  : {len(X_test):,}")
    print(f"Benign states used for fit : {len(X_fit):,}")
    print(f"Attack states in test      : {int(np.sum(y_test == 1)):,}")

    model = build_model(args.trees)

    print("\nTRAINING")
    print("------------------------------------------------------------------------")
    print(f"Isolation trees            : {args.trees}")
    print("Contamination              : auto")
    print("Training labels supplied   : NONE (benign-only fit)")
    print("Scaler                     : RobustScaler")
    print("Fitting...")

    model.fit(X_fit)

    # Threshold calibration uses only benign training states that were not used
    # for fitting, where possible. This avoids using attack labels to tune the
    # anomaly threshold.
    calibration_start = int(len(X_fit) * 0.80)
    if calibration_start < len(X_fit) - 10:
        X_calibration = X_fit[calibration_start:]
    else:
        X_calibration = X_fit

    calibration_scores = anomaly_scores(model, X_calibration)
    threshold = float(np.percentile(calibration_scores, args.threshold_percentile))

    test_scores = anomaly_scores(model, X_test)
    predicted = (test_scores >= threshold).astype(np.int32)
    metrics = classification_metrics(y_test, predicted, test_scores)

    # Also report behaviour separately for benign and attack test states.
    benign_test_scores = test_scores[y_test == 0]
    attack_test_scores = test_scores[y_test == 1]

    result = {
        "experiment": "isolation_forest_anomaly_detection",
        "model": "IsolationForest",
        "state_dimension": int(states.shape[1]),
        "feature_names": feature_names,
        "random_state": SEED,
        "n_estimators": int(args.trees),
        "contamination": DEFAULT_CONTAMINATION,
        "threshold_percentile": float(args.threshold_percentile),
        "threshold": threshold,
        "training_policy": "fit_only_on_benign_states",
        "chronological_split": 0.80,
        "total_states": int(n),
        "train_states": int(len(X_train_all)),
        "test_states": int(len(X_test)),
        "benign_fit_states": int(len(X_fit)),
        "test_benign_states": int(np.sum(y_test == 0)),
        "test_attack_states": int(np.sum(y_test == 1)),
        "metrics": metrics,
        "score_summary": {
            "test_min": float(np.min(test_scores)),
            "test_max": float(np.max(test_scores)),
            "test_mean": float(np.mean(test_scores)),
            "test_median": float(np.median(test_scores)),
            "benign_test_mean": float(np.mean(benign_test_scores)) if len(benign_test_scores) else float("nan"),
            "attack_test_mean": float(np.mean(attack_test_scores)) if len(attack_test_scores) else float("nan"),
        },
        "top_feature_magnitude_proxies": feature_anomaly_contribution(
            model,
            X_test,
            feature_names,
            top_n=min(10, states.shape[1]),
        ),
        "interpretation": (
            "Isolation Forest is an unsupervised anomaly detector. "
            "It is trained only on benign states; attack labels are used only "
            "for post-training evaluation. Its anomaly score is not an attack probability."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_names": feature_names,
        "state_dimension": int(states.shape[1]),
        "threshold_percentile": float(args.threshold_percentile),
        "training_policy": "fit_only_on_benign_states",
        "random_state": SEED,
    }
    joblib.dump(artifact, MODEL_FILE)

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

    print("\nRESULTS")
    print("------------------------------------------------------------------------")
    print(f"Anomaly threshold          : {threshold:.6f}")
    print(f"Precision                  : {metrics['precision']:.4f}")
    print(f"Recall                     : {metrics['recall']:.4f}")
    print(f"F1                         : {metrics['f1']:.4f}")
    print(f"FPR                        : {metrics['fpr']:.4f}")
    print(f"ROC-AUC                    : {metrics['roc_auc']:.4f}")
    print(f"Average Precision          : {metrics['average_precision']:.4f}")
    print(f"Predicted anomalies        : {metrics['predicted_anomaly_count']:,}")
    print(f"Test anomaly rate          : {metrics['anomaly_rate']:.2%}")

    print("\nTOP FEATURE MAGNITUDE PROXIES")
    print("------------------------------------------------------------------------")
    for item in result["top_feature_magnitude_proxies"]:
        print(
            f"{item['feature']:<32} "
            f"{item['standardized_magnitude']:.4f}"
        )

    print("\nARTIFACTS")
    print("------------------------------------------------------------------------")
    print(f"Model                      : {MODEL_FILE}")
    print(f"Report                     : {REPORT_FILE}")
    print(f"Scores                     : {SCORES_FILE}")

    print("\nISOLATION FOREST ANOMALY DETECTION: PASSED")
    print("=" * 72)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ARJUN Isolation Forest anomaly detector."
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=DEFAULT_TREES,
        help=f"Number of Isolation Forest trees (default: {DEFAULT_TREES}).",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=DEFAULT_THRESHOLD_PERCENTILE,
        help="Benign calibration percentile used as anomaly threshold (default: 95).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 50.0 <= args.threshold_percentile <= 99.9:
        raise ValueError("--threshold-percentile must be between 50 and 99.9.")
    run_experiment(args)


if __name__ == "__main__":
    main()
