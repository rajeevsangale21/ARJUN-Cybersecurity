"""
ARJUN - TEMPORAL ISOLATION FOREST ANOMALY DETECTION

Uses the existing 10-state sequence artifact and builds a temporal
representation from:
    1. latest state S_t
    2. latest change Delta S_t
    3. 10-state mean
    4. 10-state standard deviation

Training is benign-only and chronological. The anomaly threshold is
calibrated on benign validation data only.

This is an EXPERIMENT. It does not modify the production World Model,
Zeek collector, risk engine, dashboard, or alert pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SEQUENCE_FILE = ROOT / "data" / "processed" / "graph_sequences_states.npz"
DEFAULT_LABEL_FILE = ROOT / "data" / "processed" / "unseen_attack_state_labels.npz"

MODEL_FILE = ROOT / "saved_models" / "temporal_isolation_forest.joblib"
REPORT_FILE = ROOT / "evaluation" / "temporal_isolation_forest_report.json"
SCORES_FILE = ROOT / "data" / "processed" / "temporal_isolation_forest_scores.npz"

FEATURE_NAMES = [
    "flow_count",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_dst_ports",
    "total_bytes",
    "total_packets",
    "avg_duration",
    "avg_packet_length",
    "avg_bytes_per_packet",
    "avg_packets_per_second",
    "avg_bytes_per_second",
    "avg_iat",
    "iat_variance",
    "src_iat_mean",
    "src_iat_variance",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "tcp_window_variance",
    "avg_tcp_window",
    "ttl_variance",
    "avg_ttl",
    "fragment_count",
    "retransmission_count",
    "payload_size_variance",
    "avg_payload_size",
    "port_scan_count",
    "failed_connection_count",
    "inbound_outbound_ratio",
    "bytes_packets_ratio",
]


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Compress heavy-tailed magnitudes while preserving sign."""
    return np.sign(x) * np.log1p(np.abs(x))


def robust_clip(x: np.ndarray, scale_limit: float = 8.0) -> np.ndarray:
    """Clip each feature using training-set median/IQR-derived bounds."""
    med = np.median(x, axis=0)
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    scale = (q3 - q1) / 1.349
    scale = np.where(scale < 1e-9, 1.0, scale)
    lo = med - scale_limit * scale
    hi = med + scale_limit * scale
    return np.clip(x, lo, hi)


def build_temporal_features(X: np.ndarray) -> np.ndarray:
    """
    X: (N, T, D)

    Returns:
        (N, 4D):
          latest, latest_delta, sequence_mean, sequence_std
    """
    if X.ndim != 3:
        raise ValueError(f"Expected (N,T,D), got {X.shape}")
    if X.shape[1] < 2:
        raise ValueError("Need at least two states per sequence.")

    latest = X[:, -1, :]
    previous = X[:, -2, :]
    delta = latest - previous
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)

    return np.concatenate([latest, delta, mean, std], axis=1).astype(np.float32)


def load_sequences(path: Path):
    data = np.load(path, allow_pickle=False)

    if "X" not in data:
        raise KeyError(f"{path} does not contain key 'X'.")

    X = np.asarray(data["X"], dtype=np.float32)

    if "y" in data:
        y_state = np.asarray(data["y"], dtype=np.float32)
    else:
        raise KeyError(f"{path} does not contain target key 'y'.")

    if X.ndim != 3:
        raise ValueError(f"Sequence X must be 3-D, got {X.shape}")

    if y_state.ndim != 2:
        raise ValueError(f"Sequence y must be 2-D, got {y_state.shape}")

    if X.shape[0] != y_state.shape[0]:
        raise ValueError("X/y sequence count mismatch.")

    return X, y_state


def load_target_labels(label_file: Path, n: int) -> np.ndarray:
    """
    Load sequence-level target labels from the existing ARJUN
    unseen-attack preparation artifact.

    Current artifact format:
      target_families : family name for each sequence target

    For compatibility, also accepts:
      target_labels
      labels
      y

    Important:
      train/test masks in this artifact are NOT used here to manufacture
      labels. They remain available for the experiment's leakage-safe
      split logic.
    """
    if not label_file.exists():
        raise FileNotFoundError(
            f"Label file not found: {label_file}\n"
            "This experiment requires sequence-level target labels."
        )

    data = np.load(label_file, allow_pickle=True)

    if "target_families" in data:
        labels = np.asarray(data["target_families"]).reshape(-1)
    else:
        labels = None
        for key in ("target_labels", "labels", "y"):
            if key in data:
                labels = np.asarray(data[key]).reshape(-1)
                break

        if labels is None:
            raise KeyError(
                f"No supported target label key found in {label_file}. "
                f"Available keys: {list(data.keys())}"
            )

    if len(labels) != n:
        raise ValueError(
            f"Label count {len(labels)} does not match sequence count {n}."
        )

    return labels


def binary_attack_labels(raw_labels: np.ndarray) -> np.ndarray:
    """
    Converts the existing labels to benign(0)/attack(1).

    Handles numeric labels and common string labels. For string labels,
    'Benign'/'BENIGN' is benign and everything else is attack.
    """
    if np.issubdtype(raw_labels.dtype, np.number):
        return (raw_labels.astype(float) != 0).astype(np.int8)

    labels = np.asarray(raw_labels).astype(str)
    normalized = np.char.lower(np.char.strip(labels))
    return (
        ~np.isin(normalized, ["benign", "normal", "0", "nan", "none", ""])
    ).astype(np.int8)


def make_chronological_split(n: int):
    """
    80% detector/calibration region, 20% final test.

    Within the first 80%:
      first 80% -> detector fit
      last 20%  -> benign calibration

    This keeps threshold calibration separate from detector fitting.
    """
    train_end = int(n * 0.80)
    fit_end = int(train_end * 0.80)
    return fit_end, train_end


def preprocess_train_fit(X_fit: np.ndarray):
    """
    Fit preprocessing parameters on benign detector-fit samples only.
    """
    transformed = signed_log1p(X_fit)

    med = np.median(transformed, axis=0)
    q1 = np.percentile(transformed, 25, axis=0)
    q3 = np.percentile(transformed, 75, axis=0)
    scale = (q3 - q1) / 1.349
    scale = np.where(scale < 1e-9, 1.0, scale)

    lo = med - 8.0 * scale
    hi = med + 8.0 * scale

    clipped = np.clip(transformed, lo, hi)

    scaler = RobustScaler()
    scaler.fit(clipped)

    params = {
        "median": med.astype(np.float64),
        "scale": scale.astype(np.float64),
        "clip_lo": lo.astype(np.float64),
        "clip_hi": hi.astype(np.float64),
        "scaler": scaler,
    }

    return params


def apply_preprocessing(X: np.ndarray, params) -> np.ndarray:
    transformed = signed_log1p(X)
    clipped = np.clip(transformed, params["clip_lo"], params["clip_hi"])
    return params["scaler"].transform(clipped).astype(np.float32)


def anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    # Higher = more anomalous.
    return (-model.decision_function(X)).astype(np.float64)


def evaluate_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(np.int8)

    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    fpr = fp / max(tn + fp, 1)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "predicted_anomalies": int(pred.sum()),
        "test_anomaly_rate": float(pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def percentile_diagnostics(y_true, scores, benign_cal_scores):
    rows = []

    for pct in (90.0, 95.0, 97.5, 99.0, 99.5, 99.9):
        threshold = float(np.percentile(benign_cal_scores, pct))
        result = evaluate_threshold(y_true, scores, threshold)
        result["percentile"] = pct
        rows.append(result)

    return rows


def temporal_feature_names():
    names = []
    names.extend([f"latest__{x}" for x in FEATURE_NAMES])
    names.extend([f"delta__{x}" for x in FEATURE_NAMES])
    names.extend([f"mean10__{x}" for x in FEATURE_NAMES])
    names.extend([f"std10__{x}" for x in FEATURE_NAMES])
    return names


def magnitude_diagnostics(raw_features: np.ndarray):
    """
    Diagnostic only. This is NOT model feature importance or SHAP.
    """
    names = temporal_feature_names()
    values = np.nanmedian(np.abs(raw_features), axis=0)
    order = np.argsort(values)[::-1][:10]

    return [
        {"feature": names[int(i)], "median_absolute_value": float(values[i])}
        for i in order
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-file",
        type=Path,
        default=DEFAULT_SEQUENCE_FILE,
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=DEFAULT_LABEL_FILE,
    )
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=99.0,
        help="Benign calibration percentile used for configured threshold.",
    )
    args = parser.parse_args()

    print("=" * 76)
    print("ARJUN - TEMPORAL ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 76)

    X_seq, y_state = load_sequences(args.sequence_file)

    n, t, d = X_seq.shape

    if d != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} state features, got {d}."
        )

    raw_labels = load_target_labels(args.label_file, n)
    y_binary = binary_attack_labels(raw_labels)

    temporal_X = build_temporal_features(X_seq)

    fit_end, train_end = make_chronological_split(n)

    X_fit = temporal_X[:fit_end]
    X_cal = temporal_X[fit_end:train_end]
    X_test = temporal_X[train_end:]

    y_fit = y_binary[:fit_end]
    y_cal = y_binary[fit_end:train_end]
    y_test = y_binary[train_end:]

    benign_fit = X_fit[y_fit == 0]
    benign_cal = X_cal[y_cal == 0]

    if len(benign_fit) == 0:
        raise RuntimeError("No benign samples available for detector fitting.")
    if len(benign_cal) == 0:
        raise RuntimeError("No benign samples available for threshold calibration.")
    if len(np.unique(y_test)) < 2:
        raise RuntimeError("Final test set must contain both benign and attack states.")

    params = preprocess_train_fit(benign_fit)

    X_fit_proc = apply_preprocessing(benign_fit, params)
    X_cal_proc = apply_preprocessing(benign_cal, params)
    X_test_proc = apply_preprocessing(X_test, params)

    model = IsolationForest(
        n_estimators=args.trees,
        contamination="auto",
        max_samples=min(args.max_samples, len(X_fit_proc)),
        max_features=1.0,
        n_jobs=-1,
        random_state=args.random_state,
    )

    print()
    print("DATA")
    print("-" * 76)
    print(f"Sequences                   : {n:,}")
    print(f"Sequence length             : {t}")
    print(f"State dimension             : {d}")
    print(f"Temporal feature dimension  : {temporal_X.shape[1]}")
    print(f"Detector-fit sequences      : {fit_end:,}")
    print(f"Calibration sequences       : {train_end - fit_end:,}")
    print(f"Final test sequences        : {n - train_end:,}")
    print(f"Benign detector-fit samples : {len(benign_fit):,}")
    print(f"Benign calibration samples  : {len(benign_cal):,}")
    print(f"Attack samples in test      : {int(y_test.sum()):,}")

    print()
    print("MODEL")
    print("-" * 76)
    print(f"Isolation trees             : {args.trees}")
    print(f"Max samples                 : {min(args.max_samples, len(X_fit_proc)):,}")
    print("Contamination               : auto")
    print("Labels supplied to fit      : NONE (benign-only fit)")
    print("Temporal representation     : latest + delta + mean10 + std10")
    print("Preprocessing               : signed_log1p -> robust clip -> RobustScaler")
    print("Robust clip                 : +/- 8 robust scales")
    print("Fitting...", end=" ")

    model.fit(X_fit_proc)
    print("DONE")

    cal_scores = anomaly_scores(model, X_cal_proc)
    test_scores = anomaly_scores(model, X_test_proc)

    threshold = float(np.percentile(cal_scores, args.threshold_percentile))
    configured = evaluate_threshold(y_test, test_scores, threshold)

    try:
        roc_auc = float(roc_auc_score(y_test, test_scores))
    except ValueError:
        roc_auc = None

    try:
        avg_precision = float(average_precision_score(y_test, test_scores))
    except ValueError:
        avg_precision = None

    diagnostics = percentile_diagnostics(y_test, test_scores, cal_scores)

    report = {
        "experiment": "temporal_isolation_forest",
        "sequence_file": str(args.sequence_file),
        "label_file": str(args.label_file),
        "sequence_count": int(n),
        "sequence_length": int(t),
        "state_dimension": int(d),
        "temporal_feature_dimension": int(temporal_X.shape[1]),
        "split": {
            "detector_fit_end": int(fit_end),
            "calibration_start": int(fit_end),
            "test_start": int(train_end),
            "test_count": int(n - train_end),
        },
        "training": {
            "benign_only": True,
            "benign_fit_count": int(len(benign_fit)),
            "trees": int(args.trees),
            "max_samples": int(min(args.max_samples, len(X_fit_proc))),
            "contamination": "auto",
            "random_state": int(args.random_state),
        },
        "representation": {
            "components": ["latest", "delta", "mean10", "std10"],
            "features_per_component": int(d),
        },
        "preprocessing": {
            "signed_log1p": True,
            "robust_clip_scale": 8.0,
            "robust_scaler": True,
        },
        "configured_threshold_percentile": float(args.threshold_percentile),
        "configured_result": configured,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "threshold_tradeoff": diagnostics,
        "feature_magnitude_diagnostic": magnitude_diagnostics(temporal_X),
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "preprocessing": params,
            "threshold": threshold,
            "threshold_percentile": args.threshold_percentile,
            "state_dimension": d,
            "sequence_length": t,
            "feature_names": FEATURE_NAMES,
            "temporal_feature_names": temporal_feature_names(),
            "representation": ["latest", "delta", "mean10", "std10"],
        },
        MODEL_FILE,
    )

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    np.savez_compressed(
        SCORES_FILE,
        test_scores=test_scores,
        test_labels=y_test,
        calibration_scores=cal_scores,
        configured_threshold=np.asarray(threshold),
        test_start=np.asarray(train_end),
    )

    print()
    print("RESULTS")
    print("-" * 76)
    print(f"Configured threshold          : {threshold:.6f}")
    print(f"Precision                     : {configured['precision']:.4f}")
    print(f"Recall                        : {configured['recall']:.4f}")
    print(f"F1                            : {configured['f1']:.4f}")
    print(f"FPR                           : {configured['fpr']:.4f}")
    print(f"ROC-AUC                       : {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC                       : N/A")
    print(
        f"Average Precision             : {avg_precision:.4f}"
        if avg_precision is not None
        else "Average Precision             : N/A"
    )
    print(f"Predicted anomalies            : {configured['predicted_anomalies']:,}")
    print(f"Test anomaly rate              : {configured['test_anomaly_rate']:.2%}")

    print()
    print("THRESHOLD TRADE-OFF (diagnostic only)")
    print("-" * 76)
    print("Percentile | Precision | Recall | F1     | FPR")
    for row in diagnostics:
        print(
            f"{row['percentile']:9.1f}% | "
            f"{row['precision']:.4f}     | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['fpr']:.4f}"
        )

    print()
    print("TOP TEMPORAL FEATURE MAGNITUDE PROXIES")
    print("-" * 76)
    for row in report["feature_magnitude_diagnostic"]:
        print(
            f"{row['feature']:<40} "
            f"{row['median_absolute_value']:.4f}"
        )

    print()
    print("ARTIFACTS")
    print("-" * 76)
    print(f"Model                         : {MODEL_FILE}")
    print(f"Report                        : {REPORT_FILE}")
    print(f"Scores                        : {SCORES_FILE}")

    print()
    print("TEMPORAL ISOLATION FOREST: PASSED")
    print("=" * 76)


if __name__ == "__main__":
    main()
