from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is required. Run: pip install xgboost"
    ) from exc


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.normalizer import NetworkStateNormalizer
from world_model.state_encoder import NetworkStateEncoder


STATE = ROOT / "data" / "processed" / "training_states.npz"
MODEL_PATH = ROOT / "saved_models" / "xgboost_attack_classifier.pkl"
FAMILY_PATH = ROOT / "data" / "processed" / "unseen_attack_state_labels.npz"

IMPROVED_MODEL_PATH = (
    ROOT / "saved_models" / "xgboost_attack_classifier_improved.pkl"
)
REPORT_PATH = ROOT / "evaluation" / "xgboost_phase_a1_report.json"


def load_data():
    with np.load(STATE, allow_pickle=True) as d:
        X = np.asarray(d["states"], dtype=np.float32)
        y = np.asarray(d["labels"]).reshape(-1)

    if FAMILY_PATH.exists():
        with np.load(FAMILY_PATH, allow_pickle=True) as d:
            families = np.asarray(
                d["state_families"], dtype=str
            ).reshape(-1)
    else:
        families = None

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if X.ndim != 2 or X.shape[1] != 33:
        raise ValueError(f"Expected (N,33), got {X.shape}")

    if len(X) != len(y):
        raise ValueError("State/label length mismatch.")

    return X, y, families


def chronological_split(X, y, families):
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    return (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:],
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
        None if families is None else families[:train_end],
        None if families is None else families[train_end:val_end],
        None if families is None else families[val_end:],
    )


def print_metrics(title, y_true, probability, threshold):
    pred = (probability >= threshold).astype(np.int8)

    precision = precision_score(
        y_true, pred, zero_division=0
    )
    recall = recall_score(
        y_true, pred, zero_division=0
    )
    f1 = f1_score(
        y_true, pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    fpr = fp / max(fp + tn, 1)

    print(f"\n{title}")
    print("-" * 78)
    print(f"Threshold   : {threshold:.3f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1          : {f1:.4f}")
    print(f"FPR         : {fpr:.4f}")
    print(f"TP / FP     : {tp:,} / {fp:,}")
    print(f"FN / TN     : {fn:,} / {tn:,}")

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def choose_threshold(y_val, val_probability):
    """
    Select threshold using VALIDATION ONLY.

    Primary objective: maximize F1.
    Tie-breaker: higher recall.
    Safety constraint: do not select a threshold that causes
    validation FPR > 0.50 unless no candidate satisfies the
    constraint.
    """
    candidates = np.arange(0.05, 0.951, 0.01)

    rows = []

    for threshold in candidates:
        pred = (val_probability >= threshold).astype(np.int8)

        precision = precision_score(
            y_val, pred, zero_division=0
        )
        recall = recall_score(
            y_val, pred, zero_division=0
        )
        f1 = f1_score(
            y_val, pred, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val, pred, labels=[0, 1]
        ).ravel()

        fpr = fp / max(fp + tn, 1)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fpr": float(fpr),
            }
        )

    safe = [r for r in rows if r["fpr"] <= 0.50]

    if safe:
        best = max(
            safe,
            key=lambda r: (
                r["f1"],
                r["recall"],
                r["precision"],
            ),
        )
    else:
        best = max(
            rows,
            key=lambda r: (
                r["f1"],
                r["recall"],
            ),
        )

    return best, rows


def build_improved_model():
    """
    More recall-oriented XGBoost configuration.

    scale_pos_weight is calculated from the training split,
    never from validation/test.
    """
    return XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_weight=1,
        gamma=0.0,
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )


def main():
    print("=" * 78)
    print("ARJUN PHASE A.1 - IMPROVED XGBOOST")
    print("=" * 78)

    if not STATE.exists():
        raise FileNotFoundError(STATE)

    X, y, families = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
        _,
        _,
    ) = chronological_split(X, y, families)

    print(f"\nStates       : {len(X):,}")
    print(f"Features     : {X.shape[1]}")
    print(f"Train       : {len(X_train):,}")
    print(f"Validation  : {len(X_val):,}")
    print(f"Test        : {len(X_test):,}")
    print(f"Train benign: {int(np.sum(y_train == 0)):,}")
    print(f"Train attack: {int(np.sum(y_train == 1)):,}")

    normalizer = NetworkStateNormalizer().fit(X_train)

    X_train_n = normalizer.transform(X_train)
    X_val_n = normalizer.transform(X_val)
    X_test_n = normalizer.transform(X_test)

    positive = max(int(np.sum(y_train == 1)), 1)
    negative = max(int(np.sum(y_train == 0)), 1)
    scale_pos_weight = negative / positive

    print(f"\nTraining scale_pos_weight: {scale_pos_weight:.4f}")

    model = build_improved_model()

    # XGBoost's scale_pos_weight is intentionally applied only to
    # the training objective. Validation/test remain untouched.
    model.set_params(scale_pos_weight=scale_pos_weight)

    print("\nTRAINING")
    print("-" * 78)

    model.fit(
        X_train_n,
        y_train,
        eval_set=[(X_val_n, y_val)],
        verbose=False,
    )

    print("Training: PASS")

    val_probability = model.predict_proba(X_val_n)[:, 1]
    test_probability = model.predict_proba(X_test_n)[:, 1]

    print("\nBASELINE REFERENCE")
    print("-" * 78)

    baseline = None
    if MODEL_PATH.exists():
        try:
            old = joblib.load(MODEL_PATH)

            if hasattr(old, "predict_proba"):
                old_val = old.predict_proba(X_val_n)
                old_test = old.predict_proba(X_test_n)

                if old_val.ndim == 2:
                    old_val = old_val[:, 1]
                if old_test.ndim == 2:
                    old_test = old_test[:, 1]

                baseline = {
                    "validation": print_metrics(
                        "Existing XGBoost @ 0.50",
                        y_val,
                        old_val,
                        0.50,
                    ),
                    "test": print_metrics(
                        "Existing XGBoost @ 0.50",
                        y_test,
                        old_test,
                        0.50,
                    ),
                }
        except Exception as exc:
            print(f"Could not load old model for comparison: {exc}")

    print("\nTHRESHOLD SEARCH")
    print("-" * 78)

    best, threshold_rows = choose_threshold(
        y_val,
        val_probability,
    )

    print(
        f"Chosen validation threshold : {best['threshold']:.2f}\n"
        f"Validation F1               : {best['f1']:.4f}\n"
        f"Validation Recall           : {best['recall']:.4f}\n"
        f"Validation Precision        : {best['precision']:.4f}\n"
        f"Validation FPR             : {best['fpr']:.4f}"
    )

    improved_val = print_metrics(
        "Improved XGBoost - Validation",
        y_val,
        val_probability,
        best["threshold"],
    )

    improved_test = print_metrics(
        "Improved XGBoost - UNTOUCHED TEST",
        y_test,
        test_probability,
        best["threshold"],
    )

    print("\nSAVING")
    print("-" * 78)

    artifact = {
        "model": model,
        "threshold": float(best["threshold"]),
        "feature_names": list(
            NetworkStateEncoder().feature_names
        ),
        "normalizer": normalizer,
        "training_split": {
            "train": len(X_train),
            "validation": len(X_val),
            "test": len(X_test),
        },
        "method": "recall_oriented_xgboost_with_validation_threshold",
    }

    IMPROVED_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(artifact, IMPROVED_MODEL_PATH)

    print(f"Saved: {IMPROVED_MODEL_PATH}")

    # Verify artifact immediately.
    loaded = joblib.load(IMPROVED_MODEL_PATH)

    if "model" not in loaded or "threshold" not in loaded:
        raise RuntimeError("Improved model artifact is invalid.")

    sample = loaded["normalizer"].transform(X_test[:5])
    sample_probability = loaded["model"].predict_proba(sample)[:, 1]

    if not np.isfinite(sample_probability).all():
        raise RuntimeError("Loaded model produced non-finite probabilities.")

    print("Artifact load smoke test: PASS")

    report = {
        "experiment": "arjun_phase_a1_improved_xgboost",
        "state_count": int(len(X)),
        "feature_count": int(X.shape[1]),
        "scale_pos_weight": float(scale_pos_weight),
        "chosen_threshold": float(best["threshold"]),
        "baseline": baseline,
        "improved_validation": improved_val,
        "improved_test": improved_test,
        "threshold_search": threshold_rows,
        "artifact": str(
            IMPROVED_MODEL_PATH.relative_to(ROOT)
        ),
        "test_was_not_used_for_threshold_selection": True,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Report: {REPORT_PATH}")

    print("\n" + "=" * 78)
    print("PHASE A.1: COMPLETE")
    print("=" * 78)
    print(
        "The improved classifier uses training-only class weighting "
        "and a validation-selected attack threshold."
    )
    print(
        "The final test metrics above are from the untouched "
        "chronological test split."
    )


if __name__ == "__main__":
    main()
