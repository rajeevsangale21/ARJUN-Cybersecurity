"""
ARJUN - Model Evaluation

Evaluates:

    1. Logistic Regression baseline
    2. XGBoost classifier

using the authoritative ARJUN sequence dataset.

Evaluation split:

    70% chronological training
    15% chronological validation
    15% chronological test

No shuffling is performed.

Required metrics:

    Precision
    Recall
    F1
    False Positive Rate
"""

from pathlib import Path
import json

import numpy as np

from evaluation.dataset_split import (
    chronological_split,
)

from evaluation.metrics import (
    calculate_metrics,
)

from evaluation.baseline import (
    LogisticRegressionBaseline,
)

from evaluation.xgboost_classifier import (
    XGBoostAttackClassifier,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

SEQUENCE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "graph_sequences_states.npz"
)

ORIGINAL_STATE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "training_states.npz"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "model_evaluation_results.json"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    if not SEQUENCE_PATH.exists():

        raise FileNotFoundError(
            f"Sequence dataset not found:\n"
            f"{SEQUENCE_PATH}"
        )

    if not ORIGINAL_STATE_PATH.exists():

        raise FileNotFoundError(
            f"Original state dataset not found:\n"
            f"{ORIGINAL_STATE_PATH}"
        )

    sequence_data = np.load(
        SEQUENCE_PATH,
        allow_pickle=True,
    )

    original_data = np.load(
        ORIGINAL_STATE_PATH,
        allow_pickle=True,
    )

    # --------------------------------------------------------
    # Sequence states
    # --------------------------------------------------------

    if "states" in sequence_data:

        sequences = np.asarray(
            sequence_data["states"],
            dtype=np.float32,
        )

    elif "X" in sequence_data:

        sequences = np.asarray(
            sequence_data["X"],
            dtype=np.float32,
        )

    else:

        raise KeyError(
            "graph_sequences_states.npz does not contain "
            "'states' or 'X'."
        )

    # --------------------------------------------------------
    # Original labels
    # --------------------------------------------------------

    if "labels" not in original_data:

        raise KeyError(
            "training_states.npz does not contain "
            "'labels'."
        )

    original_labels = np.asarray(
        original_data["labels"]
    ).reshape(-1)

    original_labels = original_labels.astype(int)

    return sequences, original_labels


# ============================================================
# LABEL ALIGNMENT
# ============================================================

def align_labels(
    sequences,
    original_labels,
):

    num_sequences = len(sequences)

    sequence_length = sequences.shape[1]

    start_index = sequence_length

    end_index = (
        start_index
        + num_sequences
    )

    if end_index > len(original_labels):

        raise ValueError(
            "Sequence targets exceed the "
            "available original labels."
        )

    labels = original_labels[
        start_index:end_index
    ]

    labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    unique = np.unique(labels)

    if not np.all(
        np.isin(
            unique,
            [0, 1],
        )
    ):

        raise ValueError(
            "Labels must be binary 0/1. "
            f"Found: {unique}"
        )

    return labels


# ============================================================
# BASELINE FEATURE CONSTRUCTION
# ============================================================

def make_baseline_features(
    sequences,
):

    if sequences.ndim != 3:

        raise ValueError(
            "Expected sequence array with shape "
            "(samples, sequence_length, features). "
            f"Received: {sequences.shape}"
        )

    if sequences.shape[1] < 2:

        raise ValueError(
            "At least two states are required."
        )

    latest = sequences[:, -1, :]

    previous = sequences[:, -2, :]

    delta = latest - previous

    features = np.concatenate(
        [
            latest,
            delta,
        ],
        axis=1,
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return features.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ARJUN MODEL EVALUATION")
    print("=" * 70)

    print()
    print(
        f"Sequence dataset : {SEQUENCE_PATH}"
    )

    print(
        f"Original states  : {ORIGINAL_STATE_PATH}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    sequences, original_labels = load_data()

    print()
    print("DATA LOADED")
    print("-" * 70)

    print(
        f"Sequences        : {sequences.shape}"
    )

    print(
        f"Original states  : {original_labels.shape}"
    )

    print(
        f"Original benign  : "
        f"{int((original_labels == 0).sum())}"
    )

    print(
        f"Original attack  : "
        f"{int((original_labels == 1).sum())}"
    )

    # --------------------------------------------------------
    # Align labels
    # --------------------------------------------------------

    y = align_labels(
        sequences,
        original_labels,
    )

    print()
    print("TARGET ALIGNMENT")
    print("-" * 70)

    print(
        f"Target samples   : {len(y)}"
    )

    print(
        f"Benign targets   : "
        f"{int((y == 0).sum())}"
    )

    print(
        f"Attack targets   : "
        f"{int((y == 1).sum())}"
    )

    print(
        f"Attack percentage: "
        f"{100.0 * y.mean():.2f}%"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = make_baseline_features(
        sequences
    )

    print()
    print("FEATURE MATRIX")
    print("-" * 70)

    print(
        f"Features         : {X.shape}"
    )

    print(
        "33 current-state features "
        "+ 33 movement features = 66"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split = chronological_split(
        X=X,
        y=y,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    X_train = split["X_train"]

    y_train = split["y_train"]

    X_validation = split[
        "X_validation"
    ]

    y_validation = split[
        "y_validation"
    ]

    X_test = split["X_test"]

    y_test = split["y_test"]

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 70)

    print(
        f"Training   : {len(X_train):,}"
    )

    print(
        f"Validation : {len(X_validation):,}"
    )

    print(
        f"Test       : {len(X_test):,}"
    )

    print()
    print(
        f"Train benign : "
        f"{int((y_train == 0).sum()):,}"
    )

    print(
        f"Train attack : "
        f"{int((y_train == 1).sum()):,}"
    )

    print(
        f"Test benign  : "
        f"{int((y_test == 0).sum()):,}"
    )

    print(
        f"Test attack  : "
        f"{int((y_test == 1).sum()):,}"
    )

    if len(np.unique(y_train)) < 2:

        raise ValueError(
            "Training set contains only one class."
        )

    if len(np.unique(y_test)) < 2:

        raise ValueError(
            "Test set contains only one class."
        )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION BASELINE")
    print("=" * 70)

    baseline = LogisticRegressionBaseline()

    print(
        "Training Logistic Regression..."
    )

    baseline.fit(
        X_train,
        y_train,
    )

    baseline_predictions = (
        baseline.predict(
            X_test
        )
    )

    baseline_metrics = (
        calculate_metrics(
            y_test,
            baseline_predictions,
        )
    )

    print()
    print(
        f"Precision : "
        f"{baseline_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{baseline_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{baseline_metrics['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{baseline_metrics['false_positive_rate']:.4f}"
    )

    # ========================================================
    # XGBOOST
    # ========================================================

    print()
    print("=" * 70)
    print("XGBOOST CLASSIFIER")
    print("=" * 70)

    xgb = XGBoostAttackClassifier()

    print(
        "Training XGBoost..."
    )

    xgb.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
    )

    xgb_predictions = (
        xgb.predict(
            X_test
        )
    )

    xgb_metrics = (
        calculate_metrics(
            y_test,
            xgb_predictions,
        )
    )

    print()
    print(
        f"Precision : "
        f"{xgb_metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{xgb_metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{xgb_metrics['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{xgb_metrics['false_positive_rate']:.4f}"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print()

    print(
        f"{'Metric':<22}"
        f"{'Logistic Regression':>22}"
        f"{'XGBoost':>15}"
    )

    print("-" * 60)

    for metric in [
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
    ]:

        display_name = {
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "false_positive_rate": "FPR",
        }[metric]

        print(
            f"{display_name:<22}"
            f"{baseline_metrics[metric]:>22.4f}"
            f"{xgb_metrics[metric]:>15.4f}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "dataset": {
            "sequence_count": int(
                len(sequences)
            ),

            "sequence_shape": list(
                sequences.shape
            ),

            "feature_count": int(
                X.shape[1]
            ),
        },

        "split": {
            "train_samples": int(
                len(X_train)
            ),

            "validation_samples": int(
                len(X_validation)
            ),

            "test_samples": int(
                len(X_test)
            ),

            "method": "chronological",

            "train_ratio": 0.70,

            "validation_ratio": 0.15,

            "test_ratio": 0.15,
        },

        "logistic_regression": {
            **baseline_metrics
        },

        "xgboost": {
            **xgb_metrics
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved results: {OUTPUT_PATH}"
    )

    print()
    print("=" * 70)
    print("MODEL EVALUATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()