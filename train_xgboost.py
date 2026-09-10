import argparse
from pathlib import Path

import numpy as np

from pipeline import run_pipeline

from preprocessing.normalizer import (
    NetworkStateNormalizer,
)

from evaluation.dataset_split import (
    create_window_labels,
    split_states_and_labels,
)

from evaluation.xgboost_classifier import (
    XGBoostAttackClassifier,
)

from evaluation.metrics import (
    calculate_metrics,
    print_metrics,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train ARJUN XGBoost Attack Classifier"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV, PCAP, PCAPNG or Zeek log",
    )

    parser.add_argument(
        "--label-column",
        default="label",
        help="Attack label column",
    )

    parser.add_argument(
        "--output",
        default=(
            "saved_models/"
            "xgboost_attack_classifier.pkl"
        ),
        help="Output classifier path",
    )

    parser.add_argument(
        "--normalizer",
        default=(
            "saved_models/"
            "state_normalizer.pkl"
        ),
        help="Output normalizer path",
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("ARJUN XGBOOST ATTACK CLASSIFIER")
    print("=" * 70)

    # --------------------------------------------------
    # 1. PROCESS TELEMETRY
    # --------------------------------------------------

    print(
        "\n[1/7] Processing network telemetry..."
    )

    result = run_pipeline(
        args.input
    )

    states = np.asarray(
        result["states"],
        dtype=np.float32,
    )

    window_ids = np.asarray(
        result["window_ids"]
    )

    df = result["features"]

    feature_names = list(
        result.get(
            "feature_names",
            [],
        )
    )

    if len(states) == 0:
        raise ValueError(
            "Pipeline produced no network states."
        )

    print(
        f"Network states: {len(states)}"
    )

    print(
        f"State dimension: {states.shape[1]}"
    )

    # --------------------------------------------------
    # 2. CHECK LABEL COLUMN
    # --------------------------------------------------

    print(
        "\n[2/7] Creating window-level labels..."
    )

    if args.label_column not in df.columns:
        raise ValueError(
            f"Label column '{args.label_column}' "
            "was not found in the dataset."
        )

    labels = create_window_labels(
        df,
        window_ids,
        label_column=args.label_column,
    )

    if len(labels) != len(states):
        raise ValueError(
            "Number of labels does not match "
            "number of network states."
        )

    print(
        f"Benign windows: "
        f"{int(np.sum(labels == 0))}"
    )

    print(
        f"Attack windows: "
        f"{int(np.sum(labels == 1))}"
    )

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "The dataset contains only one class. "
            "XGBoost requires both benign and attack "
            "windows for training."
        )

    # --------------------------------------------------
    # 3. CHRONOLOGICAL SPLIT
    # --------------------------------------------------

    print(
        "\n[3/7] Creating chronological split..."
    )

    splits = split_states_and_labels(
        states,
        labels,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    X_train = splits["X_train"]
    y_train = splits["y_train"]

    X_validation = splits[
        "X_validation"
    ]
    y_validation = splits[
        "y_validation"
    ]

    X_test = splits["X_test"]
    y_test = splits["y_test"]

    print(
        f"Train samples      : {len(X_train)}"
    )

    print(
        f"Validation samples : {len(X_validation)}"
    )

    print(
        f"Test samples       : {len(X_test)}"
    )

    # --------------------------------------------------
    # 4. NORMALIZATION
    # --------------------------------------------------

    print(
        "\n[4/7] Fitting state normalizer "
        "on training data..."
    )

    normalizer = (
        NetworkStateNormalizer()
    )

    X_train_normalized = (
        normalizer.fit_transform(
            X_train
        )
    )

    X_validation_normalized = (
        normalizer.transform(
            X_validation
        )
    )

    X_test_normalized = (
        normalizer.transform(
            X_test
        )
    )

    normalizer.save(
        args.normalizer
    )

    print(
        f"Normalizer saved: "
        f"{args.normalizer}"
    )

    # --------------------------------------------------
    # 5. TRAIN XGBOOST
    # --------------------------------------------------

    print(
        "\n[5/7] Training XGBoost classifier..."
    )

    classifier = (
        XGBoostAttackClassifier()
    )

    classifier.fit(
        X_train_normalized,
        y_train,
        feature_names=feature_names,
        validation_data=(
            X_validation_normalized,
            y_validation,
        ),
    )

    classifier_path = Path(
        args.output
    )

    classifier_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    classifier.save(
        classifier_path
    )

    print(
        f"XGBoost model saved: "
        f"{classifier_path}"
    )

    # --------------------------------------------------
    # 6. TEST MODEL
    # --------------------------------------------------

    print(
        "\n[6/7] Evaluating XGBoost..."
    )

    predictions = classifier.predict(
        X_test_normalized
    )

    probabilities = (
        classifier.predict_proba(
            X_test_normalized
        )
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print_metrics(
        "XGBOOST TEST RESULTS",
        metrics,
    )

    print(
        f"\nAverage attack probability: "
        f"{float(np.mean(probabilities)):.4f}"
    )

    # --------------------------------------------------
    # 7. FEATURE IMPORTANCE
    # --------------------------------------------------

    print(
        "\n[7/7] Top feature importance..."
    )

    importance = (
        classifier.feature_importance()
    )

    for item in importance[:10]:

        print(
            f"  - "
            f"{item['feature']}: "
            f"{item['importance']:.6f}"
        )

    print()
    print("=" * 70)
    print(
        "XGBOOST TRAINING COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()