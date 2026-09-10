import argparse

import numpy as np

from pipeline import run_pipeline

from world_model.world_model import WorldModel

from world_model.sequence_builder import (
    build_sequences
)

from evaluation.baseline import (
    LogisticRegressionBaseline
)

from evaluation.metrics import (
    calculate_metrics,
    print_metrics
)


def calculate_forecast_errors(
    world_model,
    X,
    y
):
    """
    Calculate prediction error for each
    next-state prediction.
    """

    errors = []

    for sequence, target in zip(
        X,
        y
    ):

        prediction = (
            world_model.predict(
                sequence
            )
        )

        error = np.mean(
            np.abs(
                prediction -
                target
            )
        )

        errors.append(
            error
        )

    return np.asarray(
        errors,
        dtype=np.float32
    )


def find_threshold(
    errors,
    labels
):
    """
    Find a threshold using the training/evaluation
    data.

    The threshold is selected from observed
    forecast errors.
    """

    unique_values = np.unique(
        errors
    )

    if len(unique_values) == 1:
        return float(
            unique_values[0]
        )

    best_threshold = (
        unique_values[0]
    )

    best_f1 = -1.0

    for threshold in unique_values:

        predictions = (
            errors >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            labels,
            predictions
        )

        if metrics["f1"] > best_f1:

            best_f1 = metrics["f1"]

            best_threshold = (
                threshold
            )

    return float(
        best_threshold
    )


def main():

    parser = argparse.ArgumentParser(
        description="ARJUN Model Evaluation"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset"
    )

    parser.add_argument(
        "--model",
        default="saved_models/world_model.pt",
        help="ARJUN World Model"
    )

    parser.add_argument(
        "--label-column",
        default="label",
        help="Dataset label column"
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Temporal sequence length"
    )

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("ARJUN MODEL EVALUATION")
    print("=" * 65)

    # ---------------------------------------
    # Process dataset
    # ---------------------------------------

    result = run_pipeline(
        args.input
    )

    df = result["features"]

    states = result["states"]

    window_ids = result["window_ids"]

    # ---------------------------------------
    # Check labels
    # ---------------------------------------

    if args.label_column not in df.columns:

        raise ValueError(
            f"Dataset does not contain "
            f"'{args.label_column}'. "
            "Evaluation requires labelled data."
        )

    # ---------------------------------------
    # Window labels
    # ---------------------------------------

    from evaluation.evaluate import (
        create_binary_labels
    )

    row_labels = create_binary_labels(
        df,
        args.label_column
    )

    df["_binary_label"] = row_labels

    window_labels_series = (
        df.groupby("window_id")[
            "_binary_label"
        ].max()
    )

    # Align labels with encoder states.
    labels = np.array(
        [
            window_labels_series.get(
                window_id,
                0
            )
            for window_id in window_ids
        ],
        dtype=np.int64
    )

    # ---------------------------------------
    # Build temporal sequences
    # ---------------------------------------

    if len(states) <= args.sequence_length:

        raise ValueError(
            "Not enough states for evaluation."
        )

    X, y, target_windows = (
        build_sequences(
            states,
            window_ids,
            args.sequence_length
        )
    )

    # Labels correspond to target states.
    target_labels = np.array(
        [
            window_labels_series.get(
                window_id,
                0
            )
            for window_id in target_windows
        ],
        dtype=np.int64
    )

    # ---------------------------------------
    # Load World Model
    # ---------------------------------------

    world_model = WorldModel(
        state_dimension=states.shape[1]
    )

    world_model.load(
        args.model
    )

    print(
        f"\nEvaluating "
        f"{len(X):,} temporal sequences..."
    )

    # ---------------------------------------
    # ARJUN forecast errors
    # ---------------------------------------

    errors = calculate_forecast_errors(
        world_model,
        X,
        y
    )

    threshold = find_threshold(
        errors,
        target_labels
    )

    arjun_predictions = (
        errors >= threshold
    ).astype(int)

    arjun_metrics = calculate_metrics(
        target_labels,
        arjun_predictions
    )

    print_metrics(
        "ARJUN WORLD MODEL",
        arjun_metrics
    )

    print(
        f"\nForecast-error threshold: "
        f"{threshold:.6f}"
    )

    # ---------------------------------------
    # Logistic Regression baseline
    # ---------------------------------------

    # For a fair baseline, use the same
    # state representation but without
    # temporal sequence modelling.

    baseline_X = states[
        args.sequence_length:
    ]

    baseline_y = target_labels

    baseline = (
        LogisticRegressionBaseline()
    )

    baseline.fit(
        baseline_X,
        baseline_y
    )

    baseline_predictions = (
        baseline.predict(
            baseline_X
        )
    )

    baseline_metrics = calculate_metrics(
        baseline_y,
        baseline_predictions
    )

    print_metrics(
        "LOGISTIC REGRESSION BASELINE",
        baseline_metrics
    )

    # ---------------------------------------
    # Comparison
    # ---------------------------------------

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(
        f"{'Metric':<25}"
        f"{'ARJUN':>15}"
        f"{'Logistic Regression':>25}"
    )

    print("-" * 65)

    print(
        f"{'Precision':<25}"
        f"{arjun_metrics['precision']:>15.4f}"
        f"{baseline_metrics['precision']:>25.4f}"
    )

    print(
        f"{'Recall':<25}"
        f"{arjun_metrics['recall']:>15.4f}"
        f"{baseline_metrics['recall']:>25.4f}"
    )

    print(
        f"{'F1 Score':<25}"
        f"{arjun_metrics['f1']:>15.4f}"
        f"{baseline_metrics['f1']:>25.4f}"
    )

    print(
        f"{'False Positive Rate':<25}"
        f"{arjun_metrics['false_positive_rate']:>15.4f}"
        f"{baseline_metrics['false_positive_rate']:>25.4f}"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()