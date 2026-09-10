"""
ARJUN - Correct Network State Feature Inspection

This script inspects the actual World Model dataset.

Important:
    X = historical state sequences
    y = next-state targets

The target y is NOT a binary classification label.
"""

import os
import numpy as np


# ============================================================
# FILE
# ============================================================

STATE_FILE = "data/processed/graph_sequences_states.npz"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("ARJUN NETWORK STATE FEATURE INSPECTION")
    print("=" * 90)
    print()

    if not os.path.exists(STATE_FILE):

        raise FileNotFoundError(
            f"State dataset not found:\n{STATE_FILE}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    data = np.load(
        STATE_FILE
    )

    print("DATASET ARRAYS")
    print("-" * 90)

    for key in data.files:

        value = data[key]

        print(
            f"{key:<20} "
            f"shape={str(value.shape):<20} "
            f"dtype={value.dtype}"
        )

    print()

    # ========================================================
    # REQUIRED ARRAYS
    # ========================================================

    if "X" not in data:

        raise ValueError(
            "Dataset does not contain X."
        )

    if "y" not in data:

        raise ValueError(
            "Dataset does not contain y."
        )

    X = data["X"]

    y = data["y"]

    # ========================================================
    # SHAPES
    # ========================================================

    print("=" * 90)
    print("SHAPES")
    print("=" * 90)

    print(
        f"X shape             : {X.shape}"
    )

    print(
        f"y shape             : {y.shape}"
    )

    if X.ndim != 3:

        raise ValueError(
            "X must have shape "
            "(samples, sequence_length, features)."
        )

    if y.ndim != 2:

        raise ValueError(
            "y must have shape "
            "(samples, features)."
        )

    if X.shape[0] != y.shape[0]:

        raise ValueError(
            "X and y sample counts do not match."
        )

    if X.shape[2] != y.shape[1]:

        raise ValueError(
            "X feature count and y feature count "
            "do not match."
        )

    print()

    number_of_features = X.shape[2]

    sequence_length = X.shape[1]

    print(
        f"Samples             : {X.shape[0]}"
    )

    print(
        f"Sequence length     : {sequence_length}"
    )

    print(
        f"State features      : {number_of_features}"
    )

    print()

    # ========================================================
    # FLATTEN STATES
    # ========================================================

    states = X.reshape(
        -1,
        number_of_features
    )

    targets = y.reshape(
        -1,
        number_of_features
    )

    # ========================================================
    # NUMERICAL CHECK
    # ========================================================

    print("=" * 90)
    print("NUMERICAL VALIDATION")
    print("=" * 90)

    state_nan = int(
        np.isnan(states).sum()
    )

    state_inf = int(
        np.isinf(states).sum()
    )

    target_nan = int(
        np.isnan(targets).sum()
    )

    target_inf = int(
        np.isinf(targets).sum()
    )

    print(
        f"State NaN values    : {state_nan}"
    )

    print(
        f"State Inf values    : {state_inf}"
    )

    print(
        f"Target NaN values   : {target_nan}"
    )

    print(
        f"Target Inf values   : {target_inf}"
    )

    print()

    if (
        state_nan == 0
        and state_inf == 0
        and target_nan == 0
        and target_inf == 0
    ):

        print(
            "Numerical validation: PASSED"
        )

    else:

        print(
            "Numerical validation: FAILED"
        )

    # ========================================================
    # FEATURE STATISTICS
    # ========================================================

    print()
    print("=" * 90)
    print("STATE FEATURE STATISTICS")
    print("=" * 90)

    print()

    print(
        f"{'ID':>3} "
        f"{'Min':>14} "
        f"{'P01':>14} "
        f"{'Median':>14} "
        f"{'Mean':>14} "
        f"{'P99':>14} "
        f"{'Max':>14} "
        f"{'Std':>14} "
        f"{'Zeros':>8}"
    )

    print("-" * 90)

    constant_features = []

    large_scale_features = []

    zero_heavy_features = []

    feature_stats = []

    for i in range(
        number_of_features
    ):

        values = states[:, i]

        minimum = float(
            np.min(values)
        )

        p01 = float(
            np.percentile(values, 1)
        )

        median = float(
            np.median(values)
        )

        mean = float(
            np.mean(values)
        )

        p99 = float(
            np.percentile(values, 99)
        )

        maximum = float(
            np.max(values)
        )

        std = float(
            np.std(values)
        )

        zero_ratio = float(
            np.mean(values == 0)
        )

        feature_stats.append(
            {
                "id": i,
                "min": minimum,
                "p01": p01,
                "median": median,
                "mean": mean,
                "p99": p99,
                "max": maximum,
                "std": std,
                "zero_ratio": zero_ratio,
            }
        )

        print(
            f"{i:3d} "
            f"{minimum:14.4f} "
            f"{p01:14.4f} "
            f"{median:14.4f} "
            f"{mean:14.4f} "
            f"{p99:14.4f} "
            f"{maximum:14.4f} "
            f"{std:14.4f} "
            f"{zero_ratio * 100:7.2f}%"
        )

        if std < 1e-12:

            constant_features.append(i)

        if (
            abs(maximum) > 1000
            or abs(mean) > 1000
            or std > 1000
        ):

            large_scale_features.append(i)

        if zero_ratio >= 0.95:

            zero_heavy_features.append(i)

    # ========================================================
    # TARGET STATISTICS
    # ========================================================

    print()
    print("=" * 90)
    print("NEXT-STATE TARGET STATISTICS")
    print("=" * 90)

    target_std = np.std(
        targets,
        axis=0
    )

    target_mean = np.mean(
        targets,
        axis=0
    )

    for i in range(
        number_of_features
    ):

        print(
            f"Feature {i:02d} | "
            f"target mean={target_mean[i]:.4f} | "
            f"target std={target_std[i]:.4f}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"Features inspected    : {number_of_features}"
    )

    print(
        f"Constant features     : {len(constant_features)}"
    )

    print(
        f"Large-scale features  : {len(large_scale_features)}"
    )

    print(
        f"Zero-heavy features   : {len(zero_heavy_features)}"
    )

    print(
        f"State NaN values      : {state_nan}"
    )

    print(
        f"State Inf values      : {state_inf}"
    )

    print(
        f"Target NaN values     : {target_nan}"
    )

    print(
        f"Target Inf values     : {target_inf}"
    )

    print()

    if constant_features:

        print(
            "Constant feature IDs:"
        )

        print(
            constant_features
        )

    else:

        print(
            "Constant feature IDs: none"
        )

    print()

    if large_scale_features:

        print(
            "Large-scale feature IDs:"
        )

        print(
            large_scale_features
        )

    else:

        print(
            "Large-scale feature IDs: none"
        )

    print()

    if zero_heavy_features:

        print(
            "Zero-heavy feature IDs:"
        )

        print(
            zero_heavy_features
        )

    else:

        print(
            "Zero-heavy feature IDs: none"
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "y is the next network state, "
        "not a classification label."
    )

    print()

    print(
        "CORRECT STATE INSPECTION COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":

    main()