"""
ARJUN - Training State Dataset Inspector

Checks:
    1. Dataset shape
    2. Feature names
    3. NaN / Inf
    4. Constant features
    5. Zero-heavy features
    6. Feature ranges
    7. Attack/benign distribution
    8. Segment distribution
    9. Temporal ordering
"""

from pathlib import Path

import numpy as np


DATA_PATH = (
    Path("data")
    / "processed"
    / "training_states.npz"
)


def main():

    print("=" * 70)
    print("ARJUN TRAINING STATE DATASET INSPECTION")
    print("=" * 70)

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH.resolve()}"
        )

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    with np.load(
        DATA_PATH,
        allow_pickle=True
    ) as data:

        print()
        print("AVAILABLE FIELDS")
        print("-" * 70)

        print(
            data.files
        )

        X = data["X"]

        states = data["states"]

        labels = data["labels"]

        window_ids = data["window_ids"]

        segment_ids = data["segment_ids"]

        local_window_ids = data[
            "local_window_ids"
        ]

        source_files = data[
            "source_files"
        ]

        window_starts = data[
            "window_starts"
        ]

        window_ends = data[
            "window_ends"
        ]

        feature_names = list(
            data["feature_names"]
        )

    # ---------------------------------------------------------------
    # Basic shape
    # ---------------------------------------------------------------

    print()
    print("DATASET SHAPES")
    print("-" * 70)

    print(
        f"X                 : {X.shape}"
    )

    print(
        f"states            : {states.shape}"
    )

    print(
        f"labels            : {labels.shape}"
    )

    print(
        f"window_ids        : {window_ids.shape}"
    )

    print(
        f"segment_ids       : {segment_ids.shape}"
    )

    print(
        f"local_window_ids  : {local_window_ids.shape}"
    )

    print(
        f"feature count     : {len(feature_names)}"
    )

    # ---------------------------------------------------------------
    # Equality
    # ---------------------------------------------------------------

    print()
    print("DATASET CONSISTENCY")
    print("-" * 70)

    if np.array_equal(
        X,
        states
    ):

        print(
            "X == states       : MATCH"
        )

    else:

        print(
            "X == states       : FAILED"
        )

    # ---------------------------------------------------------------
    # Shape validation
    # ---------------------------------------------------------------

    if X.ndim != 2:

        raise RuntimeError(
            f"X must be 2D, got {X.ndim}D"
        )

    if X.shape[1] != len(
        feature_names
    ):

        raise RuntimeError(
            "Feature count does not match "
            "state dimension."
        )

    # ---------------------------------------------------------------
    # NaN / Inf
    # ---------------------------------------------------------------

    print()
    print("NUMERICAL QUALITY")
    print("-" * 70)

    nan_count = int(
        np.isnan(X).sum()
    )

    inf_count = int(
        np.isinf(X).sum()
    )

    print(
        f"NaN count         : {nan_count:,}"
    )

    print(
        f"Inf count         : {inf_count:,}"
    )

    if nan_count == 0 and inf_count == 0:

        print(
            "Numerical quality  : PASSED"
        )

    else:

        print(
            "Numerical quality  : FAILED"
        )

    # ---------------------------------------------------------------
    # Label distribution
    # ---------------------------------------------------------------

    print()
    print("LABEL DISTRIBUTION")
    print("-" * 70)

    benign_count = int(
        (labels == 0).sum()
    )

    attack_count = int(
        (labels == 1).sum()
    )

    print(
        f"Benign            : {benign_count:,}"
    )

    print(
        f"Attack            : {attack_count:,}"
    )

    print(
        f"Attack percentage : "
        f"{100.0 * attack_count / len(labels):.2f}%"
    )

    # ---------------------------------------------------------------
    # Segment distribution
    # ---------------------------------------------------------------

    print()
    print("SEGMENT DISTRIBUTION")
    print("-" * 70)

    unique_segments, segment_counts = (
        np.unique(
            segment_ids,
            return_counts=True
        )
    )

    for segment, count in zip(
        unique_segments,
        segment_counts
    ):

        matching = (
            source_files[
                segment_ids == segment
            ]
        )

        source_name = (
            str(matching[0])
            if len(matching) > 0
            else "unknown"
        )

        print(
            f"Segment {int(segment):2d} | "
            f"{count:6,} states | "
            f"{source_name}"
        )

    # ---------------------------------------------------------------
    # Window ID checks
    # ---------------------------------------------------------------

    print()
    print("WINDOW ID CHECK")
    print("-" * 70)

    expected_ids = np.arange(
        len(X),
        dtype=np.int64
    )

    if np.array_equal(
        window_ids,
        expected_ids
    ):

        print(
            "Global window IDs  : CONTIGUOUS"
        )

    else:

        print(
            "Global window IDs  : NOT CONTIGUOUS"
        )

    # ---------------------------------------------------------------
    # Local window IDs
    # ---------------------------------------------------------------

    print()
    print("LOCAL WINDOW CHECK")
    print("-" * 70)

    for segment in unique_segments:

        mask = (
            segment_ids == segment
        )

        local_ids = (
            local_window_ids[
                mask
            ]
        )

        if len(local_ids) <= 1:

            print(
                f"Segment {int(segment):2d} : "
                "not enough states"
            )

            continue

        differences = np.diff(
            local_ids
        )

        non_decreasing = np.all(
            differences >= 0
        )

        print(
            f"Segment {int(segment):2d} : "
            f"first={int(local_ids[0])}, "
            f"last={int(local_ids[-1])}, "
            f"ordered={non_decreasing}"
        )

    # ---------------------------------------------------------------
    # Feature statistics
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE STATISTICS")
    print("=" * 70)

    constant_features = []

    zero_heavy_features = []

    for index, name in enumerate(
        feature_names
    ):

        values = X[:, index]

        minimum = float(
            np.min(values)
        )

        maximum = float(
            np.max(values)
        )

        mean = float(
            np.mean(values)
        )

        median = float(
            np.median(values)
        )

        std = float(
            np.std(values)
        )

        zero_percentage = (
            100.0
            * float(
                np.sum(values == 0)
            )
            / len(values)
        )

        if minimum == maximum:

            constant_features.append(
                index
            )

        if zero_percentage >= 90.0:

            zero_heavy_features.append(
                index
            )

        print(
            f"{index:2d} "
            f"{name:28s} "
            f"min={minimum:12.4f} "
            f"median={median:12.4f} "
            f"mean={mean:12.4f} "
            f"std={std:12.4f} "
            f"max={maximum:12.4f} "
            f"zero={zero_percentage:6.2f}%"
        )

    # ---------------------------------------------------------------
    # Constant features
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("CONSTANT FEATURES")
    print("=" * 70)

    if constant_features:

        for index in constant_features:

            print(
                f"{index:2d} "
                f"{feature_names[index]}"
            )

    else:

        print(
            "None"
        )

    # ---------------------------------------------------------------
    # Zero-heavy
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("ZERO-HEAVY FEATURES (>=90% ZERO)")
    print("=" * 70)

    if zero_heavy_features:

        for index in zero_heavy_features:

            values = X[:, index]

            zero_percentage = (
                100.0
                * float(
                    np.sum(values == 0)
                )
                / len(values)
            )

            print(
                f"{index:2d} "
                f"{feature_names[index]:28s} "
                f"{zero_percentage:.2f}%"
            )

    else:

        print(
            "None"
        )

    # ---------------------------------------------------------------
    # Useful features
    # ---------------------------------------------------------------

    useful_features = [
        i
        for i in range(
            len(feature_names)
        )
        if i not in constant_features
    ]

    print()
    print("=" * 70)
    print("FEATURE USABILITY SUMMARY")
    print("=" * 70)

    print(
        f"Total features     : "
        f"{len(feature_names)}"
    )

    print(
        f"Constant features  : "
        f"{len(constant_features)}"
    )

    print(
        f"Non-constant       : "
        f"{len(useful_features)}"
    )

    print(
        f"Zero-heavy         : "
        f"{len(zero_heavy_features)}"
    )

    # ---------------------------------------------------------------
    # Temporal ordering
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("TEMPORAL ORDER CHECK")
    print("=" * 70)

    ordering_errors = 0

    for segment in unique_segments:

        mask = (
            segment_ids == segment
        )

        starts = (
            window_starts[
                mask
            ]
        )

        # Convert to timestamps.
        timestamps = pd_to_datetime_safe(
            starts
        )

        if len(timestamps) <= 1:

            continue

        diffs = np.diff(
            timestamps.astype(
                "datetime64[ns]"
            ).astype(
                np.int64
            )
        )

        if np.any(diffs < 0):

            ordering_errors += 1

            print(
                f"Segment {int(segment)} : "
                "NOT chronological"
            )

        else:

            print(
                f"Segment {int(segment)} : "
                "chronological"
            )

    if ordering_errors == 0:

        print(
            "Temporal ordering  : PASSED"
        )

    else:

        print(
            f"Temporal ordering  : "
            f"FAILED ({ordering_errors} segments)"
        )

    # ---------------------------------------------------------------
    # Final
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)

    print()
    print(
        "Do NOT train the World Model yet."
    )

    print(
        "We will use this inspection to decide "
        "which features should remain in the state."
    )


def pd_to_datetime_safe(values):
    """
    Convert object/string timestamps into numpy datetime64.
    """

    return np.asarray(
        [
            np.datetime64(
                str(value)
            )
            for value in values
        ]
    )


if __name__ == "__main__":

    main()