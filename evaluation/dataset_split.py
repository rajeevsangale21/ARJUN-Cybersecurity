"""
ARJUN - Chronological Dataset Splitting

Provides leakage-safe chronological splitting for temporal
World Model training and evaluation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# ============================================================
# Binary labels
# ============================================================

def create_binary_labels(
    labels,
) -> np.ndarray:
    """
    Convert labels into binary attack labels.

    Supported:
        0 / benign -> 0
        1 / attack -> 1
    """

    values = np.asarray(
        labels
    )

    if values.ndim != 1:
        values = values.reshape(-1)

    result = np.zeros(
        len(values),
        dtype=np.int64,
    )

    for i, value in enumerate(values):

        if isinstance(
            value,
            str,
        ):

            result[i] = int(
                value.strip().lower()
                != "benign"
            )

        else:

            try:
                result[i] = int(
                    float(value) > 0
                )

            except (
                ValueError,
                TypeError,
            ):

                result[i] = 1

    return result


# ============================================================
# Chronological split
# ============================================================

def chronological_split(
    X,
    y,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Split temporal data chronologically.

    No shuffling is performed.

    Example:

        70% earliest data  -> train
        15% next data      -> validation
        15% latest data    -> test
    """

    X = np.asarray(
        X
    )

    y = np.asarray(
        y
    )

    if len(X) != len(y):

        raise ValueError(
            "X and y must have the same "
            f"number of samples. "
            f"X={len(X)}, y={len(y)}"
        )

    if len(X) < 3:

        raise ValueError(
            "At least 3 samples are required."
        )

    if not (
        0.0 < train_ratio < 1.0
    ):

        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if not (
        0.0 <= validation_ratio < 1.0
    ):

        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if (
        train_ratio
        + validation_ratio
        >= 1.0
    ):

        raise ValueError(
            "train_ratio + validation_ratio "
            "must be less than 1."
        )

    n = len(X)

    train_end = int(
        n * train_ratio
    )

    validation_end = int(
        n * (
            train_ratio
            + validation_ratio
        )
    )

    # Safety against tiny datasets.
    train_end = max(
        1,
        min(
            train_end,
            n - 2,
        ),
    )

    validation_end = max(
        train_end + 1,
        min(
            validation_end,
            n - 1,
        ),
    )

    result = {
        "X_train": X[
            :train_end
        ],

        "y_train": y[
            :train_end
        ],

        "X_validation": X[
            train_end:validation_end
        ],

        "y_validation": y[
            train_end:validation_end
        ],

        "X_test": X[
            validation_end:
        ],

        "y_test": y[
            validation_end:
        ],
    }

    return result


# ============================================================
# Sequence split
# ============================================================

def chronological_sequence_split(
    X,
    y,
    target_windows=None,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Chronologically split temporal World Model sequences.

    If target_windows are provided, they are split using the
    exact same boundaries as X/y.
    """

    result = chronological_split(
        X=X,
        y=y,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    if target_windows is not None:

        target_windows = np.asarray(
            target_windows
        )

        if len(target_windows) != len(X):

            raise ValueError(
                "target_windows must have the same "
                "length as X."
            )

        n = len(X)

        train_end = int(
            n * train_ratio
        )

        validation_end = int(
            n * (
                train_ratio
                + validation_ratio
            )
        )

        train_end = max(
            1,
            min(
                train_end,
                n - 2,
            ),
        )

        validation_end = max(
            train_end + 1,
            min(
                validation_end,
                n - 1,
            ),
        )

        result[
            "target_windows_train"
        ] = target_windows[
            :train_end
        ]

        result[
            "target_windows_validation"
        ] = target_windows[
            train_end:validation_end
        ]

        result[
            "target_windows_test"
        ] = target_windows[
            validation_end:
        ]

    return result


# ============================================================
# State + label split
# ============================================================

def split_states_and_labels(
    states,
    labels,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper for state-level chronological splitting.
    """

    labels = create_binary_labels(
        labels
    )

    return chronological_split(
        X=np.asarray(
            states
        ),
        y=labels,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )


# ============================================================
# Timeline split
# ============================================================

def split_timeline(
    values,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Split a one-dimensional timeline chronologically.
    """

    values = np.asarray(
        values
    )

    n = len(values)

    if n < 3:

        raise ValueError(
            "At least 3 timeline values are required."
        )

    train_end = int(
        n * train_ratio
    )

    validation_end = int(
        n * (
            train_ratio
            + validation_ratio
        )
    )

    train_end = max(
        1,
        min(
            train_end,
            n - 2,
        ),
    )

    validation_end = max(
        train_end + 1,
        min(
            validation_end,
            n - 1,
        ),
    )

    return {
        "train": values[
            :train_end
        ],

        "validation": values[
            train_end:validation_end
        ],

        "test": values[
            validation_end:
        ],
    }


# ============================================================
# Diagnostics
# ============================================================

def describe_split(
    split: Dict[str, np.ndarray],
) -> Dict[str, int]:
    """
    Return sample counts for a chronological split.
    """

    return {
        "train": len(
            split["X_train"]
        ),

        "validation": len(
            split["X_validation"]
        ),

        "test": len(
            split["X_test"]
        ),
    }


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ARJUN CHRONOLOGICAL SPLIT TEST")
    print("=" * 70)

    # Simulated temporal sequence dataset.
    n = 49_990

    X = np.arange(
        n * 10 * 33,
        dtype=np.float32,
    ).reshape(
        n,
        10,
        33,
    )

    y = np.arange(
        n * 33,
        dtype=np.float32,
    ).reshape(
        n,
        33,
    )

    target_windows = np.arange(
        n,
        dtype=np.int64,
    )

    split = chronological_sequence_split(
        X=X,
        y=y,
        target_windows=target_windows,
        train_ratio=0.70,
        validation_ratio=0.15,
    )

    print(
        f"Total sequences : {n:,}"
    )

    print(
        f"Train           : "
        f"{len(split['X_train']):,}"
    )

    print(
        f"Validation      : "
        f"{len(split['X_validation']):,}"
    )

    print(
        f"Test            : "
        f"{len(split['X_test']):,}"
    )

    print()
    print(
        "Train shape:"
    )
    print(
        split["X_train"].shape
    )

    print()
    print(
        "Validation shape:"
    )
    print(
        split["X_validation"].shape
    )

    print()
    print(
        "Test shape:"
    )
    print(
        split["X_test"].shape
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    train_last = (
        split[
            "target_windows_train"
        ][-1]
    )

    validation_first = (
        split[
            "target_windows_validation"
        ][0]
    )

    validation_last = (
        split[
            "target_windows_validation"
        ][-1]
    )

    test_first = (
        split[
            "target_windows_test"
        ][0]
    )

    assert (
        train_last
        < validation_first
    )

    assert (
        validation_last
        < test_first
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert (
        split["X_train"].shape[1:]
        == (10, 33)
    )

    assert (
        split["y_train"].shape[1:]
        == (33,)
    )

    assert (
        len(
            split["target_windows_train"]
        )
        == len(
            split["X_train"]
        )
    )

    print()
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT TEST PASSED")
    print("=" * 70)