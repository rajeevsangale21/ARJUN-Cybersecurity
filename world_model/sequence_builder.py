"""
ARJUN - Temporal Sequence Builder

Converts chronological network states into supervised temporal
sequences for World Model training.

Input:
    states      -> (N, D)
    window_ids  -> (N,)

Output:
    X           -> (M, sequence_length, D)
    y           -> (M, D)
    target_ids  -> (M,)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _validate_inputs(
    states: np.ndarray,
    window_ids: np.ndarray,
    sequence_length: int,
) -> None:

    if not isinstance(
        states,
        np.ndarray,
    ):
        raise TypeError(
            "states must be a numpy array."
        )

    if not isinstance(
        window_ids,
        np.ndarray,
    ):
        raise TypeError(
            "window_ids must be a numpy array."
        )

    if states.ndim != 2:
        raise ValueError(
            "states must have shape "
            "(N, state_dimension). "
            f"Received {states.shape}."
        )

    if window_ids.ndim != 1:
        raise ValueError(
            "window_ids must be one-dimensional."
        )

    if len(states) != len(window_ids):
        raise ValueError(
            "states and window_ids must contain "
            "the same number of entries."
        )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be >= 1."
        )

    if len(states) <= sequence_length:
        raise ValueError(
            f"Need more than {sequence_length} states "
            f"but only {len(states)} were provided."
        )


def _sort_chronologically(
    states: np.ndarray,
    window_ids: np.ndarray,
):
    """
    Sort states by window ID.

    The dataset builder generates globally increasing window IDs,
    so this guarantees chronological ordering.
    """

    order = np.argsort(
        window_ids,
        kind="stable",
    )

    return (
        states[order],
        window_ids[order],
    )


def build_sequences(
    states: np.ndarray,
    window_ids: np.ndarray,
    sequence_length: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Build supervised temporal sequences.

    For every sequence:

        X[i] =
            S[t-sequence_length+1 : t+1]

        y[i] =
            S[t+1]

    Therefore the World Model learns:

        P(S[t+1] | S[t-sequence_length+1:t+1])

    Returns:

        {
            "X":             (M, sequence_length, D),
            "y":             (M, D),
            "target_windows": (M,)
        }
    """

    states = np.asarray(
        states,
        dtype=np.float32,
    )

    window_ids = np.asarray(
        window_ids,
        dtype=np.int64,
    )

    _validate_inputs(
        states,
        window_ids,
        sequence_length,
    )

    # --------------------------------------------------------
    # Remove NaN / Inf
    # --------------------------------------------------------

    states = np.nan_to_num(
        states,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    states, window_ids = (
        _sort_chronologically(
            states,
            window_ids,
        )
    )

    # --------------------------------------------------------
    # Build sequences
    # --------------------------------------------------------

    sequence_count = (
        len(states)
        - sequence_length
    )

    X = np.empty(
        (
            sequence_count,
            sequence_length,
            states.shape[1],
        ),
        dtype=np.float32,
    )

    y = np.empty(
        (
            sequence_count,
            states.shape[1],
        ),
        dtype=np.float32,
    )

    target_windows = np.empty(
        sequence_count,
        dtype=np.int64,
    )

    for i in range(sequence_count):

        start = i
        end = (
            i + sequence_length
        )

        X[i] = states[
            start:end
        ]

        y[i] = states[
            end
        ]

        target_windows[i] = (
            window_ids[end]
        )

    return {
        "X": X,
        "y": y,
        "target_windows": target_windows,
        "sequence_length": sequence_length,
        "state_dimension": states.shape[1],
    }


def build_sequences_from_npz(
    path: str,
    sequence_length: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Convenience function for loading training_states.npz.
    """

    data = np.load(
        path,
        allow_pickle=True,
    )

    if "X" not in data:
        raise ValueError(
            "NPZ file does not contain X."
        )

    if "window_ids" not in data:
        raise ValueError(
            "NPZ file does not contain window_ids."
        )

    return build_sequences(
        states=data["X"],
        window_ids=data["window_ids"],
        sequence_length=sequence_length,
    )


# ============================================================
# Compatibility wrapper
# ============================================================

def build_sequence_arrays(
    states: np.ndarray,
    window_ids: np.ndarray,
    sequence_length: int = 10,
):
    """
    Compatibility helper.

    Returns the traditional tuple:

        X, y, target_windows
    """

    result = build_sequences(
        states,
        window_ids,
        sequence_length,
    )

    return (
        result["X"],
        result["y"],
        result["target_windows"],
    )


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    import sys

    dataset_path = (
        "data/processed/training_states.npz"
    )

    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    print("=" * 70)
    print("ARJUN TEMPORAL SEQUENCE TEST")
    print("=" * 70)

    result = build_sequences_from_npz(
        dataset_path,
        sequence_length=10,
    )

    print(
        f"Input states       : "
        f"{result['X'].shape[0] + 10:,}"
    )

    print(
        f"State dimension    : "
        f"{result['state_dimension']}"
    )

    print(
        f"Sequence length    : "
        f"{result['sequence_length']}"
    )

    print(
        f"Sequences created  : "
        f"{len(result['X']):,}"
    )

    print(
        f"X shape            : "
        f"{result['X'].shape}"
    )

    print(
        f"y shape            : "
        f"{result['y'].shape}"
    )

    print(
        f"Target windows     : "
        f"{result['target_windows'].shape}"
    )

    print(
        f"NaN in X           : "
        f"{np.isnan(result['X']).sum()}"
    )

    print(
        f"Inf in X           : "
        f"{np.isinf(result['X']).sum()}"
    )

    print(
        f"NaN in y           : "
        f"{np.isnan(result['y']).sum()}"
    )

    print(
        f"Inf in y           : "
        f"{np.isinf(result['y']).sum()}"
    )

    print("=" * 70)
    print("SEQUENCE BUILDER TEST PASSED")
    print("=" * 70)