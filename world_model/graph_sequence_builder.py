"""
ARJUN - Graph-Aware Temporal Sequence Builder

Builds aligned temporal sequences for the Hybrid World Model.

Input
-----
training_states.npz
training_graphs.pkl

Output
------
graph_sequences_states.npz
graph_sequences.pkl

For every training sample:

    States:
        S[t], S[t+1], ..., S[t+L-1]

    Graphs:
        G[t], G[t+1], ..., G[t+L-1]

    Target:
        S[t+L]

Therefore:

    World Model:
        P(S[t+L] | S[t:t+L-1], G[t:t+L-1])

Important
---------
State and graph IDs are verified at every timestep.
No state/graph pair is silently dropped or reordered.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np


# ================================================================
# Paths
# ================================================================

ROOT = Path(
    __file__
).resolve().parents[1]

STATE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "training_states.npz"
)

GRAPH_FILE = (
    ROOT
    / "data"
    / "processed"
    / "training_graphs.pkl"
)

STATE_SEQUENCE_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "graph_sequences_states.npz"
)

GRAPH_SEQUENCE_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "graph_sequences.pkl"
)


# ================================================================
# Configuration
# ================================================================

DEFAULT_SEQUENCE_LENGTH = 10

EXPECTED_STATE_DIMENSION = 33

EXPECTED_GRAPH_FEATURES = 6


# ================================================================
# Validation
# ================================================================

def validate_state_dataset(
    states: np.ndarray,
    labels: np.ndarray,
    window_ids: np.ndarray,
) -> None:
    """
    Validate the complete state dataset.
    """

    if states.ndim != 2:

        raise ValueError(
            "States must have shape "
            "(N, state_dimension). "
            f"Received {states.shape}."
        )

    if states.shape[1] != EXPECTED_STATE_DIMENSION:

        raise ValueError(
            "Invalid state dimension. "
            f"Expected {EXPECTED_STATE_DIMENSION}, "
            f"received {states.shape[1]}."
        )

    if labels.ndim != 1:

        raise ValueError(
            "Labels must be one-dimensional."
        )

    if window_ids.ndim != 1:

        raise ValueError(
            "Window IDs must be one-dimensional."
        )

    if len(states) != len(labels):

        raise ValueError(
            "States and labels have different lengths."
        )

    if len(states) != len(window_ids):

        raise ValueError(
            "States and window IDs have different lengths."
        )

    if not np.isfinite(
        states
    ).all():

        raise ValueError(
            "States contain NaN or Inf."
        )

    if len(states) == 0:

        raise ValueError(
            "State dataset is empty."
        )


def validate_graph(
    graph,
    expected_id: int,
) -> None:
    """
    Validate one graph.
    """

    if not isinstance(
        graph,
        dict,
    ):

        raise ValueError(
            f"Graph {expected_id} must be a dictionary."
        )

    required_keys = {
        "window_id",
        "nodes",
        "node_features",
        "adjacency",
    }

    missing = (
        required_keys
        - set(graph.keys())
    )

    if missing:

        raise ValueError(
            f"Graph {expected_id} is missing keys: "
            f"{sorted(missing)}"
        )

    graph_id = int(
        graph["window_id"]
    )

    if graph_id != expected_id:

        raise ValueError(
            f"Graph ID mismatch. "
            f"Expected {expected_id}, "
            f"found {graph_id}."
        )

    node_features = np.asarray(
        graph["node_features"],
        dtype=np.float32,
    )

    adjacency = np.asarray(
        graph["adjacency"],
        dtype=np.float32,
    )

    if node_features.ndim != 2:

        raise ValueError(
            f"Graph {expected_id}: "
            "node_features must be 2-D."
        )

    if node_features.shape[1] != EXPECTED_GRAPH_FEATURES:

        raise ValueError(
            f"Graph {expected_id}: expected "
            f"{EXPECTED_GRAPH_FEATURES} node features, "
            f"received {node_features.shape[1]}."
        )

    node_count = (
        node_features.shape[0]
    )

    if node_count == 0:

        raise ValueError(
            f"Graph {expected_id} contains no nodes."
        )

    expected_adjacency_shape = (
        node_count,
        node_count,
    )

    if adjacency.shape != (
        expected_adjacency_shape
    ):

        raise ValueError(
            f"Graph {expected_id}: invalid adjacency "
            f"shape {adjacency.shape}; expected "
            f"{expected_adjacency_shape}."
        )

    if not np.isfinite(
        node_features
    ).all():

        raise ValueError(
            f"Graph {expected_id}: "
            "node features contain NaN/Inf."
        )

    if not np.isfinite(
        adjacency
    ).all():

        raise ValueError(
            f"Graph {expected_id}: "
            "adjacency contains NaN/Inf."
        )


def validate_all_graphs(
    graphs,
    expected_count: int,
) -> None:
    """
    Validate graph count and every graph ID.
    """

    if not isinstance(
        graphs,
        dict,
    ):

        raise ValueError(
            "Graph dataset must be a dictionary."
        )

    if len(graphs) != expected_count:

        raise ValueError(
            "State/graph count mismatch. "
            f"States={expected_count}, "
            f"Graphs={len(graphs)}."
        )

    for index in range(
        expected_count
    ):

        if index not in graphs:

            raise ValueError(
                f"Graph {index} is missing."
            )

        validate_graph(
            graphs[index],
            index,
        )


# ================================================================
# Build sequences
# ================================================================

def build_graph_sequences(
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
):
    """
    Build aligned state and graph sequences.

    Example with sequence_length=10:

        Input:
            S0 S1 S2 ... S9 S10

        Sequence:
            S0 ... S9

        Target:
            S10

        Graph sequence:
            G0 ... G9

    Returns
    -------
    state_sequences
        Shape:
            (N, sequence_length, 33)

    targets
        Shape:
            (N, 33)

    target_windows
        Shape:
            (N,)

    graph_sequences
        List containing N graph sequences.
    """

    if sequence_length < 2:

        raise ValueError(
            "sequence_length must be at least 2."
        )

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    print("=" * 70)

    print(
        "ARJUN GRAPH-AWARE SEQUENCE BUILDER"
    )

    print("=" * 70)

    print(
        f"Sequence length    : "
        f"{sequence_length}"
    )

    print()

    # ------------------------------------------------------------
    # Load states
    # ------------------------------------------------------------

    if not STATE_FILE.exists():

        raise FileNotFoundError(
            "State dataset not found:\n"
            f"{STATE_FILE}"
        )

    state_data = np.load(
        STATE_FILE,
        allow_pickle=True,
    )

    if "X" not in state_data:

        raise ValueError(
            "training_states.npz does not contain X."
        )

    if "window_ids" not in state_data:

        raise ValueError(
            "training_states.npz does not contain "
            "window_ids."
        )

    if "y" in state_data:

        label_key = "y"

    elif "labels" in state_data:

        label_key = "labels"

    else:

        raise ValueError(
            "training_states.npz contains neither "
            "'y' nor 'labels'."
        )

    states = np.asarray(
        state_data["X"],
        dtype=np.float32,
    )

    labels = np.asarray(
        state_data[label_key],
        dtype=np.int8,
    )

    window_ids = np.asarray(
        state_data["window_ids"],
        dtype=np.int64,
    )

    global_state_ids = None

    if "global_state_ids" in state_data:

        global_state_ids = np.asarray(
            state_data["global_state_ids"],
            dtype=np.int64,
        )

    validate_state_dataset(
        states,
        labels,
        window_ids,
    )

    # ------------------------------------------------------------
    # State ID validation
    # ------------------------------------------------------------

    expected_ids = np.arange(
        len(states),
        dtype=np.int64,
    )

    if not np.array_equal(
        window_ids,
        expected_ids,
    ):

        raise ValueError(
            "window_ids are not sequential. "
            "State and graph alignment cannot be "
            "guaranteed."
        )

    if global_state_ids is not None:

        if not np.array_equal(
            global_state_ids,
            expected_ids,
        ):

            raise ValueError(
                "global_state_ids are not sequential."
            )

    print(
        "STATE DATASET"
    )

    print(
        "-" * 70
    )

    print(
        f"States             : "
        f"{len(states):,}"
    )

    print(
        f"State dimension    : "
        f"{states.shape[1]}"
    )

    print(
        f"Labels             : "
        f"{len(labels):,}"
    )

    print(
        f"Window IDs         : "
        f"{len(window_ids):,}"
    )

    print()

    # ------------------------------------------------------------
    # Load graphs
    # ------------------------------------------------------------

    if not GRAPH_FILE.exists():

        raise FileNotFoundError(
            "Graph dataset not found:\n"
            f"{GRAPH_FILE}"
        )

    with open(
        GRAPH_FILE,
        "rb",
    ) as file:

        graphs = pickle.load(
            file
        )

    validate_all_graphs(
        graphs,
        len(states),
    )

    print(
        "GRAPH DATASET"
    )

    print(
        "-" * 70
    )

    print(
        f"Graphs             : "
        f"{len(graphs):,}"
    )

    print(
        "Graph features     : "
        f"{EXPECTED_GRAPH_FEATURES}"
    )

    print()

    # ------------------------------------------------------------
    # Common count
    # ------------------------------------------------------------

    common_count = min(
        len(states),
        len(graphs),
    )

    required_count = (
        sequence_length + 1
    )

    if common_count < required_count:

        raise ValueError(
            f"Only {common_count} aligned items "
            f"are available. At least "
            f"{required_count} are required for "
            f"sequence_length={sequence_length}."
        )

    # ------------------------------------------------------------
    # Verify state ↔ graph alignment
    # ------------------------------------------------------------

    print(
        "VERIFYING STATE ↔ GRAPH ALIGNMENT"
    )

    print(
        "-" * 70
    )

    for index in range(
        common_count
    ):

        state_id = int(
            window_ids[index]
        )

        if state_id != index:

            raise ValueError(
                f"State ID mismatch at index "
                f"{index}: found {state_id}."
            )

        graph = graphs[index]

        graph_id = int(
            graph["window_id"]
        )

        if graph_id != index:

            raise ValueError(
                f"Graph ID mismatch at index "
                f"{index}: found {graph_id}."
            )

    print(
        f"Verified {common_count:,} "
        "state/graph pairs."
    )

    print()

    # ------------------------------------------------------------
    # Sequence count
    # ------------------------------------------------------------

    sequence_count = (
        common_count
        - sequence_length
    )

    if sequence_count <= 0:

        raise ValueError(
            "No sequences can be created."
        )

    print(
        "BUILDING SEQUENCES"
    )

    print(
        "-" * 70
    )

    print(
        f"Sequence length    : "
        f"{sequence_length}"
    )

    print(
        f"Sequences           : "
        f"{sequence_count:,}"
    )

    # ------------------------------------------------------------
    # Allocate state arrays
    # ------------------------------------------------------------

    state_sequences = np.empty(
        (
            sequence_count,
            sequence_length,
            EXPECTED_STATE_DIMENSION,
        ),
        dtype=np.float32,
    )

    targets = np.empty(
        (
            sequence_count,
            EXPECTED_STATE_DIMENSION,
        ),
        dtype=np.float32,
    )

    target_windows = np.empty(
        sequence_count,
        dtype=np.int64,
    )

    target_labels = np.empty(
        sequence_count,
        dtype=np.int8,
    )

    graph_sequences = []

    # ------------------------------------------------------------
    # Construct sequences
    # ------------------------------------------------------------

    for sequence_index in range(
        sequence_count
    ):

        start = sequence_index

        end = (
            start
            + sequence_length
        )

        target_index = end

        # --------------------------------------------------------
        # States
        # --------------------------------------------------------

        state_sequence = states[
            start:end
        ]

        target_state = states[
            target_index
        ]

        state_sequences[
            sequence_index
        ] = state_sequence

        targets[
            sequence_index
        ] = target_state

        # --------------------------------------------------------
        # Target metadata
        # --------------------------------------------------------

        target_windows[
            sequence_index
        ] = int(
            window_ids[
                target_index
            ]
        )

        target_labels[
            sequence_index
        ] = int(
            labels[
                target_index
            ]
        )

        # --------------------------------------------------------
        # Graph sequence
        # --------------------------------------------------------

        sequence_graphs = []

        for graph_index in range(
            start,
            end,
        ):

            graph = graphs[
                graph_index
            ]

            # Validate again at sequence boundary.
            validate_graph(
                graph,
                graph_index,
            )

            sequence_graphs.append(
                graph
            )

        if len(
            sequence_graphs
        ) != sequence_length:

            raise RuntimeError(
                f"Sequence {sequence_index} "
                "has incorrect graph length."
            )

        graph_sequences.append(
            sequence_graphs
        )

    # ------------------------------------------------------------
    # Final numerical validation
    # ------------------------------------------------------------

    if not np.isfinite(
        state_sequences
    ).all():

        raise RuntimeError(
            "State sequences contain NaN or Inf."
        )

    if not np.isfinite(
        targets
    ).all():

        raise RuntimeError(
            "Targets contain NaN or Inf."
        )

    # ------------------------------------------------------------
    # Save state sequences
    # ------------------------------------------------------------

    STATE_SEQUENCE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        STATE_SEQUENCE_OUTPUT,

        X=state_sequences,

        y=targets,

        states=state_sequences,

        targets=targets,

        target_windows=target_windows,

        target_labels=target_labels,

        sequence_length=np.asarray(
            sequence_length,
            dtype=np.int64,
        ),

        state_dimension=np.asarray(
            EXPECTED_STATE_DIMENSION,
            dtype=np.int64,
        ),
    )

    # ------------------------------------------------------------
    # Save graph sequences
    # ------------------------------------------------------------

    with open(
        GRAPH_SEQUENCE_OUTPUT,
        "wb",
    ) as file:

        pickle.dump(
            graph_sequences,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------

    if len(
        graph_sequences
    ) != sequence_count:

        raise RuntimeError(
            "Final graph sequence count does not "
            "match state sequence count."
        )

    for index in range(
        sequence_count
    ):

        if len(
            graph_sequences[index]
        ) != sequence_length:

            raise RuntimeError(
                f"Graph sequence {index} "
                "has incorrect length."
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "GRAPH SEQUENCE BUILD COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Input states       : "
        f"{common_count:,}"
    )

    print(
        f"State dimension    : "
        f"{EXPECTED_STATE_DIMENSION}"
    )

    print(
        f"Sequence length    : "
        f"{sequence_length}"
    )

    print(
        f"Sequences created  : "
        f"{sequence_count:,}"
    )

    print(
        f"X shape            : "
        f"{state_sequences.shape}"
    )

    print(
        f"y shape            : "
        f"{targets.shape}"
    )

    print(
        f"Graph sequences    : "
        f"{len(graph_sequences):,}"
    )

    print()

    print(
        "State sequence file:"
    )

    print(
        f"  {STATE_SEQUENCE_OUTPUT}"
    )

    print()

    print(
        "Graph sequence file:"
    )

    print(
        f"  {GRAPH_SEQUENCE_OUTPUT}"
    )

    print()

    print(
        "GRAPH SEQUENCE BUILD: PASSED"
    )

    return (
        state_sequences,
        targets,
        target_windows,
        target_labels,
        graph_sequences,
    )


# ================================================================
# Standalone entry point
# ================================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build ARJUN graph-aware "
            "temporal sequences."
        )
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help=(
            "Number of historical states/graphs "
            "used to predict the next state."
        ),
    )

    args = parser.parse_args()

    build_graph_sequences(
        sequence_length=args.sequence_length
    )


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":

    main()