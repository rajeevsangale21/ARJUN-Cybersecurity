"""
ARJUN
Verify alignment between the existing state dataset
and the saved graph dataset.
"""

from pathlib import Path
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parent

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


def main():

    print("=" * 70)
    print("ARJUN STATE ↔ GRAPH ALIGNMENT TEST")
    print("=" * 70)

    # ========================================================
    # Load states
    # ========================================================

    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"State dataset not found:\n{STATE_FILE}"
        )

    states = np.load(
        STATE_FILE,
        allow_pickle=True
    )

    X_states = states["X"]
    y_states = states["y"]
    window_ids = states["window_ids"]

    print()
    print("STATE DATASET")
    print("-" * 70)

    print(
        "States shape       :",
        X_states.shape
    )

    print(
        "Labels shape       :",
        y_states.shape
    )

    print(
        "Window IDs shape   :",
        window_ids.shape
    )

    print(
        "State dimension    :",
        X_states.shape[1]
    )

    # ========================================================
    # Basic state checks
    # ========================================================

    if X_states.ndim != 2:
        raise ValueError(
            "Expected states to be a 2-D array."
        )

    if X_states.shape[1] != 33:
        raise ValueError(
            f"Expected 33 state features, "
            f"found {X_states.shape[1]}."
        )

    if len(X_states) != len(y_states):
        raise ValueError(
            "State count and label count differ."
        )

    if len(X_states) != len(window_ids):
        raise ValueError(
            "State count and window ID count differ."
        )

    if not np.isfinite(X_states).all():
        raise ValueError(
            "State dataset contains NaN or Inf."
        )

    # ========================================================
    # Load graphs
    # ========================================================

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph dataset not found:\n{GRAPH_FILE}"
        )

    with open(
        GRAPH_FILE,
        "rb"
    ) as file:

        graphs = pickle.load(file)

    print()
    print("GRAPH DATASET")
    print("-" * 70)

    print(
        "Graphs loaded      :",
        len(graphs)
    )

    # ========================================================
    # Determine common range
    # ========================================================

    common_count = min(
        len(X_states),
        len(graphs)
    )

    print(
        "Common items       :",
        common_count
    )

    if common_count == 0:
        raise ValueError(
            "No common state/graph entries."
        )

    # ========================================================
    # ID alignment
    # ========================================================

    print()
    print("ID ALIGNMENT")
    print("-" * 70)

    for i in range(common_count):

        if i not in graphs:
            raise ValueError(
                f"Graph {i} is missing."
            )

        graph_id = int(
            graphs[i]["window_id"]
        )

        state_id = int(
            window_ids[i]
        )

        if graph_id != state_id:

            raise ValueError(
                "STATE ↔ GRAPH ID MISMATCH\n"
                f"Index       : {i}\n"
                f"State ID    : {state_id}\n"
                f"Graph ID    : {graph_id}"
            )

    print(
        "All common IDs    : MATCH"
    )

    # ========================================================
    # Graph structure checks
    # ========================================================

    print()
    print("GRAPH STRUCTURE")
    print("-" * 70)

    graph_feature_dimensions = set()
    node_counts = []

    for i in range(common_count):

        graph = graphs[i]

        node_features = np.asarray(
            graph["node_features"],
            dtype=np.float32
        )

        adjacency = np.asarray(
            graph["adjacency"],
            dtype=np.float32
        )

        if node_features.ndim != 2:
            raise ValueError(
                f"Graph {i}: node features are not 2-D."
            )

        if node_features.shape[1] != 6:
            raise ValueError(
                f"Graph {i}: expected 6 node features, "
                f"found {node_features.shape[1]}."
            )

        node_count = node_features.shape[0]

        if adjacency.shape != (
            node_count,
            node_count
        ):
            raise ValueError(
                f"Graph {i}: invalid adjacency shape "
                f"{adjacency.shape} for "
                f"{node_count} nodes."
            )

        if not np.isfinite(
            node_features
        ).all():

            raise ValueError(
                f"Graph {i}: NaN/Inf in node features."
            )

        if not np.isfinite(
            adjacency
        ).all():

            raise ValueError(
                f"Graph {i}: NaN/Inf in adjacency."
            )

        graph_feature_dimensions.add(
            node_features.shape[1]
        )

        node_counts.append(
            node_count
        )

    print(
        "Node feature dims  :",
        sorted(graph_feature_dimensions)
    )

    print(
        "Minimum nodes      :",
        min(node_counts)
    )

    print(
        "Maximum nodes      :",
        max(node_counts)
    )

    print(
        "Average nodes      :",
        round(
            float(np.mean(node_counts)),
            2
        )
    )

    # ========================================================
    # State/graph count check
    # ========================================================

    print()
    print("COUNT CHECK")
    print("-" * 70)

    print(
        "States available   :",
        len(X_states)
    )

    print(
        "Graphs available   :",
        len(graphs)
    )

    if len(graphs) > len(X_states):
        raise ValueError(
            "More graphs than states. "
            "Alignment cannot be guaranteed."
        )

    print(
        "Graph count is compatible with states."
    )

    # ========================================================
    # Label distribution
    # ========================================================

    print()
    print("STATE LABELS")
    print("-" * 70)

    unique, counts = np.unique(
        y_states,
        return_counts=True
    )

    for label, count in zip(
        unique,
        counts
    ):

        print(
            f"Label {int(label)} : "
            f"{int(count):,}"
        )

    # ========================================================
    # Final result
    # ========================================================

    print()
    print("=" * 70)
    print("STATE ↔ GRAPH ALIGNMENT TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()