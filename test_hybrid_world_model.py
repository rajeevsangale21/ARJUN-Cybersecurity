"""
ARJUN
Phase 4.1 - Hybrid World Model Forward-Pass Test

Loads the real graph-aware sequences and performs a tiny
forward pass through:

    State sequence
        +
    Graph sequence
        ↓
    Temporal GNN
        ↓
    LSTM
        ↓
    Predicted next state
"""

from pathlib import Path
import pickle

import numpy as np
import torch

from world_model.hybrid_world_model import HybridWorldModel


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

STATE_SEQUENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph_sequences_states.npz"
)

GRAPH_SEQUENCE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "graph_sequences.pkl"
)


# ============================================================
# Configuration
# ============================================================

STATE_DIMENSION = 33
GRAPH_INPUT_FEATURES = 6
GRAPH_HIDDEN = 32
GRAPH_OUTPUT = 32
LSTM_HIDDEN = 128
LSTM_LAYERS = 2

TEST_BATCH_SIZE = 2


# ============================================================
# Main test
# ============================================================

def main():

    print("=" * 70)
    print("ARJUN HYBRID WORLD MODEL FORWARD-PASS TEST")
    print("=" * 70)

    # ========================================================
    # Check files
    # ========================================================

    if not STATE_SEQUENCE_FILE.exists():
        raise FileNotFoundError(
            f"State sequence file not found:\n"
            f"{STATE_SEQUENCE_FILE}"
        )

    if not GRAPH_SEQUENCE_FILE.exists():
        raise FileNotFoundError(
            f"Graph sequence file not found:\n"
            f"{GRAPH_SEQUENCE_FILE}"
        )

    # ========================================================
    # Load states
    # ========================================================

    state_data = np.load(
        STATE_SEQUENCE_FILE,
        allow_pickle=True
    )

    X = np.asarray(
        state_data["X"],
        dtype=np.float32
    )

    y = np.asarray(
        state_data["y"],
        dtype=np.float32
    )

    print()
    print("STATE INPUT")
    print("-" * 70)

    print(
        "X shape             :",
        X.shape
    )

    print(
        "Target shape        :",
        y.shape
    )

    if X.ndim != 3:
        raise ValueError(
            f"Expected X to be 3-D, found {X.ndim}-D."
        )

    if X.shape[2] != STATE_DIMENSION:
        raise ValueError(
            f"Expected state dimension "
            f"{STATE_DIMENSION}, "
            f"found {X.shape[2]}."
        )

    if X.shape[0] < TEST_BATCH_SIZE:
        raise ValueError(
            "Not enough samples for test batch."
        )

    sequence_length = X.shape[1]

    print(
        "Sequence length     :",
        sequence_length
    )

    print(
        "State dimension     :",
        X.shape[2]
    )

    # ========================================================
    # Check numerical values
    # ========================================================

    if not np.isfinite(X).all():
        raise ValueError(
            "State sequences contain NaN or Inf."
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "Targets contain NaN or Inf."
        )

    # ========================================================
    # Load graph sequences
    # ========================================================

    with open(
        GRAPH_SEQUENCE_FILE,
        "rb"
    ) as file:

        graph_payload = pickle.load(
            file
        )

    graph_sequences = graph_payload[
        "graph_sequences"
    ]

    print()
    print("GRAPH INPUT")
    print("-" * 70)

    print(
        "Graph sequences    :",
        len(graph_sequences)
    )

    print(
        "Graphs per sequence:",
        len(graph_sequences[0])
    )

    if len(graph_sequences) != len(X):
        raise ValueError(
            "Number of graph sequences does not "
            "match number of state sequences."
        )

    if len(graph_sequences[0]) != sequence_length:
        raise ValueError(
            "Number of graphs per sequence does not "
            "match state sequence length."
        )

    # ========================================================
    # Inspect first graphs
    # ========================================================

    print()
    print("FIRST GRAPH SEQUENCE")
    print("-" * 70)

    for time_index, graph in enumerate(
        graph_sequences[0]
    ):

        node_features = np.asarray(
            graph["node_features"],
            dtype=np.float32
        )

        adjacency = np.asarray(
            graph["adjacency"],
            dtype=np.float32
        )

        print(
            f"t={time_index:02d} | "
            f"graph_id={graph['window_id']:>4} | "
            f"nodes={node_features.shape[0]:>3} | "
            f"node_features={node_features.shape} | "
            f"adjacency={adjacency.shape}"
        )

        if node_features.shape[1] != (
            GRAPH_INPUT_FEATURES
        ):
            raise ValueError(
                f"Graph {graph['window_id']} has "
                f"{node_features.shape[1]} features; "
                f"expected {GRAPH_INPUT_FEATURES}."
            )

        if adjacency.shape != (
            node_features.shape[0],
            node_features.shape[0]
        ):
            raise ValueError(
                f"Graph {graph['window_id']} "
                "has invalid adjacency shape."
            )

        if not np.isfinite(
            node_features
        ).all():

            raise ValueError(
                f"Graph {graph['window_id']} "
                "contains NaN/Inf in node features."
            )

        if not np.isfinite(
            adjacency
        ).all():

            raise ValueError(
                f"Graph {graph['window_id']} "
                "contains NaN/Inf in adjacency."
            )

    # ========================================================
    # Select tiny batch
    # ========================================================

    X_test = X[
        :TEST_BATCH_SIZE
    ]

    graph_test = graph_sequences[
        :TEST_BATCH_SIZE
    ]

    print()
    print("TEST BATCH")
    print("-" * 70)

    print(
        "State batch shape   :",
        X_test.shape
    )

    print(
        "Graph batch size    :",
        len(graph_test)
    )

    # ========================================================
    # Select device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device              :",
        device
    )

    # ========================================================
    # Create model
    # ========================================================

    model = HybridWorldModel(
        state_dimension=STATE_DIMENSION,
        graph_input_features=GRAPH_INPUT_FEATURES,
        graph_hidden=GRAPH_HIDDEN,
        graph_output=GRAPH_OUTPUT,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS
    )

    model = model.to(
        device
    )

    model.eval()

    print()
    print("MODEL")
    print("-" * 70)

    print(
        "State dimension     :",
        STATE_DIMENSION
    )

    print(
        "Graph input         :",
        GRAPH_INPUT_FEATURES
    )

    print(
        "Graph embedding     :",
        GRAPH_OUTPUT
    )

    print(
        "LSTM hidden         :",
        LSTM_HIDDEN
    )

    print(
        "LSTM layers         :",
        LSTM_LAYERS
    )

    # ========================================================
    # Convert states to tensor
    # ========================================================

    state_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=device
    )

    # ========================================================
    # Forward pass
    # ========================================================

    print()
    print("RUNNING FORWARD PASS...")
    print("-" * 70)

    with torch.no_grad():

        prediction = model(
            state_tensor,
            graph_test
        )

    # ========================================================
    # Validate prediction
    # ========================================================

    print()
    print("MODEL OUTPUT")
    print("-" * 70)

    print(
        "Prediction type     :",
        type(prediction).__name__
    )

    print(
        "Prediction shape    :",
        tuple(prediction.shape)
    )

    print(
        "Expected shape      :",
        (TEST_BATCH_SIZE, STATE_DIMENSION)
    )

    if prediction.shape != (
        TEST_BATCH_SIZE,
        STATE_DIMENSION
    ):
        raise ValueError(
            "Hybrid World Model produced "
            "an incorrect output shape."
        )

    if not torch.isfinite(
        prediction
    ).all():

        raise ValueError(
            "Model prediction contains NaN or Inf."
        )

    # ========================================================
    # Test graph encoder separately
    # ========================================================

    print()
    print("GRAPH ENCODER TEST")
    print("-" * 70)

    first_graph = graph_sequences[0][0]

    node_features = torch.tensor(
        first_graph["node_features"],
        dtype=torch.float32,
        device=device
    )

    adjacency = torch.tensor(
        first_graph["adjacency"],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        graph_embedding = (
            model.encode_graph(
                node_features,
                adjacency
            )
        )

    print(
        "Graph ID            :",
        first_graph["window_id"]
    )

    print(
        "Node features       :",
        tuple(node_features.shape)
    )

    print(
        "Graph embedding     :",
        tuple(graph_embedding.shape)
    )

    print(
        "Expected embedding  :",
        (GRAPH_OUTPUT,)
    )

    if graph_embedding.shape != (
        GRAPH_OUTPUT,
    ):
        raise ValueError(
            "Incorrect graph embedding shape."
        )

    if not torch.isfinite(
        graph_embedding
    ).all():

        raise ValueError(
            "Graph embedding contains NaN or Inf."
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print("HYBRID WORLD MODEL FORWARD-PASS TEST: PASSED")
    print("=" * 70)

    print()
    print(
        "Verified:"
    )

    print(
        "  ✓ 33-dimensional network states"
    )

    print(
        "  ✓ 6-dimensional graph node features"
    )

    print(
        "  ✓ Variable-size graph processing"
    )

    print(
        "  ✓ GNN graph embedding"
    )

    print(
        "  ✓ State + graph fusion"
    )

    print(
        "  ✓ Temporal LSTM"
    )

    print(
        "  ✓ 33-dimensional next-state prediction"
    )

    print()
    print(
        "The Hybrid World Model is ready "
        "for the training test."
    )


if __name__ == "__main__":
    main()