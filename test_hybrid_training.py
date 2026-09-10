"""
ARJUN - Hybrid World Model Training Smoke Test

Purpose:
    Verify that the Hybrid World Model can actually train.

Checks:
    1. Dataset loading
    2. State/graph alignment
    3. Forward pass
    4. Backpropagation
    5. Loss reduction
    6. Optimizer updates
    7. Checkpoint saving
    8. Checkpoint loading
"""

import os
import pickle

import numpy as np
import torch

from world_model.hybrid_world_model import HybridWorldModel


# ============================================================
# CONFIGURATION
# ============================================================

STATE_FILE = "data/processed/graph_sequences_states.npz"
GRAPH_FILE = "data/processed/graph_sequences.pkl"

CHECKPOINT_DIR = "saved_models"

CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR,
    "hybrid_world_model_smoke_test.pt"
)

TEST_SAMPLES = 64
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    if not os.path.exists(STATE_FILE):
        raise FileNotFoundError(
            f"State dataset not found:\n{STATE_FILE}"
        )

    if not os.path.exists(GRAPH_FILE):
        raise FileNotFoundError(
            f"Graph dataset not found:\n{GRAPH_FILE}"
        )

    # --------------------------------------------------------
    # State sequences
    # --------------------------------------------------------

    state_data = np.load(STATE_FILE)

    X = state_data["X"]
    y = state_data["y"]

    # --------------------------------------------------------
    # Graph sequences
    # --------------------------------------------------------

    with open(GRAPH_FILE, "rb") as f:
        graph_data = pickle.load(f)

    # The saved file is a dictionary.
    # The actual graph sequences are stored under
    # the "graph_sequences" key.
    if isinstance(graph_data, dict):

        if "graph_sequences" not in graph_data:
            raise ValueError(
                "graph_sequences.pkl is a dictionary, "
                "but the 'graph_sequences' key is missing."
            )

        graph_sequences = graph_data["graph_sequences"]

    elif isinstance(graph_data, list):

        # Compatibility with a raw-list format.
        graph_sequences = graph_data

    else:

        raise TypeError(
            "Unsupported graph_sequences.pkl format: "
            f"{type(graph_data)}"
        )

    print(f"State sequences     : {X.shape}")
    print(f"Targets             : {y.shape}")
    print(f"Graph sequences     : {len(graph_sequences)}")

    # --------------------------------------------------------
    # Validate alignment
    # --------------------------------------------------------

    if len(X) != len(y):
        raise ValueError(
            "Number of state sequences does not match "
            "number of targets."
        )

    if len(X) != len(graph_sequences):
        raise ValueError(
            "Number of state sequences does not match "
            "number of graph sequences."
        )

    sequence_length = X.shape[1]
    state_dimension = X.shape[2]

    saved_sequence_length = graph_data.get(
        "sequence_length",
        sequence_length
    ) if isinstance(graph_data, dict) else sequence_length

    if saved_sequence_length != sequence_length:
        raise ValueError(
            "Sequence length mismatch: "
            f"states={sequence_length}, "
            f"graphs={saved_sequence_length}"
        )

    print("Dataset alignment   : OK")

    return X, y, graph_sequences


# ============================================================
# GRAPH CONVERSION
# ============================================================

def convert_graph_batch(graph_batch, device):

    converted = []

    for sequence in graph_batch:

        converted_sequence = []

        if len(sequence) == 0:
            raise ValueError(
                "Encountered an empty graph sequence."
            )

        for graph in sequence:

            if not isinstance(graph, dict):
                raise TypeError(
                    "Each graph must be a dictionary."
                )

            if "node_features" not in graph:
                raise ValueError(
                    "Graph is missing 'node_features'."
                )

            if "adjacency" not in graph:
                raise ValueError(
                    "Graph is missing 'adjacency'."
                )

            node_features = graph["node_features"]
            adjacency = graph["adjacency"]

            # ------------------------------------------------
            # Node features
            # ------------------------------------------------

            if not torch.is_tensor(node_features):

                node_features = torch.tensor(
                    node_features,
                    dtype=torch.float32
                )

            else:

                node_features = node_features.float()

            # ------------------------------------------------
            # Adjacency
            # ------------------------------------------------

            if not torch.is_tensor(adjacency):

                adjacency = torch.tensor(
                    adjacency,
                    dtype=torch.float32
                )

            else:

                adjacency = adjacency.float()

            node_features = node_features.to(device)
            adjacency = adjacency.to(device)

            # ------------------------------------------------
            # Validate dimensions
            # ------------------------------------------------

            if node_features.ndim != 2:

                raise ValueError(
                    "Node features must be 2-dimensional. "
                    f"Got shape {tuple(node_features.shape)}"
                )

            if adjacency.ndim != 2:

                raise ValueError(
                    "Adjacency must be 2-dimensional. "
                    f"Got shape {tuple(adjacency.shape)}"
                )

            if adjacency.shape[0] != adjacency.shape[1]:

                raise ValueError(
                    "Adjacency matrix must be square."
                )

            if adjacency.shape[0] != node_features.shape[0]:

                raise ValueError(
                    "Number of graph nodes does not match "
                    "adjacency dimensions."
                )

            if node_features.shape[1] != 6:

                raise ValueError(
                    "Expected 6 graph node features. "
                    f"Got {node_features.shape[1]}"
                )

            converted_sequence.append(
                {
                    "window_id": graph.get("window_id"),
                    "nodes": graph.get("nodes"),
                    "node_features": node_features,
                    "adjacency": adjacency,
                }
            )

        converted.append(converted_sequence)

    return converted


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ARJUN HYBRID WORLD MODEL TRAINING SMOKE TEST")
    print("=" * 70)
    print()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device              : {device}")
    print()

    # ========================================================
    # LOAD DATA
    # ========================================================

    X, y, graph_sequences = load_dataset()

    print()

    # ========================================================
    # LIMIT DATASET
    # ========================================================

    sample_count = min(
        TEST_SAMPLES,
        len(X)
    )

    X = X[:sample_count]
    y = y[:sample_count]
    graph_sequences = graph_sequences[:sample_count]

    print("=" * 70)
    print("SMOKE TEST DATA")
    print("=" * 70)

    print(f"Samples             : {sample_count}")
    print(f"Input shape         : {X.shape}")
    print(f"Target shape        : {y.shape}")
    print(f"Graph sequences     : {len(graph_sequences)}")
    print(
        f"Graphs per sequence : "
        f"{len(graph_sequences[0])}"
    )
    print()

    # ========================================================
    # STATE TENSORS
    # ========================================================

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
        device=device
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    state_dimension = X.shape[-1]

    model = HybridWorldModel(
        state_dimension=state_dimension,
        graph_input_features=6,
        graph_hidden_features=32,
        graph_output_features=32,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.1,
    ).to(device)

    print(f"State dimension     : {state_dimension}")
    print("Graph input         : 6")
    print("Graph embedding     : 32")
    print("LSTM hidden         : 128")
    print("LSTM layers         : 2")
    print()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = torch.nn.MSELoss()

    # ========================================================
    # INITIAL LOSS
    # ========================================================

    print("=" * 70)
    print("INITIAL LOSS")
    print("=" * 70)

    model.eval()

    with torch.no_grad():

        initial_graphs = convert_graph_batch(
            graph_sequences[:BATCH_SIZE],
            device
        )

        initial_prediction = model(
            X_tensor[:BATCH_SIZE],
            initial_graphs
        )

        initial_loss = criterion(
            initial_prediction,
            y_tensor[:BATCH_SIZE]
        )

    print(
        f"Initial MSE loss    : "
        f"{initial_loss.item():.6f}"
    )

    print()

    # ========================================================
    # TRAINING
    # ========================================================

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    model.train()

    losses = []

    for epoch in range(EPOCHS):

        epoch_loss = 0.0
        batches = 0

        # Shuffle samples
        indices = np.random.permutation(
            sample_count
        )

        for start in range(
            0,
            sample_count,
            BATCH_SIZE
        ):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            if len(batch_indices) == 0:
                continue

            batch_indices_tensor = torch.tensor(
                batch_indices,
                dtype=torch.long,
                device=device
            )

            batch_X = X_tensor[
                batch_indices_tensor
            ]

            batch_y = y_tensor[
                batch_indices_tensor
            ]

            batch_graphs = [
                graph_sequences[int(i)]
                for i in batch_indices
            ]

            batch_graphs = convert_graph_batch(
                batch_graphs,
                device
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            optimizer.zero_grad()

            prediction = model(
                batch_X,
                batch_graphs
            )

            loss = criterion(
                prediction,
                batch_y
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        average_loss = (
            epoch_loss /
            max(batches, 1)
        )

        losses.append(
            average_loss
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| MSE Loss: {average_loss:.6f}"
        )

    # ========================================================
    # FINAL LOSS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL LOSS")
    print("=" * 70)

    model.eval()

    with torch.no_grad():

        final_graphs = convert_graph_batch(
            graph_sequences[:BATCH_SIZE],
            device
        )

        final_prediction = model(
            X_tensor[:BATCH_SIZE],
            final_graphs
        )

        final_loss = criterion(
            final_prediction,
            y_tensor[:BATCH_SIZE]
        )

    print(
        f"Initial MSE loss    : "
        f"{initial_loss.item():.6f}"
    )

    print(
        f"Final MSE loss      : "
        f"{final_loss.item():.6f}"
    )

    # ========================================================
    # LOSS CHECK
    # ========================================================

    print()

    if final_loss.item() < initial_loss.item():

        print("LOSS REDUCTION      : PASSED")

    else:

        print("LOSS REDUCTION      : WARNING")

        print(
            "Training completed, but the final loss on "
            "the first batch did not decrease."
        )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    print()
    print("=" * 70)
    print("CHECKPOINT TEST")
    print("=" * 70)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "state_dimension": state_dimension,

        "graph_input_features": 6,

        "graph_hidden_features": 32,

        "graph_output_features": 32,

        "lstm_hidden": 128,

        "lstm_layers": 2,

        "dropout": 0.1,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_FILE
    )

    print(
        f"Checkpoint saved   : "
        f"{CHECKPOINT_FILE}"
    )

    if not os.path.exists(
        CHECKPOINT_FILE
    ):

        raise RuntimeError(
            "Checkpoint file was not created."
        )

    print("Checkpoint test    : PASSED")

    # ========================================================
    # RELOAD TEST
    # ========================================================

    print()
    print("=" * 70)
    print("CHECKPOINT RELOAD TEST")
    print("=" * 70)

    loaded = torch.load(
        CHECKPOINT_FILE,
        map_location=device
    )

    reloaded_model = HybridWorldModel(
        state_dimension=loaded[
            "state_dimension"
        ],

        graph_input_features=loaded[
            "graph_input_features"
        ],

        graph_hidden_features=loaded[
            "graph_hidden_features"
        ],

        graph_output_features=loaded[
            "graph_output_features"
        ],

        lstm_hidden=loaded[
            "lstm_hidden"
        ],

        lstm_layers=loaded[
            "lstm_layers"
        ],

        dropout=loaded[
            "dropout"
        ],
    ).to(device)

    reloaded_model.load_state_dict(
        loaded["model_state_dict"]
    )

    reloaded_model.eval()

    with torch.no_grad():

        reload_prediction = reloaded_model(
            X_tensor[:BATCH_SIZE],
            final_graphs
        )

    if reload_prediction.shape != final_prediction.shape:

        raise RuntimeError(
            "Reloaded model produced an "
            "incorrect output shape."
        )

    print(
        f"Reloaded output    : "
        f"{tuple(reload_prediction.shape)}"
    )

    print("Reload test         : PASSED")

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("HYBRID WORLD MODEL TRAINING SMOKE TEST")
    print("=" * 70)

    if final_loss.item() < initial_loss.item():

        print("RESULT              : PASSED")

    else:

        print("RESULT              : COMPLETED WITH WARNING")

    print()
    print("Verified:")
    print("  ✓ Dataset loading")
    print("  ✓ State/graph alignment")
    print("  ✓ Forward propagation")
    print("  ✓ MSE loss calculation")
    print("  ✓ Backpropagation")
    print("  ✓ Adam optimizer")
    print("  ✓ Gradient clipping")
    print("  ✓ Training loop")
    print("  ✓ Checkpoint saving")
    print("  ✓ Checkpoint loading")
    print()

    if final_loss.item() < initial_loss.item():

        print(
            "The Hybrid World Model is ready "
            "for full training."
        )

    else:

        print(
            "Training infrastructure works, but "
            "loss behavior should be investigated "
            "before full training."
        )

    print()


if __name__ == "__main__":
    main()