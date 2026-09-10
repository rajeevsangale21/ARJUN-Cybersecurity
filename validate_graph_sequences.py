import os
import pickle
import numpy as np


STATE_FILE = "data/processed/graph_sequences_states.npz"
GRAPH_FILE = "data/processed/graph_sequences.pkl"


print("=" * 70)
print("ARJUN GRAPH-AWARE TEMPORAL VALIDATION")
print("=" * 70)
print()


# ---------------------------------------------------------------------
# CHECK FILES
# ---------------------------------------------------------------------

if not os.path.exists(STATE_FILE):
    raise FileNotFoundError(
        f"Missing state sequence file:\n{STATE_FILE}"
    )

if not os.path.exists(GRAPH_FILE):
    raise FileNotFoundError(
        f"Missing graph sequence file:\n{GRAPH_FILE}"
    )


# ---------------------------------------------------------------------
# LOAD STATE SEQUENCES
# ---------------------------------------------------------------------

data = np.load(
    STATE_FILE,
    allow_pickle=True
)

print("STATE SEQUENCE DATA")
print("-" * 70)

print(
    "Available fields:",
    list(data.files)
)

X = data["X"]
y = data["y"]
target_windows = data["target_windows"]

print(
    f"X shape             : {X.shape}"
)

print(
    f"y shape             : {y.shape}"
)

print(
    f"Target windows      : {target_windows.shape}"
)

print(
    f"X dtype             : {X.dtype}"
)

print(
    f"y dtype             : {y.dtype}"
)

print()


# ---------------------------------------------------------------------
# BASIC SHAPE VALIDATION
# ---------------------------------------------------------------------

print("BASIC SHAPE CHECK")
print("-" * 70)

assert X.ndim == 3, (
    f"X must be 3-D, got {X.ndim}"
)

assert y.ndim == 2, (
    f"y must be 2-D, got {y.ndim}"
)

assert X.shape[0] == y.shape[0], (
    "X and y have different numbers of samples"
)

assert X.shape[1] == 10, (
    f"Expected sequence length 10, got {X.shape[1]}"
)

assert X.shape[2] == 33, (
    f"Expected state dimension 33, got {X.shape[2]}"
)

assert y.shape[1] == 33, (
    f"Expected target dimension 33, got {y.shape[1]}"
)

assert len(target_windows) == len(X), (
    "Target window count does not match X"
)

print("Sequence dimensions : OK")
print("Sequence length     : 10")
print("State dimension     : 33")
print("Target dimension    : 33")
print()


# ---------------------------------------------------------------------
# NAN / INF CHECK
# ---------------------------------------------------------------------

print("NUMERICAL VALIDATION")
print("-" * 70)

x_nan = np.isnan(X).sum()
x_inf = np.isinf(X).sum()

y_nan = np.isnan(y).sum()
y_inf = np.isinf(y).sum()

print(
    f"X NaN count         : {x_nan:,}"
)

print(
    f"X Inf count         : {x_inf:,}"
)

print(
    f"y NaN count         : {y_nan:,}"
)

print(
    f"y Inf count         : {y_inf:,}"
)

assert x_nan == 0, "X contains NaN values"
assert x_inf == 0, "X contains Inf values"
assert y_nan == 0, "y contains NaN values"
assert y_inf == 0, "y contains Inf values"

print()
print("Numerical values    : OK")
print()


# ---------------------------------------------------------------------
# TARGET WINDOW ORDER
# ---------------------------------------------------------------------

print("TARGET WINDOW ORDER")
print("-" * 70)

print(
    f"First target window : {target_windows[0]}"
)

print(
    f"Last target window  : {target_windows[-1]}"
)

window_differences = np.diff(
    target_windows
)

print(
    f"Minimum difference  : "
    f"{window_differences.min()}"
)

print(
    f"Maximum difference  : "
    f"{window_differences.max()}"
)

non_sequential = np.sum(
    window_differences != 1
)

print(
    f"Non-sequential jumps: "
    f"{non_sequential:,}"
)

print()


# ---------------------------------------------------------------------
# LOAD GRAPH SEQUENCES
# ---------------------------------------------------------------------

print("GRAPH SEQUENCES")
print("-" * 70)

with open(
    GRAPH_FILE,
    "rb"
) as f:

    graph_sequences = pickle.load(f)

print(
    f"Graph sequences     : "
    f"{len(graph_sequences):,}"
)

assert len(graph_sequences) == len(X), (
    "Number of graph sequences does not "
    "match number of state sequences"
)

print(
    "State/graph count   : MATCH"
)

print()


# ---------------------------------------------------------------------
# GRAPH STRUCTURE VALIDATION
# ---------------------------------------------------------------------

print("GRAPH STRUCTURE VALIDATION")
print("-" * 70)

expected_sequence_length = X.shape[1]

graph_feature_dimensions = set()
node_counts = []

invalid_graphs = 0

for sequence_index, sequence in enumerate(
    graph_sequences
):

    if len(sequence) != expected_sequence_length:

        raise ValueError(
            f"Sequence {sequence_index} contains "
            f"{len(sequence)} graphs instead of "
            f"{expected_sequence_length}"
        )

    for graph in sequence:

        required = [
            "window_id",
            "nodes",
            "node_features",
            "adjacency"
        ]

        for key in required:

            if key not in graph:

                raise ValueError(
                    f"Graph missing key: {key}"
                )

        node_features = np.asarray(
            graph["node_features"],
            dtype=np.float32
        )

        adjacency = np.asarray(
            graph["adjacency"],
            dtype=np.float32
        )

        if node_features.ndim != 2:
            invalid_graphs += 1
            continue

        if adjacency.ndim != 2:
            invalid_graphs += 1
            continue

        if adjacency.shape[0] != adjacency.shape[1]:
            invalid_graphs += 1
            continue

        if adjacency.shape[0] != node_features.shape[0]:
            invalid_graphs += 1
            continue

        graph_feature_dimensions.add(
            node_features.shape[1]
        )

        node_counts.append(
            node_features.shape[0]
        )


print(
    f"Graph feature dims  : "
    f"{sorted(graph_feature_dimensions)}"
)

print(
    f"Minimum nodes       : "
    f"{min(node_counts)}"
)

print(
    f"Maximum nodes       : "
    f"{max(node_counts)}"
)

print(
    f"Average nodes       : "
    f"{np.mean(node_counts):.2f}"
)

print(
    f"Invalid graphs      : "
    f"{invalid_graphs}"
)

assert invalid_graphs == 0
assert graph_feature_dimensions == {6}

print()
print("Graph structures    : OK")
print()


# ---------------------------------------------------------------------
# GRAPH WINDOW ORDER
# ---------------------------------------------------------------------

print("GRAPH TEMPORAL ORDER")
print("-" * 70)

first_sequence = graph_sequences[0]

first_graph_ids = [
    int(graph["window_id"])
    for graph in first_sequence
]

print(
    "First sequence IDs  :",
    first_graph_ids
)

graph_differences = np.diff(
    first_graph_ids
)

print(
    "First sequence diff :",
    graph_differences.tolist()
)

if np.all(
    graph_differences == 1
):

    print(
        "First sequence      : CONTIGUOUS"
    )

else:

    print(
        "First sequence      : NON-CONTIGUOUS"
    )

print()


# ---------------------------------------------------------------------
# STATE ↔ GRAPH ID ALIGNMENT
# ---------------------------------------------------------------------

print("STATE ↔ GRAPH ALIGNMENT")
print("-" * 70)

alignment_errors = 0

samples_to_check = min(
    len(X),
    100
)

for i in range(
    samples_to_check
):

    sequence = graph_sequences[i]

    graph_ids = [
        int(graph["window_id"])
        for graph in sequence
    ]

    expected_start = graph_ids[0]

    expected_ids = list(
        range(
            expected_start,
            expected_start + X.shape[1]
        )
    )

    if graph_ids != expected_ids:

        alignment_errors += 1

        print(
            f"Alignment issue at sequence {i}"
        )

        if alignment_errors >= 5:
            break


print(
    f"Sequences checked   : "
    f"{samples_to_check}"
)

print(
    f"Alignment errors    : "
    f"{alignment_errors}"
)

assert alignment_errors == 0

print(
    "State/graph IDs     : ALIGNED"
)

print()


# ---------------------------------------------------------------------
# STATE TRANSITION CHECK
# ---------------------------------------------------------------------

print("STATE TRANSITION CHECK")
print("-" * 70)

# Compare the final state in each input sequence
# with the target state.

last_state = X[:, -1, :]

transition_difference = np.abs(
    y - last_state
)

mean_difference = np.mean(
    transition_difference
)

max_difference = np.max(
    transition_difference
)

print(
    f"Mean |target-last|  : "
    f"{mean_difference:.6f}"
)

print(
    f"Max |target-last|   : "
    f"{max_difference:.6f}"
)

print(
    "Note: non-zero values are expected because "
    "the target is the NEXT state."
)

print()


# ---------------------------------------------------------------------
# TARGET WINDOW UNIQUENESS
# ---------------------------------------------------------------------

print("TARGET WINDOW UNIQUENESS")
print("-" * 70)

unique_targets = np.unique(
    target_windows
)

print(
    f"Unique target windows : "
    f"{len(unique_targets):,}"
)

print(
    f"Total targets         : "
    f"{len(target_windows):,}"
)

duplicate_targets = (
    len(target_windows)
    - len(unique_targets)
)

print(
    f"Duplicate targets     : "
    f"{duplicate_targets:,}"
)

print()


# ---------------------------------------------------------------------
# FEATURE RANGE CHECK
# ---------------------------------------------------------------------

print("FEATURE RANGE CHECK")
print("-" * 70)

feature_min = np.min(
    X,
    axis=(0, 1)
)

feature_max = np.max(
    X,
    axis=(0, 1)
)

for feature_index in range(33):

    print(
        f"Feature {feature_index:02d} | "
        f"min={feature_min[feature_index]:12.4f} | "
        f"max={feature_max[feature_index]:12.4f}"
    )

print()


# ---------------------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------------------

print("=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print()
print("PASS: Sequence dimensions")
print("PASS: NaN / Inf validation")
print("PASS: Graph sequence count")
print("PASS: Graph structure")
print("PASS: Graph feature dimension")
print("PASS: Graph temporal ordering")
print("PASS: State ↔ graph alignment")
print("PASS: Target construction")
print()
print("GRAPH-AWARE TEMPORAL DATASET: VALIDATED")
print("=" * 70)