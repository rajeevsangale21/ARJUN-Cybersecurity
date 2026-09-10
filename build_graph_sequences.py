from pathlib import Path
import pickle
import numpy as np


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

STATE_FILE = PROCESSED / "training_states.npz"
GRAPH_FILE = PROCESSED / "training_graphs.pkl"

OUTPUT_STATES = PROCESSED / "graph_sequences_states.npz"
OUTPUT_GRAPHS = PROCESSED / "graph_sequences.pkl"

SEQUENCE_LENGTH = 10


def load_state_data():

    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"State dataset not found:\n{STATE_FILE}"
        )

    data = np.load(
        STATE_FILE,
        allow_pickle=True
    )

    print("Available state fields:")
    print(list(data.files))
    print()

    required = [
        "X",
        "y",
        "window_ids"
    ]

    for key in required:

        if key not in data:

            raise KeyError(
                f"'{key}' is missing from {STATE_FILE}\n"
                f"Available fields: {list(data.files)}"
            )

    states = np.asarray(
        data["X"],
        dtype=np.float32
    )

    labels = np.asarray(
        data["y"]
    )

    window_ids = np.asarray(
        data["window_ids"]
    )

    print(
        f"States loaded     : {states.shape}"
    )

    print(
        f"Labels loaded     : {labels.shape}"
    )

    print(
        f"Window IDs loaded : {window_ids.shape}"
    )

    if states.ndim != 2:

        raise ValueError(
            f"Expected X to have shape [N, D], "
            f"got {states.shape}"
        )

    if states.shape[1] != 33:

        raise ValueError(
            f"Expected state dimension 33, "
            f"got {states.shape[1]}"
        )

    if len(states) != len(window_ids):

        raise ValueError(
            "X and window_ids have different lengths"
        )

    return states, labels, window_ids


def load_graph_data():

    if not GRAPH_FILE.exists():

        raise FileNotFoundError(
            f"Graph dataset not found:\n{GRAPH_FILE}"
        )

    with open(
        GRAPH_FILE,
        "rb"
    ) as f:

        graphs = pickle.load(f)

    print(
        f"Graphs loaded     : {len(graphs):,}"
    )

    return graphs


def normalize_graph_container(graphs):

    if isinstance(graphs, dict):

        result = {}

        for key, graph in graphs.items():

            try:
                window_id = int(key)

            except Exception:

                window_id = int(
                    graph["window_id"]
                )

            result[window_id] = graph

        return result

    if isinstance(graphs, list):

        result = {}

        for graph in graphs:

            if not isinstance(
                graph,
                dict
            ):

                continue

            if "window_id" not in graph:

                continue

            window_id = int(
                graph["window_id"]
            )

            result[window_id] = graph

        return result

    raise TypeError(
        f"Unsupported graph container type: "
        f"{type(graphs)}"
    )


def verify_graph(graph):

    required = [
        "window_id",
        "nodes",
        "node_features",
        "adjacency"
    ]

    for key in required:

        if key not in graph:

            raise ValueError(
                f"Graph is missing required field: {key}"
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

        raise ValueError(
            f"Node features must be 2-D, "
            f"got {node_features.shape}"
        )

    if adjacency.ndim != 2:

        raise ValueError(
            f"Adjacency must be 2-D, "
            f"got {adjacency.shape}"
        )

    if adjacency.shape[0] != adjacency.shape[1]:

        raise ValueError(
            f"Adjacency must be square, "
            f"got {adjacency.shape}"
        )

    if adjacency.shape[0] != node_features.shape[0]:

        raise ValueError(
            "Node count does not match "
            "adjacency dimensions"
        )

    if node_features.shape[1] != 6:

        raise ValueError(
            f"Expected graph feature dimension 6, "
            f"got {node_features.shape[1]}"
        )


def build_sequences(
    states,
    window_ids,
    graph_dict
):

    aligned = []

    for i, window_id in enumerate(
        window_ids
    ):

        try:

            wid = int(
                window_id
            )

        except Exception:

            continue

        if wid in graph_dict:

            aligned.append(
                (i, wid)
            )

    if len(aligned) <= SEQUENCE_LENGTH:

        raise ValueError(
            f"Not enough aligned states. "
            f"Found {len(aligned)}."
        )

    print(
        f"Aligned items     : "
        f"{len(aligned):,}"
    )

    state_sequences = []
    targets = []
    sequence_graphs = []
    target_windows = []

    for start in range(
        0,
        len(aligned) - SEQUENCE_LENGTH
    ):

        sequence_part = aligned[
            start:
            start + SEQUENCE_LENGTH
        ]

        target_index = (
            start + SEQUENCE_LENGTH
        )

        state_indices = [
            item[0]
            for item in sequence_part
        ]

        graph_ids = [
            item[1]
            for item in sequence_part
        ]

        target_state_index = aligned[
            target_index
        ][0]

        target_window = aligned[
            target_index
        ][1]

        graphs_for_sequence = [
            graph_dict[gid]
            for gid in graph_ids
        ]

        for graph in graphs_for_sequence:

            verify_graph(graph)

        state_sequences.append(
            states[state_indices]
        )

        targets.append(
            states[target_state_index]
        )

        sequence_graphs.append(
            graphs_for_sequence
        )

        target_windows.append(
            target_window
        )

    X = np.asarray(
        state_sequences,
        dtype=np.float32
    )

    y = np.asarray(
        targets,
        dtype=np.float32
    )

    target_windows = np.asarray(
        target_windows,
        dtype=np.int64
    )

    return (
        X,
        y,
        sequence_graphs,
        target_windows
    )


def save_outputs(
    X,
    y,
    graph_sequences,
    target_windows
):

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_STATES,
        X=X,
        y=y,
        target_windows=target_windows
    )

    with open(
        OUTPUT_GRAPHS,
        "wb"
    ) as f:

        pickle.dump(
            graph_sequences,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    print()
    print("=" * 70)
    print("GRAPH SEQUENCES SAVED")
    print("=" * 70)

    print(
        f"State sequence shape : {X.shape}"
    )

    print(
        f"Target shape         : {y.shape}"
    )

    print(
        f"Graph sequences      : "
        f"{len(graph_sequences):,}"
    )

    print(
        f"Sequence length      : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"State dimension      : "
        f"{X.shape[2]}"
    )

    print(
        f"Graph steps/sequence : "
        f"{len(graph_sequences[0])}"
    )

    print()

    print(
        f"States saved to      : "
        f"{OUTPUT_STATES}"
    )

    print(
        f"Graphs saved to      : "
        f"{OUTPUT_GRAPHS}"
    )


def verify_saved_data(
    X,
    y,
    graph_sequences,
    target_windows
):

    print()
    print("=" * 70)
    print("VERIFYING GRAPH SEQUENCES")
    print("=" * 70)

    assert X.ndim == 3

    assert y.ndim == 2

    assert X.shape[1] == SEQUENCE_LENGTH

    assert X.shape[2] == 33

    assert y.shape[1] == 33

    assert len(graph_sequences) == len(X)

    assert len(target_windows) == len(X)

    for sequence in graph_sequences:

        assert len(sequence) == SEQUENCE_LENGTH

        for graph in sequence:

            verify_graph(graph)

    print(
        f"Sequences            : "
        f"{len(X):,}"
    )

    print(
        f"State sequence shape : "
        f"{X.shape}"
    )

    print(
        f"Target shape         : "
        f"{y.shape}"
    )

    print(
        f"Graph sequences      : "
        f"{len(graph_sequences):,}"
    )

    if len(graph_sequences) > 0:

        first = graph_sequences[0]

        last = graph_sequences[-1]

        print(
            "First graph IDs      : "
            f"{first[0]['window_id']}.."
            f"{first[-1]['window_id']}"
        )

        print(
            "Last graph IDs       : "
            f"{last[0]['window_id']}.."
            f"{last[-1]['window_id']}"
        )

        print(
            "First target window  : "
            f"{target_windows[0]}"
        )

    print()

    print(
        "GRAPH SEQUENCE TEST: PASSED"
    )


def main():

    print("=" * 70)
    print("ARJUN GRAPH-AWARE SEQUENCE BUILDER")
    print("=" * 70)

    print(
        f"State file           : "
        f"{STATE_FILE}"
    )

    print(
        f"Graph file           : "
        f"{GRAPH_FILE}"
    )

    print(
        f"Sequence length      : "
        f"{SEQUENCE_LENGTH}"
    )

    print()

    states, labels, window_ids = (
        load_state_data()
    )

    graphs = load_graph_data()

    graph_dict = normalize_graph_container(
        graphs
    )

    print(
        f"Graph dictionary    : "
        f"{len(graph_dict):,}"
    )

    (
        X,
        y,
        graph_sequences,
        target_windows
    ) = build_sequences(
        states,
        window_ids,
        graph_dict
    )

    save_outputs(
        X,
        y,
        graph_sequences,
        target_windows
    )

    verify_saved_data(
        X,
        y,
        graph_sequences,
        target_windows
    )


if __name__ == "__main__":
    main()