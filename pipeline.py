from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.csv_loader import load_csv
from ingestion.pcap_loader import load_pcap
from ingestion.zeek_loader import load_zeek_conn

from preprocessing.parser import standardize_columns
from preprocessing.cleaner import clean_data
from preprocessing.timestamp_aligner import align_timestamps
from preprocessing.pcap_processor import add_packet_derived_features

from features.flow_features import add_flow_features
from features.packet_features import add_packet_features
from features.temporal_features import add_temporal_features
from features.behavioral_features import add_behavioral_features
from features.contextual_features import add_contextual_features
from features.graph_builder import build_window_graphs

from world_model.state_encoder import NetworkStateEncoder
from world_model.sequence_builder import build_sequences


PCAP_EXTENSIONS = {
    ".pcap",
    ".pcapng",
    ".cap",
}


def _load_input(input_path):
    """
    Load supported ARJUN telemetry formats.
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_csv(str(path))

    if suffix in PCAP_EXTENSIONS:
        return load_pcap(str(path))

    if suffix in {
        ".log",
        ".tsv",
        ".txt",
    }:
        return load_zeek_conn(str(path))

    raise ValueError(
        f"Unsupported input format: {suffix}. "
        "Supported formats are CSV, PCAP, PCAPNG, CAP, "
        "LOG, TSV and TXT."
    )


def _ensure_dataframe(data):
    """
    Ensure loader output is a DataFrame.
    """

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, list):
        return pd.DataFrame(data)

    if isinstance(data, dict):
        return pd.DataFrame(data)

    raise TypeError(
        f"Expected DataFrame-compatible data, "
        f"received {type(data)}."
    )


def _ensure_basic_columns(df):
    """
    Add safe defaults for datasets that do not provide
    all network-flow fields.
    """

    df = df.copy()

    defaults = {
        "src_ip": "unknown",
        "dst_ip": "unknown",
        "src_port": 0,
        "dst_port": 0,
        "protocol": 0,
        "bytes": 0.0,
        "packets": 1.0,
        "duration": 0.0,
        "ttl": 0.0,
        "tcp_window": 0.0,
        "payload_size": 0.0,
        "packet_length": 0.0,
        "flags": "",
        "Label": "Benign",
    }

    for column, default in defaults.items():

        if column not in df.columns:
            df[column] = default

    return df


def _apply_features(df):
    """
    Apply all ARJUN feature-engineering stages.
    """

    result = df.copy()

    functions = [
        add_flow_features,
        add_packet_features,
        add_temporal_features,
        add_behavioral_features,
        add_contextual_features,
    ]

    for function in functions:

        try:

            output = function(result)

            if isinstance(output, pd.DataFrame):
                result = output

        except Exception as exc:

            raise RuntimeError(
                f"Feature engineering failed in "
                f"{function.__name__}: {exc}"
            ) from exc

    return result


def _convert_graphs_to_tensors(graphs):
    """
    Convert graph NumPy arrays into the tensor format
    expected by HybridWorldModel.
    """

    import torch

    converted = {}

    for window_id, graph in graphs.items():

        node_features = np.asarray(
            graph["node_features"],
            dtype=np.float32,
        )

        adjacency = np.asarray(
            graph["adjacency"],
            dtype=np.float32,
        )

        converted[int(window_id)] = {
            "nodes": graph.get(
                "nodes",
                [],
            ),

            "node_features": torch.tensor(
                node_features,
                dtype=torch.float32,
            ),

            "adjacency": torch.tensor(
                adjacency,
                dtype=torch.float32,
            ),
        }

    return converted


def _build_graph_sequences(
    graphs,
    sequence_length,
    window_ids,
):
    """
    Align graph windows with temporal state sequences.

    For:

        S1 S2 S3 -> S4

    the corresponding graph sequence is:

        G1 G2 G3
    """

    graph_sequences = []

    for index in range(
        sequence_length,
        len(window_ids),
    ):

        history_window_ids = window_ids[
            index - sequence_length:index
        ]

        sequence_graphs = []

        for window_id in history_window_ids:

            window_id = int(window_id)

            if window_id in graphs:

                graph = graphs[window_id]

            else:

                # Safe empty graph.
                import torch

                graph = {
                    "nodes": [],

                    "node_features": torch.zeros(
                        (0, 6),
                        dtype=torch.float32,
                    ),

                    "adjacency": torch.zeros(
                        (0, 0),
                        dtype=torch.float32,
                    ),
                }

            sequence_graphs.append(
                graph
            )

        graph_sequences.append(
            sequence_graphs
        )

    return graph_sequences


def run_pipeline(
    input_path,
    window_seconds=5,
    sequence_length=5,
):
    """
    Complete ARJUN telemetry pipeline.

    Pipeline:

        Input
          ↓
        Parsing
          ↓
        Cleaning
          ↓
        Timestamp alignment
          ↓
        Feature engineering
          ↓
        Network state representation
          ↓
        Communication graphs
          ↓
        Temporal sequences

    Returns
    -------
    dict
        features
        data
        states
        window_ids
        X
        y
        target_windows
        graph_sequences
        graphs
        feature_names
        sequence_length
        state_dimension
    """

    if window_seconds < 1:
        raise ValueError(
            "window_seconds must be >= 1."
        )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be >= 1."
        )

    # ==================================================
    # 1. INGESTION
    # ==================================================

    raw = _load_input(
        input_path
    )

    df = _ensure_dataframe(
        raw
    )

    if df.empty:
        raise ValueError(
            "Input contains no records."
        )

    # ==================================================
    # 2. PARSING & BASIC COLUMNS
    # ==================================================

    df = standardize_columns(
        df
    )

    df = _ensure_basic_columns(
        df
    )

    # ==================================================
    # 3. CLEANING
    # ==================================================

    df = clean_data(
        df
    )

    if df.empty:
        raise ValueError(
            "No records remain after cleaning."
        )

    # ==================================================
    # 4. PCAP PROCESSING
    # ==================================================

    suffix = Path(
        input_path
    ).suffix.lower()

    if suffix in PCAP_EXTENSIONS:

        # The parser canonicalizes tcp_flags -> flags.
        # The packet processor expects tcp_flags.
        if (
            "flags" in df.columns
            and "tcp_flags" not in df.columns
        ):
            df["tcp_flags"] = df["flags"]

        df = add_packet_derived_features(
            df
        )

    # ==================================================
    # 6. TIMESTAMP ALIGNMENT
    # ==================================================

    df = align_timestamps(
        df,
        window_seconds=window_seconds,
    )

    if df.empty:
        raise ValueError(
            "No records remain after timestamp alignment."
        )

    # ==================================================
    # 7. FEATURE ENGINEERING
    # ==================================================

    df = _apply_features(
        df
    )

    # ==================================================
    # 8. NETWORK STATE REPRESENTATION
    # ==================================================

    encoder = NetworkStateEncoder()

    states, state_window_ids = (
        encoder.encode_dataset(
            df
        )
    )

    states = np.asarray(
        states,
        dtype=np.float32,
    )

    states = np.nan_to_num(
        states,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if states.ndim != 2:
        raise RuntimeError(
            "Network states must have shape "
            "(num_windows, state_dimension)."
        )

    if len(states) < 2:
        raise ValueError(
            "At least two temporal network states "
            "are required."
        )

    # The encoder returns one window ID per state.
    window_ids = np.asarray(
        state_window_ids,
        dtype=np.int64,
    )

    # ==================================================
    # 9. BUILD COMMUNICATION GRAPHS
    # ==================================================

    graphs = build_window_graphs(
        df
    )

    graphs = _convert_graphs_to_tensors(
        graphs
    )

    # ==================================================
    # 10. TEMPORAL SEQUENCES
    # ==================================================

    effective_sequence_length = min(
        sequence_length,
        len(states) - 1,
    )

    seq_result = build_sequences(
        states=states,
        window_ids=window_ids,
        sequence_length=effective_sequence_length,
    )

    if isinstance(seq_result, dict):
        X = seq_result["X"]
        y = seq_result["y"]
        target_windows = seq_result["target_windows"]
    else:
        X, y, target_windows = seq_result[:3]

    # ==================================================
    # 11. GRAPH SEQUENCES
    # ==================================================

    graph_sequences = (
        _build_graph_sequences(
            graphs=graphs,
            sequence_length=(
                effective_sequence_length
            ),
            window_ids=window_ids,
        )
    )

    if len(graph_sequences) != len(X):
        raise RuntimeError(
            "Graph sequences are not aligned "
            "with state sequences."
        )

    return {
        "features": df,

        "data": df,

        "states": states,

        "window_ids": window_ids,

        "X": X,

        "y": y,

        "target_windows": target_windows,

        "graphs": graphs,

        "graph_sequences": graph_sequences,

        "sequence_length": (
            effective_sequence_length
        ),

        "state_dimension": int(
            states.shape[1]
        ),

        "feature_names": list(
            encoder.feature_names
        ),
    }