"""
ARJUN - Aligned State + Graph Dataset Builder

Builds the authoritative state + graph dataset used by the
ARJUN World Model.

Pipeline
--------
Raw CIC-IDS-2018 CSV
        |
        v
Robust parser
        |
        v
Cleaning
        |
        v
Chronological sorting
        |
        +-------------------+
        |                   |
        v                   v
Network state S_t       Graph G_t
        |                   |
        +---------+---------+
                  |
                  v
        Same temporal window

Outputs
-------
data/processed/training_states.npz
data/processed/training_states_metadata.json
data/processed/training_graphs.pkl
data/processed/training_graphs_metadata.json

Important
---------
- Timestamp parsing is handled ONLY by preprocessing/parser.py.
- This file never reparses timestamps with dayfirst=True.
- Each complete CSV is sorted chronologically before windowing.
- State and graph are generated from the same temporal window.
- Missing telemetry is not fabricated.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.parser import parse_dataframe
from preprocessing.cleaner import clean_data
from preprocessing.pcap_processor import add_packet_derived_features

from features.flow_features import add_flow_features
from features.packet_features import add_packet_features
from features.temporal_features import add_temporal_features
from features.behavioral_features import add_behavioral_features
from features.contextual_features import add_contextual_features

from world_model.state_encoder import NetworkStateEncoder
from features.graph_builder import build_window_graphs


# ================================================================
# Configuration
# ================================================================

DEFAULT_DATASET_DIR = (
    Path("data")
    / "raw"
    / "dataset"
)

PROCESSED_DIR = (
    Path("data")
    / "processed"
)

DEFAULT_WINDOW_SECONDS = 5

DEFAULT_MAX_STATES = None  # No limit

MAX_REASONABLE_SPAN_HOURS = 72.0


# ================================================================
# Utility functions
# ================================================================

def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def is_benign_label(value) -> bool:
    """
    Return True when a label represents benign traffic.
    """

    if pd.isna(value):
        return True

    text = (
        str(value)
        .strip()
        .lower()
    )

    return text in {
        "",
        "benign",
        "normal",
        "0",
        "0.0",
        "nan",
        "none",
    }


def make_binary_label(
    series: pd.Series,
) -> pd.Series:
    """
    Convert:

        benign -> 0
        attack -> 1
    """

    return (
        ~series.apply(
            is_benign_label
        )
    ).astype(
        np.int64
    )


def safe_numeric(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Safely obtain a numeric column.

    Missing columns become zeros.
    Invalid/infinite values become zero.
    """

    if column not in df.columns:

        return pd.Series(
            0.0,
            index=df.index,
            dtype=np.float64,
        )

    return (
        pd.to_numeric(
            df[column],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )


# ================================================================
# Graph preparation
# ================================================================

def ensure_graph_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure fields expected by graph_builder exist.

    Missing telemetry is represented by zero where a numeric
    aggregate is required.

    We do NOT fabricate:
        - IP addresses
        - TTL
        - payload
        - TCP flags
        - retransmissions
    """

    result = df.copy()

    # ------------------------------------------------------------
    # Bytes
    # ------------------------------------------------------------

    if "bytes" not in result.columns:

        result["bytes"] = (
            safe_numeric(
                result,
                "fwd_bytes",
            )
            + safe_numeric(
                result,
                "bwd_bytes",
            )
        )

    # ------------------------------------------------------------
    # Packets
    # ------------------------------------------------------------

    if "packets" not in result.columns:

        result["packets"] = (
            safe_numeric(
                result,
                "fwd_packets",
            )
            + safe_numeric(
                result,
                "bwd_packets",
            )
        )

    # ------------------------------------------------------------
    # Destination port
    # ------------------------------------------------------------

    if "dst_port" not in result.columns:

        result["dst_port"] = 0.0

    # ------------------------------------------------------------
    # Source port
    # ------------------------------------------------------------

    if "src_port" not in result.columns:

        result["src_port"] = 0.0

    # ------------------------------------------------------------
    # SYN count
    # ------------------------------------------------------------

    if "syn_count" not in result.columns:

        if "flag_syn" in result.columns:

            result["syn_count"] = (
                safe_numeric(
                    result,
                    "flag_syn",
                )
            )

        else:

            result["syn_count"] = 0.0

    return result


# ================================================================
# Timestamp validation
# ================================================================

def validate_dataframe_timestamps(
    df: pd.DataFrame,
    stage: str,
) -> None:
    """
    Perform strict timestamp validation.

    Raises an exception instead of allowing corrupted temporal
    data to continue into the world-model dataset.
    """

    if "timestamp" not in df.columns:

        raise RuntimeError(
            f"{stage}: timestamp column is missing."
        )

    timestamp = df["timestamp"]

    if not pd.api.types.is_datetime64_any_dtype(
        timestamp
    ):

        raise RuntimeError(
            f"{stage}: timestamp is not datetime64."
        )

    valid = timestamp.dropna()

    if valid.empty:

        raise RuntimeError(
            f"{stage}: no valid timestamps remain."
        )

    years = valid.dt.year

    valid_year_ratio = float(
        years.between(
            2000,
            2100,
        ).mean()
    )

    if valid_year_ratio < 0.95:

        raise RuntimeError(
            f"{stage}: only "
            f"{valid_year_ratio * 100:.2f}% of timestamps "
            "are within 2000-2100."
        )

    minimum = valid.min()
    maximum = valid.max()

    span_seconds = (
        maximum - minimum
    ).total_seconds()

    if span_seconds < 0:

        raise RuntimeError(
            f"{stage}: negative timestamp span."
        )

    span_hours = (
        span_seconds / 3600.0
    )

    if (
        span_hours
        > MAX_REASONABLE_SPAN_HOURS
    ):

        raise RuntimeError(
            f"{stage}: unreasonable timestamp span "
            f"{span_hours:.2f} hours. "
            "Temporal window creation stopped."
        )


# ================================================================
# CIC timestamp format handling
# ================================================================

def apply_cic_timestamp_format(
    raw_df: pd.DataFrame,
    source_file: str | None = None,
) -> pd.DataFrame:
    """
    Resolve the known CIC-IDS-2018 timestamp ambiguity.

    Most February CIC files use MM/DD/YYYY, while the March files
    03-01-2018.csv and 03-02-2018.csv contain DD/MM/YYYY values:

        03-01-2018.csv -> 01/03/2018 -> March 1, 2018
        03-02-2018.csv -> 02/03/2018 -> March 2, 2018

    The generic parser remains authoritative for all other inputs.
    For these two source files only, timestamps are normalized to ISO
    strings before parse_dataframe() is called.
    """

    if raw_df is None or raw_df.empty:
        return raw_df

    if source_file is None:
        return raw_df

    filename = Path(str(source_file)).name.lower()

    if filename not in {
        "03-01-2018.csv",
        "03-02-2018.csv",
    }:
        return raw_df

    result = raw_df.copy()

    # Find the raw Timestamp column without assuming exact casing.
    timestamp_column = None

    for column in result.columns:
        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if normalized in {
            "timestamp",
            "datetime",
            "date_time",
        }:
            timestamp_column = column
            break

    if timestamp_column is None:
        return result

    raw_timestamp = (
        result[timestamp_column]
        .astype("string")
        .str.strip()
    )

    parsed = pd.to_datetime(
        raw_timestamp,
        format="%d/%m/%Y %H:%M:%S.%f",
        errors="coerce",
    )

    missing = parsed.isna()

    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            raw_timestamp.loc[missing],
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )

    missing = parsed.isna()

    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            raw_timestamp.loc[missing],
            format="%d/%m/%Y %H:%M",
            errors="coerce",
        )

    # Preserve invalid values as NaT so the existing validation /
    # cleaning path can handle them.
    result[timestamp_column] = parsed

    return result


# ================================================================
# Prepare one complete source dataframe
# ================================================================

def prepare_dataframe(
    raw_df: pd.DataFrame,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    source_file: str | None = None,
) -> pd.DataFrame:
    """
    Parse, clean, sort, feature-engineer and window one complete
    source CSV.

    Timestamp rule
    --------------
    parse_dataframe() is the single authority for timestamp
    interpretation.

    No later pd.to_datetime() call is performed.
    """

    if (
        raw_df is None
        or raw_df.empty
    ):

        return pd.DataFrame()

    # ------------------------------------------------------------
    # Resolve known CIC source-file timestamp ambiguity
    # ------------------------------------------------------------

    raw_df = apply_cic_timestamp_format(
        raw_df,
        source_file=source_file,
    )

    # ------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------

    df = parse_dataframe(
        raw_df
    )

    if df.empty:

        return df

    # ------------------------------------------------------------
    # Parser timestamp validation
    # ------------------------------------------------------------

    validate_dataframe_timestamps(
        df,
        "After parser",
    )

    parser_timestamp = (
        df["timestamp"].copy()
    )

    print(
        "  Parser timestamp check : OK"
    )

    print(
        "  Parser timestamp range : "
        f"{parser_timestamp.min()} -> "
        f"{parser_timestamp.max()}"
    )

    # ------------------------------------------------------------
    # Clean
    # ------------------------------------------------------------

    df = clean_data(
        df,
        label_column="label",
    )

    if df.empty:

        return df

    # ------------------------------------------------------------
    # Protect parser-generated timestamp
    # ------------------------------------------------------------

    df["timestamp"] = (
        parser_timestamp
        .reindex(
            df.index
        )
    )

    # ------------------------------------------------------------
    # Remove invalid timestamps
    # ------------------------------------------------------------

    invalid_count = int(
        df["timestamp"]
        .isna()
        .sum()
    )

    if invalid_count:

        print(
            "  Removing invalid timestamps: "
            f"{invalid_count:,}"
        )

        df = df.dropna(
            subset=[
                "timestamp"
            ]
        )

    if df.empty:

        return df

    # ------------------------------------------------------------
    # Timestamp validation after cleaning
    # ------------------------------------------------------------

    validate_dataframe_timestamps(
        df,
        "After cleaning",
    )

    # ------------------------------------------------------------
    # Chronological ordering
    # ------------------------------------------------------------

    df = (
        df
        .sort_values(
            "timestamp",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------------
    # Final timestamp range
    # ------------------------------------------------------------

    minimum = df[
        "timestamp"
    ].min()

    maximum = df[
        "timestamp"
    ].max()

    span_seconds = (
        maximum - minimum
    ).total_seconds()

    print(
        "  Final timestamp check  : OK"
    )

    print(
        "  Final timestamp range  : "
        f"{minimum} -> {maximum}"
    )

    print(
        "  Final time span        : "
        f"{span_seconds / 3600.0:.2f} hours"
    )

    # ------------------------------------------------------------
    # Duration safety
    # ------------------------------------------------------------

    if "duration" in df.columns:

        df["duration"] = (
            pd.to_numeric(
                df["duration"],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .clip(
                lower=0.0
            )
        )

    # ------------------------------------------------------------
    # IAT safety
    # ------------------------------------------------------------

    for column in (
        "iat_seconds",
        "src_iat_seconds",
    ):

        if column not in df.columns:
            continue

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
            .clip(
                lower=0.0
            )
        )

    # ------------------------------------------------------------
    # Packet-derived features
    # ------------------------------------------------------------

    df = add_packet_derived_features(
        df
    )

    # ------------------------------------------------------------
    # Flow features
    # ------------------------------------------------------------

    df = add_flow_features(
        df
    )

    # ------------------------------------------------------------
    # Reference time
    # ------------------------------------------------------------

    reference_time = (
        df["timestamp"].iloc[0]
    )

    elapsed_seconds = (
        df["timestamp"]
        - reference_time
    ).dt.total_seconds()

    elapsed_seconds = (
        elapsed_seconds
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
        .clip(
            lower=0.0
        )
    )

    # ------------------------------------------------------------
    # Local temporal window
    # ------------------------------------------------------------

    if window_seconds <= 0:

        raise ValueError(
            "window_seconds must be greater than zero."
        )

    df["local_window_id"] = (
        np.floor(
            elapsed_seconds
            / float(window_seconds)
        )
        .astype(
            np.int64
        )
    )

    # ------------------------------------------------------------
    # Feature modules require window_id
    # ------------------------------------------------------------

    df["window_id"] = (
        df["local_window_id"]
    )

    # ------------------------------------------------------------
    # Packet features
    # ------------------------------------------------------------

    df = add_packet_features(
        df
    )

    # ------------------------------------------------------------
    # Temporal features
    # ------------------------------------------------------------

    df = add_temporal_features(
        df
    )

    # ------------------------------------------------------------
    # Behavioural features
    # ------------------------------------------------------------

    df = add_behavioral_features(
        df
    )

    # ------------------------------------------------------------
    # Contextual features
    # ------------------------------------------------------------

    df = add_contextual_features(
        df
    )

    # ------------------------------------------------------------
    # Graph columns
    # ------------------------------------------------------------

    df = ensure_graph_columns(
        df
    )

    # ------------------------------------------------------------
    # Binary label
    # ------------------------------------------------------------

    if "label" in df.columns:

        df["binary_label"] = (
            make_binary_label(
                df["label"]
            )
        )

    else:

        df["binary_label"] = 0

    # ------------------------------------------------------------
    # Numerical safety
    # ------------------------------------------------------------

    numeric_columns = [

        "duration",

        "bytes",
        "packets",

        "fwd_packets",
        "bwd_packets",

        "fwd_bytes",
        "bwd_bytes",

        "src_port",
        "dst_port",

        "protocol",

        "packet_length",

        "ttl",
        "tcp_window",
        "payload_size",

        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "psh_count",
        "urg_count",

        "fragment",
        "retransmission",

        "iat_seconds",
        "src_iat_seconds",
    ]

    for column in numeric_columns:

        if column not in df.columns:
            continue

        values = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        if column != "protocol":

            values = values.clip(
                lower=0.0
            )

        df[column] = values

    # ------------------------------------------------------------
    # Final timestamp protection
    # ------------------------------------------------------------

    validate_dataframe_timestamps(
        df,
        "After feature engineering",
    )

    return df


# ================================================================
# Build one state + graph window
# ================================================================

def build_window(
    group: pd.DataFrame,
    global_state_id: int,
    source_file: str,
    local_window_id: int,
    encoder: NetworkStateEncoder,
):
    """
    Build S_t and G_t from exactly the same temporal window.
    """

    group = group.copy()

    global_state_id = int(
        global_state_id
    )

    local_window_id = int(
        local_window_id
    )

    # ------------------------------------------------------------
    # Global ID
    # ------------------------------------------------------------

    group["window_id"] = (
        global_state_id
    )

    # ------------------------------------------------------------
    # State
    # ------------------------------------------------------------

    state = encoder.encode_window(
        group
    )

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if state.ndim != 1:

        raise ValueError(
            "State must be a 1-D vector. "
            f"Got shape {state.shape}."
        )

    if not np.isfinite(
        state
    ).all():

        raise ValueError(
            "State contains NaN or Inf."
        )

    # ------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------

    graph_dict = build_window_graphs(
        group
    )

    if not graph_dict:

        raise RuntimeError(
            "Graph construction returned no graph "
            f"for {source_file}, "
            f"window {local_window_id}."
        )

    # The group contains exactly one temporal window.
    #
    # Therefore the graph dictionary should contain one graph.
    # We deliberately use the sole graph instead of depending on
    # whether graph_builder uses an integer or string window key.

    if len(graph_dict) == 1:

        graph = next(
            iter(
                graph_dict.values()
            )
        )

    else:

        # Try the current global ID.
        graph = graph_dict.get(
            global_state_id
        )

        if graph is None:

            graph = graph_dict.get(
                str(global_state_id)
            )

        if graph is None:

            # Try the local window ID.
            graph = graph_dict.get(
                local_window_id
            )

        if graph is None:

            graph = graph_dict.get(
                str(local_window_id)
            )

        if graph is None:

            raise RuntimeError(
                "Unable to locate graph for "
                f"global window {global_state_id}, "
                f"local window {local_window_id}."
            )

    # ------------------------------------------------------------
    # Graph provenance
    # ------------------------------------------------------------

    graph["window_id"] = (
        global_state_id
    )

    graph["global_state_id"] = (
        global_state_id
    )

    graph["local_window_id"] = (
        local_window_id
    )

    graph["source_file"] = (
        str(source_file)
    )

    # ------------------------------------------------------------
    # Graph validation
    # ------------------------------------------------------------

    if "node_features" not in graph:

        raise RuntimeError(
            "Graph is missing node_features."
        )

    if "adjacency" not in graph:

        raise RuntimeError(
            "Graph is missing adjacency."
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
            "Graph node_features must be 2-D. "
            f"Got {node_features.shape}."
        )

    if node_features.shape[1] != 6:

        raise ValueError(
            "Expected graph feature dimension 6. "
            f"Got {node_features.shape}."
        )

    expected_adjacency_shape = (
        node_features.shape[0],
        node_features.shape[0],
    )

    if adjacency.shape != (
        expected_adjacency_shape
    ):

        raise ValueError(
            "Graph adjacency shape mismatch. "
            f"Expected {expected_adjacency_shape}, "
            f"got {adjacency.shape}."
        )

    if not np.isfinite(
        node_features
    ).all():

        raise ValueError(
            "Graph node features contain NaN/Inf."
        )

    if not np.isfinite(
        adjacency
    ).all():

        raise ValueError(
            "Graph adjacency contains NaN/Inf."
        )

    # ------------------------------------------------------------
    # Attack label
    # ------------------------------------------------------------

    if "binary_label" in group.columns:

        numeric_labels = pd.to_numeric(
            group["binary_label"],
            errors="coerce",
        ).fillna(0.0)

        attack_label = int(
            numeric_labels.max()
        )

    else:

        attack_label = 0

    # ------------------------------------------------------------
    # Window timestamps
    # ------------------------------------------------------------

    start_time = (
        group["timestamp"].min()
    )

    end_time = (
        group["timestamp"].max()
    )

    return {

        "state": state,

        "graph": graph,

        "label": attack_label,

        "source_file": (
            str(source_file)
        ),

        "local_window_id": (
            local_window_id
        ),

        "window_start": (
            start_time.isoformat()
            if not pd.isna(start_time)
            else None
        ),

        "window_end": (
            end_time.isoformat()
            if not pd.isna(end_time)
            else None
        ),
    }


# ================================================================
# Read source CSV
# ================================================================

def read_source_csv(
    csv_path: Path,
) -> pd.DataFrame:
    """
    Read one CIC source CSV in chunks and combine the chunks.

    CIC-IDS files can contain more than one million rows. Reading
    the complete CSV with a single pd.read_csv() call can consume
    excessive memory and may become extremely slow on some systems.

    Reading in chunks reduces peak parser memory usage while still
    returning the complete DataFrame so the existing chronological
    processing logic remains unchanged.
    """

    print(
        "  Reading CSV in memory-safe chunks..."
    )

    chunk_size = 100_000

    chunks = []
    total_rows = 0

    try:

        reader = pd.read_csv(
            csv_path,
            low_memory=False,
            chunksize=chunk_size,
        )

        for chunk_number, chunk in enumerate(
            reader,
            start=1,
        ):

            chunks.append(chunk)

            total_rows += len(chunk)

            print(
                f"\r  Reading chunk {chunk_number}: "
                f"{total_rows:,} rows",
                end="",
                flush=True,
            )

    except Exception:

        print()

        raise

    print()

    if not chunks:

        print(
            "  Raw rows: 0"
        )

        return pd.DataFrame()

    df = pd.concat(
        chunks,
        axis=0,
        ignore_index=True,
    )

    del chunks

    print(
        f"  Raw rows: {len(df):,}"
    )

    return df


# ================================================================
# Process one source file
# ================================================================

def process_source_file(
    csv_path: Path,
    encoder: NetworkStateEncoder,
    global_state_start: int,
    max_records,
    window_seconds: int,
):
    """
    Process one complete source CSV.

    Returns
    -------
    records
    next_global_id
    statistics
    """

    raw_df = read_source_csv(
        csv_path
    )

    if raw_df.empty:

        return (
            [],
            global_state_start,
            {},
        )

    # ------------------------------------------------------------
    # Prepare complete file
    # ------------------------------------------------------------

    df = prepare_dataframe(
        raw_df,
        window_seconds,
        source_file=csv_path.name,
    )

    del raw_df

    if df.empty:

        return (
            [],
            global_state_start,
            {},
        )

    # ------------------------------------------------------------
    # Timestamp summary
    # ------------------------------------------------------------

    min_time = (
        df["timestamp"].min()
    )

    max_time = (
        df["timestamp"].max()
    )

    span_seconds = (
        max_time - min_time
    ).total_seconds()

    print(
        f"  Parsed time range : "
        f"{min_time} -> {max_time}"
    )

    print(
        f"  Time span         : "
        f"{span_seconds / 3600.0:.2f} hours"
    )

    # ------------------------------------------------------------
    # Window IDs
    # ------------------------------------------------------------

    local_window_ids = np.sort(
        df[
            "local_window_id"
        ]
        .unique()
    )

    print(
        f"  Local windows     : "
        f"{len(local_window_ids):,}"
    )

    if len(local_window_ids):

        print(
            f"  Window ID range   : "
            f"{local_window_ids[0]} -> "
            f"{local_window_ids[-1]}"
        )

    # ------------------------------------------------------------
    # Generate records
    # ------------------------------------------------------------

    records = []

    global_state_id = int(
        global_state_start
    )

    grouped = df.groupby(
        "local_window_id",
        sort=True,
    )

    for local_window_id, group in grouped:

        if (
            max_records is not None
            and len(records) >= max_records
        ):

            break

        result = build_window(
            group=group,
            global_state_id=global_state_id,
            source_file=csv_path.name,
            local_window_id=int(
                local_window_id
            ),
            encoder=encoder,
        )

        result["segment_id"] = 0

        result["global_state_id"] = (
            global_state_id
        )

        records.append(
            result
        )

        global_state_id += 1

    statistics = {

        "source_file": (
            csv_path.name
        ),

        "rows": int(
            len(df)
        ),

        "windows": int(
            len(records)
        ),

        "min_timestamp": (
            min_time.isoformat()
        ),

        "max_timestamp": (
            max_time.isoformat()
        ),

        "span_seconds": float(
            span_seconds
        ),
    }

    return (
        records,
        global_state_id,
        statistics,
    )


# ================================================================
# Save state dataset
# ================================================================

def save_state_dataset(
    records,
    encoder,
    output_path: Path,
    metadata_path: Path,
    window_seconds: int,
):
    """
    Save the state dataset.
    """

    states = np.asarray(
        [
            record["state"]
            for record in records
        ],
        dtype=np.float32,
    )

    labels = np.asarray(
        [
            record["label"]
            for record in records
        ],
        dtype=np.int64,
    )

    window_ids = np.arange(
        len(records),
        dtype=np.int64,
    )

    segment_ids = np.asarray(
        [
            record.get(
                "segment_id",
                0,
            )
            for record in records
        ],
        dtype=np.int64,
    )

    global_state_ids = np.asarray(
        [
            record.get(
                "global_state_id",
                index,
            )
            for index, record
            in enumerate(records)
        ],
        dtype=np.int64,
    )

    local_window_ids = np.asarray(
        [
            record[
                "local_window_id"
            ]
            for record in records
        ],
        dtype=np.int64,
    )

    source_files = np.asarray(
        [
            record[
                "source_file"
            ]
            for record in records
        ],
        dtype=str,
    )

    window_starts = np.asarray(
        [
            record[
                "window_start"
            ]
            for record in records
        ],
        dtype=str,
    )

    window_ends = np.asarray(
        [
            record[
                "window_end"
            ]
            for record in records
        ],
        dtype=str,
    )

    feature_names = np.asarray(
        encoder.feature_names,
        dtype=str,
    )

    np.savez_compressed(
        output_path,

        X=states,

        states=states,

        labels=labels,

        y=labels,

        window_ids=window_ids,

        global_state_ids=global_state_ids,

        segment_ids=segment_ids,

        local_window_ids=local_window_ids,

        source_files=source_files,

        window_starts=window_starts,

        window_ends=window_ends,

        feature_names=feature_names,
    )

    metadata = {

        "samples": int(
            len(states)
        ),

        "state_dimension": (
            int(
                states.shape[1]
            )
            if states.ndim == 2
            else 0
        ),

        "feature_names": list(
            encoder.feature_names
        ),

        "window_seconds": int(
            window_seconds
        ),

        "description": (
            "ARJUN chronological "
            "network-state dataset."
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


# ================================================================
# Save graph dataset
# ================================================================

def save_graph_dataset(
    records,
    output_path: Path,
    metadata_path: Path,
):
    """
    Save graphs aligned with state IDs.
    """

    graphs = {}

    node_counts = []

    for state_id, record in enumerate(
        records
    ):

        graph = record[
            "graph"
        ]

        graph["window_id"] = int(
            state_id
        )

        graph["global_state_id"] = int(
            state_id
        )

        graphs[
            int(state_id)
        ] = graph

        node_counts.append(
            len(
                graph.get(
                    "nodes",
                    [],
                )
            )
        )

    with open(
        output_path,
        "wb",
    ) as file:

        pickle.dump(
            graphs,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metadata = {

        "graphs": int(
            len(graphs)
        ),

        "graph_feature_dimension": 6,

        "minimum_nodes": (
            int(
                min(node_counts)
            )
            if node_counts
            else 0
        ),

        "maximum_nodes": (
            int(
                max(node_counts)
            )
            if node_counts
            else 0
        ),

        "average_nodes": (
            float(
                np.mean(
                    node_counts
                )
            )
            if node_counts
            else 0.0
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


# ================================================================
# Main dataset builder
# ================================================================

def build_dataset(
    dataset_dir,
    max_states=DEFAULT_MAX_STATES,
    window_seconds=DEFAULT_WINDOW_SECONDS,
):
    """
    Build the aligned state + graph dataset.
    """

    dataset_dir = Path(
        dataset_dir
    )

    if not dataset_dir.exists():

        raise FileNotFoundError(
            "Dataset directory not found: "
            f"{dataset_dir.resolve()}"
        )

    csv_files = sorted(
        dataset_dir.glob(
            "*.csv"
        )
    )

    if not csv_files:

        raise FileNotFoundError(
            "No CSV files found in: "
            f"{dataset_dir.resolve()}"
        )

    if (
        max_states is not None
        and max_states <= 0
    ):

        raise ValueError(
            "max_states must be greater than zero "
            "or None."
        )

    if window_seconds <= 0:

        raise ValueError(
            "window_seconds must be greater than zero."
        )

    ensure_directory(
        PROCESSED_DIR
    )

    encoder = NetworkStateEncoder()

    expected_dimension = len(
        encoder.feature_names
    )

    print("=" * 70)

    print(
        "ARJUN ALIGNED STATE + GRAPH DATASET BUILDER"
    )

    print("=" * 70)

    print(
        "Dataset directory : "
        f"{dataset_dir.resolve()}"
    )

    print(
        "CSV files         : "
        f"{len(csv_files)}"
    )

    print(
        "Window size       : "
        f"{window_seconds} seconds"
    )

    print(
    "Maximum states    : "
    + str(
        max_states
        if max_states is not None
        else "ALL"
    )
)

    print(
        "State dimension   : "
        f"{expected_dimension}"
    )

    print()

    all_records = []

    global_state_id = 0

    file_statistics = []

    # ------------------------------------------------------------
    # Process source files
    # ------------------------------------------------------------

    for file_index, csv_path in enumerate(
        csv_files,
        start=1,
    ):

        if (
            max_states is not None
            and len(all_records)
            >= max_states
        ):

            break

        remaining = None

        if max_states is not None:

            remaining = (
                max_states
                - len(all_records)
            )

        print(
            f"[{file_index}/{len(csv_files)}] "
            f"{csv_path.name}"
        )

        (
            records,
            global_state_id,
            statistics,
        ) = process_source_file(
            csv_path=csv_path,
            encoder=encoder,
            global_state_start=global_state_id,
            max_records=remaining,
            window_seconds=window_seconds,
        )

        # --------------------------------------------------------
        # Segment information
        # --------------------------------------------------------

        for record in records:

            record["segment_id"] = (
                file_index - 1
            )

        all_records.extend(
            records
        )

        if statistics:

            file_statistics.append(
                statistics
            )

        print(
            f"  Generated states : "
            f"{len(records):,}"
        )

        print()

    # ------------------------------------------------------------
    # Exact final limit
    # ------------------------------------------------------------

    if max_states is not None:

        all_records = (
            all_records[
                :max_states
            ]
        )

    if not all_records:

        raise RuntimeError(
            "No training states were generated."
        )

    # ------------------------------------------------------------
    # Reassign final state IDs
    #
    # This guarantees:
    #
    # training_states index i
    #        ==
    # training_graphs key i
    # ------------------------------------------------------------

    for state_id, record in enumerate(
        all_records
    ):

        record["global_state_id"] = (
            state_id
        )

        record["graph"][
            "window_id"
        ] = state_id

        record["graph"][
            "global_state_id"
        ] = state_id

    # ------------------------------------------------------------
    # Build state matrix
    # ------------------------------------------------------------

    states = np.asarray(
        [
            record["state"]
            for record in all_records
        ],
        dtype=np.float32,
    )

    if states.ndim != 2:

        raise RuntimeError(
            "Invalid state matrix shape: "
            f"{states.shape}"
        )

    if states.shape[1] != expected_dimension:

        raise RuntimeError(
            "State dimension mismatch. "
            f"Generated {states.shape[1]}, "
            f"expected {expected_dimension}."
        )

    if not np.isfinite(
        states
    ).all():

        raise RuntimeError(
            "Final state dataset contains NaN or Inf."
        )

    # ------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------

    state_output = (
        PROCESSED_DIR
        / "training_states.npz"
    )

    state_metadata = (
        PROCESSED_DIR
        / "training_states_metadata.json"
    )

    graph_output = (
        PROCESSED_DIR
        / "training_graphs.pkl"
    )

    graph_metadata = (
        PROCESSED_DIR
        / "training_graphs_metadata.json"
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    save_state_dataset(
        records=all_records,
        encoder=encoder,
        output_path=state_output,
        metadata_path=state_metadata,
        window_seconds=window_seconds,
    )

    save_graph_dataset(
        records=all_records,
        output_path=graph_output,
        metadata_path=graph_metadata,
    )

    # ------------------------------------------------------------
    # Final statistics
    # ------------------------------------------------------------

    labels = np.asarray(
        [
            record["label"]
            for record in all_records
        ],
        dtype=np.int64,
    )

    node_counts = [

        len(
            record[
                "graph"
            ].get(
                "nodes",
                [],
            )
        )

        for record in all_records
    ]

    print("=" * 70)

    print(
        "DATASET BUILD COMPLETE"
    )

    print("=" * 70)

    print(
        "States             : "
        f"{len(all_records):,}"
    )

    print(
        "State dimension     : "
        f"{states.shape[1]}"
    )

    print(
        "Benign states       : "
        f"{int((labels == 0).sum()):,}"
    )

    print(
        "Attack states       : "
        f"{int((labels == 1).sum()):,}"
    )

    print(
        "Attack percentage   : "
        f"{100.0 * labels.mean():.2f}%"
    )

    print(
        "Average graph nodes : "
        f"{np.mean(node_counts):.2f}"
    )

    print(
        "Minimum graph nodes : "
        f"{min(node_counts)}"
    )

    print(
        "Maximum graph nodes : "
        f"{max(node_counts)}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        f"  {state_output}"
    )

    print(
        f"  {state_metadata}"
    )

    print(
        f"  {graph_output}"
    )

    print(
        f"  {graph_metadata}"
    )

    print()

    print(
        "STATE/GRAPH DATASET BUILD: PASSED"
    )


# ================================================================
# CLI
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build ARJUN aligned "
            "state + graph dataset."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(
            DEFAULT_DATASET_DIR
        ),
        help=(
            "Directory containing "
            "CIC-IDS-2018 CSV files."
        ),
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=DEFAULT_MAX_STATES,
        help=(
            "Maximum number of temporal "
            "states to generate."
        ),
    )

    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help=(
            "Temporal window size "
            "in seconds."
        ),
    )

    args = parser.parse_args()

    build_dataset(
        dataset_dir=args.dataset_dir,
        max_states=args.max_states,
        window_seconds=args.window_seconds,
    )


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":

    main()