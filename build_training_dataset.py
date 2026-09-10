"""
ARJUN - Optimized Large Scale Training Dataset Builder

Builds 33-dimensional Network State vectors from CIC-IDS-2018
flow CSV files.

The output is compatible with:

    world_model/state_encoder.py

Output:

    data/processed/training_states.npz
    data/processed/training_states_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ingestion.csv_loader import discover_csv_files, iter_csv
from preprocessing.cleaner import (
    clean_data,
    create_binary_attack_label,
    create_attack_type,
)
from preprocessing.parser import standardize_columns
from world_model.state_encoder import NetworkStateEncoder


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "data" / "raw" / "dataset"
PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_NPZ = PROCESSED_DIR / "training_states.npz"
OUTPUT_METADATA = (
    PROCESSED_DIR / "training_states_metadata.json"
)


# ============================================================
# Exact 33-dimensional state representation
# ============================================================

STATE_FEATURE_NAMES = [
    "flow_count",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_dst_ports",
    "total_bytes",
    "total_packets",
    "avg_duration",
    "avg_packet_length",
    "avg_ttl",
    "ttl_variance",
    "avg_tcp_window",
    "tcp_window_variance",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "fragment_count",
    "retransmission_count",
    "avg_payload_size",
    "payload_variance",
    "max_payload_size",
    "avg_iat",
    "iat_variance",
    "avg_src_iat",
    "src_iat_variance",
    "avg_bytes_per_packet",
    "avg_packets_per_second",
    "avg_bytes_per_second",
    "avg_unique_dst_ports",
    "avg_unique_dst_hosts",
    "port_scan_count",
]


# ============================================================
# Utility functions
# ============================================================

def numeric(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """
    Return a numeric Series.

    Missing columns become zeros.
    """

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype=np.float64,
        )

    values = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return values.fillna(default)


def ensure_numeric(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> None:

    if column not in df.columns:

        df[column] = default

    else:

        df[column] = numeric(
            df,
            column,
            default,
        )


def safe_mean(
    values: pd.Series,
) -> float:

    if values is None or len(values) == 0:
        return 0.0

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(values) == 0:
        return 0.0

    return float(
        values.mean()
    )


def safe_variance(
    values: pd.Series,
) -> float:

    if values is None or len(values) <= 1:
        return 0.0

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(values) <= 1:
        return 0.0

    value = values.var(
        ddof=0
    )

    if pd.isna(value):
        return 0.0

    return float(value)


def safe_unique(
    df: pd.DataFrame,
    column: str,
) -> float:

    if column not in df.columns:
        return 0.0

    values = df[column].dropna()

    if len(values) == 0:
        return 0.0

    return float(
        values.astype(str).nunique()
    )


# ============================================================
# Prepare flow dataframe
# ============================================================

def prepare_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare one cleaned chunk for state generation.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Canonical numeric columns
    # --------------------------------------------------------

    required_numeric = [
        "duration",
        "bytes",
        "packets",
        "fwd_bytes",
        "bwd_bytes",
        "fwd_packets",
        "bwd_packets",
        "packet_length",
        "ttl",
        "tcp_window",
        "payload_size",
        "fragment_flag",
        "retransmission",
        "bytes_per_packet",
        "packets_per_second",
        "bytes_per_second",
    ]

    for column in required_numeric:
        ensure_numeric(
            df,
            column,
        )

    # --------------------------------------------------------
    # Basic flow totals
    # --------------------------------------------------------

    fwd_bytes = numeric(
        df,
        "fwd_bytes",
    )

    bwd_bytes = numeric(
        df,
        "bwd_bytes",
    )

    fwd_packets = numeric(
        df,
        "fwd_packets",
    )

    bwd_packets = numeric(
        df,
        "bwd_packets",
    )

    bytes_values = numeric(
        df,
        "bytes",
    )

    packets_values = numeric(
        df,
        "packets",
    )

    # Fill total bytes from directional values if required.
    if float(
        bytes_values.abs().sum()
    ) == 0.0:

        bytes_values = (
            fwd_bytes
            + bwd_bytes
        )

        df["bytes"] = bytes_values

    # Fill total packets from directional values.
    if float(
        packets_values.abs().sum()
    ) == 0.0:

        packets_values = (
            fwd_packets
            + bwd_packets
        )

        df["packets"] = packets_values

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = numeric(
        df,
        "duration",
    )

    if len(duration) > 0:

        median_duration = float(
            duration.median()
        )

        # CIC-IDS duration is generally microseconds.
        if median_duration > 10000:

            duration = (
                duration
                / 1_000_000.0
            )

            df["duration"] = duration

    # --------------------------------------------------------
    # Packet length
    # --------------------------------------------------------

    packet_length = numeric(
        df,
        "packet_length",
    )

    if float(
        packet_length.abs().sum()
    ) == 0.0:

        packet_length = np.divide(
            bytes_values,
            packets_values,
            out=np.zeros(
                len(df),
                dtype=np.float64,
            ),
            where=packets_values > 0,
        )

        df["packet_length"] = (
            packet_length
        )

    # --------------------------------------------------------
    # Bytes per packet
    # --------------------------------------------------------

    bytes_per_packet = np.divide(
        bytes_values,
        packets_values,
        out=np.zeros(
            len(df),
            dtype=np.float64,
        ),
        where=packets_values > 0,
    )

    df["bytes_per_packet"] = (
        bytes_per_packet
    )

    # --------------------------------------------------------
    # Traffic rates
    # --------------------------------------------------------

    packets_per_second = np.divide(
        packets_values,
        duration,
        out=np.zeros(
            len(df),
            dtype=np.float64,
        ),
        where=duration > 0,
    )

    bytes_per_second = np.divide(
        bytes_values,
        duration,
        out=np.zeros(
            len(df),
            dtype=np.float64,
        ),
        where=duration > 0,
    )

    df["packets_per_second"] = (
        packets_per_second
    )

    df["bytes_per_second"] = (
        bytes_per_second
    )

    # --------------------------------------------------------
    # TCP flags
    # --------------------------------------------------------

    aliases = {
        "flag_syn": [
            "flag_syn",
            "syn",
            "syn_count",
        ],
        "flag_ack": [
            "flag_ack",
            "ack",
            "ack_count",
        ],
        "flag_fin": [
            "flag_fin",
            "fin",
            "fin_count",
        ],
        "flag_rst": [
            "flag_rst",
            "rst",
            "rst_count",
        ],
        "flag_psh": [
            "flag_psh",
            "psh",
            "psh_count",
        ],
        "flag_urg": [
            "flag_urg",
            "urg",
            "urg_count",
        ],
    }

    for target, candidates in aliases.items():

        found = None

        for candidate in candidates:

            if candidate in df.columns:
                found = candidate
                break

        if found is None:

            df[target] = 0.0

        else:

            df[target] = numeric(
                df,
                found,
            )

    # --------------------------------------------------------
    # Missing packet-level fields
    # --------------------------------------------------------

    packet_fields = [
        "ttl",
        "tcp_window",
        "payload_size",
        "fragment_flag",
        "retransmission",
    ]

    for column in packet_fields:

        ensure_numeric(
            df,
            column,
            0.0,
        )

    # --------------------------------------------------------
    # Identifiers
    #
    # Some CIC-IDS files do not have source/destination IP.
    # Keep them as NaN rather than fabricating values.
    # --------------------------------------------------------

    for column in [
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
    ]:

        if column not in df.columns:
            df[column] = np.nan

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            dayfirst=True,
        )

    return df


# ============================================================
# Fast temporal features
# ============================================================

def add_fast_temporal_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate IAT without the expensive groupby().diff()
    operation that previously caused the large dataset run
    to stall.

    We use timestamp differences after chronological sorting.

    Source-specific IAT is only calculated when real source IP
    information exists.
    """

    df = df.copy()

    if (
        "timestamp" not in df.columns
        or df.empty
    ):

        df["iat_seconds"] = 0.0
        df["src_iat_seconds"] = 0.0

        return df

    # --------------------------------------------------------
    # Sort once by timestamp.
    # --------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    timestamps = df[
        "timestamp"
    ]

    # --------------------------------------------------------
    # Global IAT
    # --------------------------------------------------------

    iat = (
        timestamps
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
    )

    iat = iat.clip(
        lower=0.0
    )

    df["iat_seconds"] = (
        iat.astype(np.float64)
    )

    # --------------------------------------------------------
    # Source IAT
    #
    # Only do this when source IPs actually exist.
    # --------------------------------------------------------

    if (
        "src_ip" in df.columns
        and df["src_ip"].notna().any()
    ):

        # Use a lightweight loop over source groups.
        #
        # This is substantially cheaper than repeatedly
        # creating intermediate DataFrames.
        src_iat = np.zeros(
            len(df),
            dtype=np.float64,
        )

        source_values = (
            df["src_ip"]
            .astype("string")
        )

        valid_positions = (
            source_values.notna()
            & source_values.ne("")
            & source_values.ne("<NA>")
        )

        if bool(
            valid_positions.any()
        ):

            valid_indices = np.flatnonzero(
                valid_positions.to_numpy()
            )

            source_array = (
                source_values
                .to_numpy()
            )

            timestamp_array = (
                timestamps
                .astype("int64")
                .to_numpy()
            )

            # Since dataframe is already timestamp-sorted,
            # positions belonging to the same source are
            # naturally chronological.
            last_seen = {}

            for position in valid_indices:

                source = source_array[
                    position
                ]

                current_time = (
                    timestamp_array[
                        position
                    ]
                )

                previous_time = (
                    last_seen.get(source)
                )

                if previous_time is not None:

                    difference = (
                        current_time
                        - previous_time
                    ) / 1_000_000_000.0

                    if difference > 0:
                        src_iat[position] = (
                            difference
                        )

                last_seen[source] = (
                    current_time
                )

        df["src_iat_seconds"] = (
            src_iat
        )

    else:

        df["src_iat_seconds"] = 0.0

    return df


# ============================================================
# Encode one window
# ============================================================

def encode_window_fast(
    window: pd.DataFrame,
    encoder: NetworkStateEncoder,
) -> np.ndarray:
    """
    Encode one window using NetworkStateEncoder.

    The expensive temporal preparation has already been done
    outside this function.
    """

    if window.empty:

        return np.zeros(
            33,
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Window behavioural features
    # --------------------------------------------------------

    unique_ports = safe_unique(
        window,
        "dst_port",
    )

    unique_hosts = safe_unique(
        window,
        "dst_ip",
    )

    window = window.copy()

    window[
        "unique_dst_ports_per_window"
    ] = unique_ports

    window[
        "unique_dst_hosts_per_window"
    ] = unique_hosts

    # Conservative flow-level port scan indicator.
    if unique_ports >= 10:

        window[
            "port_scan_signature"
        ] = 1.0

    else:

        window[
            "port_scan_signature"
        ] = 0.0

    # --------------------------------------------------------
    # Use the authoritative encoder.
    # --------------------------------------------------------

    state = encoder.encode_window(
        window
    )

    state = np.asarray(
        state,
        dtype=np.float32,
    )

    if state.shape != (33,):

        raise RuntimeError(
            "NetworkStateEncoder returned "
            f"{state.shape}; expected (33,)."
        )

    state = np.nan_to_num(
        state,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return state


# ============================================================
# Process one CSV
# ============================================================

def process_file(
    path: Path,
    encoder: NetworkStateEncoder,
    chunk_size: int,
    window_seconds: int,
    max_states: int,
    global_state_index: int,
) -> Tuple[
    List[np.ndarray],
    List[int],
    List[int],
    int,
]:
    """
    Process one CIC-IDS CSV file.

    Returns:
        states
        labels
        window IDs
        updated global state index
    """

    print()
    print("-" * 70)
    print(
        f"Processing: {path.name}"
    )
    print("-" * 70)

    states: List[np.ndarray] = []
    labels: List[int] = []
    window_ids: List[int] = []

    rows_processed = 0

    pending_window_id = None
    pending_parts: List[pd.DataFrame] = []

    # --------------------------------------------------------
    # We determine the file reference timestamp from the first
    # valid chunk. This avoids scanning the entire huge file
    # before processing.
    # --------------------------------------------------------

    reference_time = None

    # --------------------------------------------------------
    # Chunk loop
    # --------------------------------------------------------

    for chunk_number, chunk in enumerate(
        iter_csv(
            path,
            chunk_size=chunk_size,
        ),
        start=1,
    ):

        rows_processed += len(chunk)

        # ----------------------------------------------------
        # Normalize columns
        # ----------------------------------------------------

        chunk = standardize_columns(
            chunk
        )

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        chunk = clean_data(
            chunk,
            label_column="label",
        )

        if chunk.empty:
            continue

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        binary_labels = (
            create_binary_attack_label(
                chunk,
                label_column="label",
            )
        )

        attack_types = (
            create_attack_type(
                chunk,
                label_column="label",
            )
        )

        chunk["binary_label"] = (
            binary_labels.to_numpy(
                dtype=np.int8
            )
        )

        chunk["attack_type"] = (
            attack_types.to_numpy()
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        if "timestamp" not in chunk.columns:

            print(
                "WARNING: timestamp missing; "
                "skipping chunk."
            )

            continue

        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"],
            errors="coerce",
            dayfirst=True,
        )

        chunk = chunk.dropna(
            subset=["timestamp"]
        )

        if chunk.empty:
            continue

        # ----------------------------------------------------
        # Prepare canonical features
        # ----------------------------------------------------

        chunk = prepare_dataframe(
            chunk
        )

        # ----------------------------------------------------
        # Establish reference timestamp
        # ----------------------------------------------------

        if reference_time is None:

            reference_time = (
                chunk["timestamp"].min()
            )

        # ----------------------------------------------------
        # Window assignment
        # ----------------------------------------------------

        elapsed = (
            chunk["timestamp"]
            - reference_time
        ).dt.total_seconds()

        chunk["_window_id"] = (
            np.floor(
                elapsed
                / float(window_seconds)
            )
            .astype(np.int64)
        )

        # ----------------------------------------------------
        # Calculate temporal features ONCE per chunk.
        # ----------------------------------------------------

        chunk = add_fast_temporal_features(
            chunk
        )

        # ----------------------------------------------------
        # Group windows.
        #
        # Only the final window needs to be carried into the
        # next chunk.
        # ----------------------------------------------------

        grouped = chunk.groupby(
            "_window_id",
            sort=True,
        )

        groups = list(
            grouped
        )

        if not groups:
            continue

        for group_position, (
            current_window_id,
            window,
        ) in enumerate(groups):

            current_window_id = int(
                current_window_id
            )

            window = window.drop(
                columns=["_window_id"],
                errors="ignore",
            )

            # ------------------------------------------------
            # First window
            # ------------------------------------------------

            if pending_window_id is None:

                pending_window_id = (
                    current_window_id
                )

                pending_parts = [
                    window
                ]

                continue

            # ------------------------------------------------
            # Same window
            # ------------------------------------------------

            if (
                current_window_id
                == pending_window_id
            ):

                pending_parts.append(
                    window
                )

                continue

            # ------------------------------------------------
            # Finalize pending window.
            # ------------------------------------------------

            pending = pd.concat(
                pending_parts,
                ignore_index=True,
            )

            state = encode_window_fast(
                pending,
                encoder,
            )

            binary_values = pd.to_numeric(
                pending["binary_label"],
                errors="coerce",
            ).fillna(0)

            window_label = int(
                binary_values.max() > 0
            )

            states.append(
                state
            )

            labels.append(
                window_label
            )

            window_ids.append(
                global_state_index
            )

            global_state_index += 1

            # ------------------------------------------------
            # Check maximum state limit.
            # ------------------------------------------------

            if len(states) >= max_states:

                print(
                    f"Rows processed: "
                    f"{rows_processed:,} "
                    f"| States: "
                    f"{len(states):,}"
                )

                return (
                    states,
                    labels,
                    window_ids,
                    global_state_index,
                )

            # ------------------------------------------------
            # Start new pending window.
            # ------------------------------------------------

            pending_window_id = (
                current_window_id
            )

            pending_parts = [
                window
            ]

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"Chunk {chunk_number:>4} | "
            f"Rows: {rows_processed:>10,} | "
            f"States: {len(states):>6,}"
        )

    # ========================================================
    # Final pending window
    # ========================================================

    if pending_parts:

        pending = pd.concat(
            pending_parts,
            ignore_index=True,
        )

        state = encode_window_fast(
            pending,
            encoder,
        )

        binary_values = pd.to_numeric(
            pending["binary_label"],
            errors="coerce",
        ).fillna(0)

        window_label = int(
            binary_values.max() > 0
        )

        states.append(
            state
        )

        labels.append(
            window_label
        )

        window_ids.append(
            global_state_index
        )

        global_state_index += 1

    print(
        f"Finished {path.name}: "
        f"{rows_processed:,} rows → "
        f"{len(states):,} states"
    )

    return (
        states,
        labels,
        window_ids,
        global_state_index,
    )


# ============================================================
# Save
# ============================================================

def save_dataset(
    states: List[np.ndarray],
    labels: List[int],
    window_ids: List[int],
    processed_files: List[str],
    chunk_size: int,
    window_seconds: int,
    max_states: int,
) -> None:

    if not states:

        raise RuntimeError(
            "No states were generated."
        )

    X = np.asarray(
        states,
        dtype=np.float32,
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    ids = np.asarray(
        window_ids,
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if X.ndim != 2:

        raise RuntimeError(
            f"Expected 2D X; got {X.shape}"
        )

    if X.shape[1] != 33:

        raise RuntimeError(
            "CRITICAL STATE DIMENSION ERROR: "
            f"Expected 33, got {X.shape[1]}"
        )

    if len(X) != len(y):

        raise RuntimeError(
            "X/y length mismatch."
        )

    if len(X) != len(ids):

        raise RuntimeError(
            "X/window_ids length mismatch."
        )

    # --------------------------------------------------------
    # Numerical safety
    # --------------------------------------------------------

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # --------------------------------------------------------
    # Save directory
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        window_ids=ids,
        feature_names=np.asarray(
            STATE_FEATURE_NAMES,
            dtype=str,
        ),
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "source": "CSE-CIC-IDS2018",
        "processed_files": processed_files,
        "chunk_size": chunk_size,
        "window_seconds": window_seconds,
        "maximum_states": max_states,
        "states": int(X.shape[0]),
        "dimensions": int(X.shape[1]),
        "attack_states": int(
            np.sum(y == 1)
        ),
        "benign_states": int(
            np.sum(y == 0)
        ),
        "feature_names": STATE_FEATURE_NAMES,
        "state_encoder": (
            "world_model.state_encoder.NetworkStateEncoder"
        ),
        "representation": (
            "33-dimensional window-level "
            "network state"
        ),
    }

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING DATASET SAVED")
    print("=" * 70)
    print(
        f"States       : {X.shape[0]:,}"
    )
    print(
        f"Dimensions   : {X.shape[1]}"
    )
    print(
        f"Attack       : {int(np.sum(y == 1)):,}"
    )
    print(
        f"Benign       : {int(np.sum(y == 0)):,}"
    )
    print(
        f"Dataset      : {OUTPUT_NPZ}"
    )
    print(
        f"Metadata      : {OUTPUT_METADATA}"
    )
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build ARJUN's 33-dimensional "
            "Network State dataset."
        )
    )

    parser.add_argument(
        "--input-dir",
        default=str(DATASET_DIR),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--window-seconds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ARJUN OPTIMIZED LARGE-SCALE DATASET BUILDER")
    print("=" * 70)

    input_dir = Path(
        args.input_dir
    )

    print(
        f"Input directory : {input_dir}"
    )

    print(
        f"Chunk size      : {args.chunk_size:,}"
    )

    print(
        f"Window size     : {args.window_seconds} seconds"
    )

    print(
        f"Maximum states  : {args.max_states:,}"
    )

    # --------------------------------------------------------
    # Discover files
    # --------------------------------------------------------

    files = discover_csv_files(
        input_dir
    )

    if not files:

        raise FileNotFoundError(
            f"No CSV files found in {input_dir}"
        )

    files = sorted(
        files,
        key=lambda p: p.name.lower(),
    )

    if args.limit_files is not None:

        files = files[
            :args.limit_files
        ]

    print(
        f"CSV files selected: {len(files)}"
    )

    for file in files:

        print(
            f"  - {file.name}"
        )

    # --------------------------------------------------------
    # Encoder verification
    # --------------------------------------------------------

    encoder = NetworkStateEncoder()

    if (
        list(encoder.feature_names)
        != STATE_FEATURE_NAMES
    ):

        raise RuntimeError(
            "NetworkStateEncoder feature list "
            "does not match the expected 33 "
            "feature representation."
        )

    print()
    print(
        "State encoder verified: 33 dimensions"
    )

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    all_states: List[np.ndarray] = []
    all_labels: List[int] = []
    all_window_ids: List[int] = []

    global_state_index = 0

    for file_index, path in enumerate(
        files,
        start=1,
    ):

        if len(all_states) >= args.max_states:
            break

        remaining = (
            args.max_states
            - len(all_states)
        )

        print()
        print(
            f"[{file_index}/{len(files)}]"
        )

        (
            states,
            labels,
            window_ids,
            global_state_index,
        ) = process_file(
            path=path,
            encoder=encoder,
            chunk_size=args.chunk_size,
            window_seconds=args.window_seconds,
            max_states=remaining,
            global_state_index=global_state_index,
        )

        all_states.extend(
            states
        )

        all_labels.extend(
            labels
        )

        all_window_ids.extend(
            window_ids
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        states=all_states,
        labels=all_labels,
        window_ids=all_window_ids,
        processed_files=[
            file.name
            for file in files
        ],
        chunk_size=args.chunk_size,
        window_seconds=args.window_seconds,
        max_states=args.max_states,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()