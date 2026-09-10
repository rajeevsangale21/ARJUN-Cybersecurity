"""Convert Zeek conn.log telemetry into ARJUN network states and graphs."""
from __future__ import annotations

import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.zeek_loader import load_zeek_conn
from features.graph_builder import build_window_graphs
from world_model.state_encoder import NetworkStateEncoder

WINDOW_SECONDS = 5


def _flag_count(history: pd.Series, flag: str) -> pd.Series:
    return history.fillna("").astype(str).str.count(flag).astype(float)


def prepare_zeek_features(conn: pd.DataFrame, window_seconds: int = WINDOW_SECONDS) -> pd.DataFrame:
    """Map normalized Zeek conn fields to the 33-feature state interface."""
    if conn.empty:
        raise ValueError("Zeek conn.log contains no usable rows.")

    df = conn.copy()
    df["timestamp"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["timestamp", "src_ip", "dst_ip"]).copy()
    if df.empty:
        raise ValueError("No valid Zeek rows remain after timestamp/IP validation.")

    numeric_cols = [
        "src_port", "dst_port", "duration", "orig_bytes", "resp_bytes",
        "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "missed_bytes",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Aggregate directional Zeek counters into ARJUN flow counters.
    df["bytes"] = df["orig_bytes"] + df["resp_bytes"]
    df["packets"] = df["orig_pkts"] + df["resp_pkts"]
    df["packet_length"] = np.divide(
        df["bytes"], df["packets"].replace(0, np.nan)
    )
    df["packet_length"] = df["packet_length"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Zeek conn.log does not provide packet TTL/window directly.
    # Keep these unavailable fields as zero rather than inventing values.
    df["ttl"] = 0.0
    df["tcp_window"] = 0.0
    df["fragment_flag"] = 0.0
    df["retransmission"] = 0.0

    history = df["history"].fillna("").astype(str)
    df["flag_syn"] = _flag_count(history, "S")
    df["flag_ack"] = _flag_count(history, "A")
    df["flag_fin"] = _flag_count(history, "F")
    df["flag_rst"] = _flag_count(history, "R")
    df["flag_psh"] = _flag_count(history, "P")
    df["flag_urg"] = _flag_count(history, "U")

    # Payload size cannot be recovered exactly from conn.log. Use the
    # directional byte counters as a clearly documented flow-level proxy.
    # Packet-only fields remain zero; no unrelated value is fabricated.
    df["payload_size"] = df["bytes"]

    # Fixed 5-second temporal windows, aligned to the first event.
    start = df["timestamp"].min().floor(f"{window_seconds}s")
    elapsed = (df["timestamp"] - start).dt.total_seconds().clip(lower=0)
    df["window_id"] = np.floor(elapsed / window_seconds).astype(int)

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df["iat_seconds"] = df["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0)
    df["src_iat_seconds"] = (
        df.groupby("src_ip")["timestamp"].diff().dt.total_seconds().fillna(0).clip(lower=0)
    )

    duration = df["duration"].replace(0, np.nan)
    df["bytes_per_packet"] = np.divide(
        df["bytes"], df["packets"].replace(0, np.nan)
    )
    df["bytes_per_packet"] = df["bytes_per_packet"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["packets_per_second"] = (df["packets"] / duration).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["bytes_per_second"] = (df["bytes"] / duration).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["unique_dst_ports_per_window"] = df.groupby("window_id")["dst_port"].transform("nunique")
    df["unique_dst_hosts_per_window"] = df.groupby("window_id")["dst_ip"].transform("nunique")

    port_count = df.groupby(["window_id", "src_ip"])["dst_port"].transform("nunique")
    host_count = df.groupby(["window_id", "src_ip"])["dst_ip"].transform("nunique")
    df["port_scan_signature"] = ((port_count >= 20) | (host_count >= 10)).astype(int)

    return df


def build_zeek_states(input_path: str | Path, output_dir: str | Path | None = None):
    """Load Zeek data and return aligned state vectors and communication graphs."""
    conn = load_zeek_conn(input_path)
    features = prepare_zeek_features(conn)

    encoder = NetworkStateEncoder()
    states, window_ids = encoder.encode_dataset(features)
    graphs = build_window_graphs(features)

    ordered_graphs = []
    for window_id in window_ids:
        graph = graphs.get(int(window_id))
        if graph is None:
            raise RuntimeError(f"Missing graph for Zeek window {window_id}.")
        ordered_graphs.append(graph)

    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 33:
        raise RuntimeError(f"Expected state shape (N,33), got {states.shape}.")
    if not np.isfinite(states).all():
        raise RuntimeError("Zeek-derived states contain NaN/Inf.")

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "zeek_network_states.npz",
            states=states,
            window_ids=window_ids,
            feature_names=np.asarray(encoder.feature_names, dtype=object),
            window_starts=np.asarray(
                [features.loc[features.window_id == w, "timestamp"].min().isoformat() for w in window_ids],
                dtype=object,
            ),
        )
        with (out / "zeek_network_graphs.pkl").open("wb") as fh:
            pickle.dump(ordered_graphs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return features, states, window_ids, ordered_graphs


def main():
    default_input = ROOT / "data/raw/logs/synthetic_conn.log"
    default_output = ROOT / "data/processed"

    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_input
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else default_output

    print("=" * 72)
    print("ARJUN ZEEK -> NETWORK STATE TEST")
    print("=" * 72)
    print(f"Input        : {input_path}")
    print(f"Output       : {output_dir}")

    features, states, window_ids, graphs = build_zeek_states(input_path, output_dir)

    print(f"Zeek rows    : {len(features)}")
    print(f"Windows      : {len(states)}")
    print(f"State shape  : {states.shape}")
    print(f"Graph count  : {len(graphs)}")
    print(f"Graph nodes  : {[len(g['nodes']) for g in graphs]}")
    print(f"Finite states: {np.isfinite(states).all()}")
    print()
    print("Latest state feature values:")
    encoder = NetworkStateEncoder()
    latest = encoder.describe_state(states[-1])
    for name in ["flow_count", "unique_dst_ports", "total_bytes", "total_packets", "syn_count", "port_scan_count"]:
        print(f"  {name:24s}: {latest[name]:.4f}")
    print()
    print("Artifacts saved:")
    print(f"  {output_dir / 'zeek_network_states.npz'}")
    print(f"  {output_dir / 'zeek_network_graphs.pkl'}")
    print("ZEEK -> NETWORK STATE: PASSED")


if __name__ == "__main__":
    main()
