"""
ARJUN - Network Graph Builder

Builds a graph representation for each temporal network window.

Supports:
    - datasets with source/destination IPs
    - datasets without IP identifiers
    - CIC-IDS flow-level CSV files
    - packet/Zeek-style data

Important:
    When IP addresses are unavailable, the builder DOES NOT create
    one synthetic source node per flow. That would create extremely
    large graphs for CIC-IDS data.

    Instead, missing source identifiers are represented by a shared
    "unknown_src" node.

Graph format:

{
    "window_id": int,
    "nodes": list[str],
    "node_features": np.ndarray,
    "adjacency": np.ndarray
}

Node features:

    0 = outgoing connections
    1 = incoming connections
    2 = bytes
    3 = packets
    4 = unique destination ports
    5 = SYN count
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

GRAPH_FEATURE_DIMENSION = 6

# Safety limit.
#
# A normal graph should be much smaller than this.
# This prevents pathological windows from creating enormous
# dense adjacency matrices.
MAX_GRAPH_NODES = 2048


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(
    value,
    default: float = 0.0,
) -> float:
    """Convert a value safely to float."""

    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """Convert a Series to numeric safely."""

    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)


def _first_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
):
    """Return the first existing column."""

    for column in candidates:

        if column in df.columns:
            return column

    return None


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def _ensure_graph_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure the dataframe contains all graph columns.

    Missing source IPs are represented using ONE shared
    "unknown_src" node.

    Missing destination IPs are represented using destination
    port identifiers when available.
    """

    result = df.copy()

    # --------------------------------------------------------
    # Source IP
    # --------------------------------------------------------

    src_column = _first_existing_column(
        result,
        [
            "src_ip",
            "source_ip",
            "source",
        ],
    )

    if src_column is None:

        # IMPORTANT:
        # Do NOT create one node per flow.
        #
        # The old implementation used:
        #
        #     flow_src_<row number>
        #
        # which could create tens of thousands of nodes in
        # a single five-second window.
        result["src_ip"] = "unknown_src"

    elif src_column != "src_ip":

        result["src_ip"] = (
            result[src_column]
            .astype("string")
            .fillna("unknown_src")
        )

    else:

        result["src_ip"] = (
            result["src_ip"]
            .astype("string")
            .fillna("unknown_src")
        )

    # Replace textual missing values.

    result["src_ip"] = result["src_ip"].replace(
        {
            "nan": "unknown_src",
            "None": "unknown_src",
            "<NA>": "unknown_src",
            "": "unknown_src",
        }
    )

    # --------------------------------------------------------
    # Destination IP
    # --------------------------------------------------------

    dst_column = _first_existing_column(
        result,
        [
            "dst_ip",
            "destination_ip",
            "destination",
        ],
    )

    if dst_column is None:

        # If destination IP is unavailable, use destination
        # port as a stable communication endpoint.

        if "dst_port" in result.columns:

            result["dst_ip"] = (
                "dst_port_"
                + result["dst_port"]
                .astype(str)
            )

        elif "destination_port" in result.columns:

            result["dst_ip"] = (
                "dst_port_"
                + result["destination_port"]
                .astype(str)
            )

        else:

            result["dst_ip"] = "unknown_dst"

    elif dst_column != "dst_ip":

        result["dst_ip"] = (
            result[dst_column]
            .astype("string")
            .fillna("unknown_dst")
        )

    else:

        result["dst_ip"] = (
            result["dst_ip"]
            .astype("string")
            .fillna("unknown_dst")
        )

    result["dst_ip"] = result["dst_ip"].replace(
        {
            "nan": "unknown_dst",
            "None": "unknown_dst",
            "<NA>": "unknown_dst",
            "": "unknown_dst",
        }
    )

    # --------------------------------------------------------
    # Destination port
    # --------------------------------------------------------

    dst_port_column = _first_existing_column(
        result,
        [
            "dst_port",
            "destination_port",
            "destination_port_number",
        ],
    )

    if dst_port_column is None:

        result["dst_port"] = 0.0

    elif dst_port_column != "dst_port":

        result["dst_port"] = _safe_numeric(
            result[dst_port_column]
        )

    else:

        result["dst_port"] = _safe_numeric(
            result["dst_port"]
        )

    # --------------------------------------------------------
    # Bytes
    # --------------------------------------------------------

    bytes_column = _first_existing_column(
        result,
        [
            "bytes",
            "total_bytes",
            "flow_bytes",
        ],
    )

    if bytes_column is None:

        result["bytes"] = 0.0

    elif bytes_column != "bytes":

        result["bytes"] = _safe_numeric(
            result[bytes_column]
        )

    else:

        result["bytes"] = _safe_numeric(
            result["bytes"]
        )

    # --------------------------------------------------------
    # Packets
    # --------------------------------------------------------

    packets_column = _first_existing_column(
        result,
        [
            "packets",
            "total_packets",
            "flow_packets",
        ],
    )

    if packets_column is None:

        result["packets"] = 0.0

    elif packets_column != "packets":

        result["packets"] = _safe_numeric(
            result[packets_column]
        )

    else:

        result["packets"] = _safe_numeric(
            result["packets"]
        )

    # --------------------------------------------------------
    # SYN
    # --------------------------------------------------------

    syn_column = _first_existing_column(
        result,
        [
            "flag_syn",
            "syn",
            "syn_count",
        ],
    )

    if syn_column is None:

        result["syn_count"] = 0.0

    elif syn_column != "syn_count":

        result["syn_count"] = _safe_numeric(
            result[syn_column]
        )

    else:

        result["syn_count"] = _safe_numeric(
            result["syn_count"]
        )

    return result


# ============================================================
# NODE SELECTION
# ============================================================

def _select_graph_nodes(
    group: pd.DataFrame,
) -> List[str]:
    """
    Select graph nodes.

    Normally all nodes are retained.

    If a pathological window contains more than
    MAX_GRAPH_NODES nodes, retain the most active nodes based
    on number of flows.

    This guarantees that dense adjacency remains manageable.
    """

    sources = (
        group["src_ip"]
        .astype("string")
        .fillna("unknown_src")
    )

    destinations = (
        group["dst_ip"]
        .astype("string")
        .fillna("unknown_dst")
    )

    all_nodes = (
        list(sources)
        + list(destinations)
    )

    unique_nodes = sorted(
        set(all_nodes)
    )

    if len(unique_nodes) <= MAX_GRAPH_NODES:
        return unique_nodes

    # --------------------------------------------------------
    # Node activity
    # --------------------------------------------------------

    source_counts = (
        sources.value_counts()
    )

    destination_counts = (
        destinations.value_counts()
    )

    activity = {}

    for node in unique_nodes:

        activity[node] = (
            int(source_counts.get(node, 0))
            + int(destination_counts.get(node, 0))
        )

    selected = sorted(
        unique_nodes,
        key=lambda node: (
            -activity[node],
            node,
        ),
    )

    selected = selected[
        :MAX_GRAPH_NODES
    ]

    return sorted(selected)


# ============================================================
# MAIN GRAPH BUILDER
# ============================================================

def build_window_graphs(
    df: pd.DataFrame,
) -> Dict[int, Dict]:
    """
    Build a graph for every temporal network window.

    Required column:
        window_id

    Returns:

        {
            window_id: {
                "window_id": window_id,
                "nodes": [...],
                "node_features": ndarray,
                "adjacency": ndarray
            }
        }
    """

    if df is None:

        raise ValueError(
            "Input dataframe cannot be None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:

        return {}

    if "window_id" not in df.columns:

        raise ValueError(
            "window_id column is required."
        )

    df = _ensure_graph_columns(
        df
    )

    graphs = {}

    # ========================================================
    # PROCESS EACH WINDOW
    # ========================================================

    for window_id, group in df.groupby(
        "window_id",
        sort=True,
    ):

        window_id = int(
            window_id
        )

        if group.empty:
            continue

        # ----------------------------------------------------
        # Normalize identifiers
        # ----------------------------------------------------

        sources = (
            group["src_ip"]
            .astype("string")
            .fillna("unknown_src")
        )

        destinations = (
            group["dst_ip"]
            .astype("string")
            .fillna("unknown_dst")
        )

        # ----------------------------------------------------
        # Select nodes
        # ----------------------------------------------------

        nodes = _select_graph_nodes(
            group
        )

        if not nodes:
            continue

        node_index = {
            node: index
            for index, node in enumerate(nodes)
        }

        node_count = len(nodes)

        # ----------------------------------------------------
        # Node feature matrix
        # ----------------------------------------------------

        node_features = np.zeros(
            (
                node_count,
                GRAPH_FEATURE_DIMENSION,
            ),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Dense adjacency
        #
        # Because node count is capped, this is now bounded.
        # ----------------------------------------------------

        adjacency = np.zeros(
            (
                node_count,
                node_count,
            ),
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Flow columns
        # ----------------------------------------------------

        bytes_values = (
            _safe_numeric(
                group["bytes"]
            )
            .to_numpy()
        )

        packet_values = (
            _safe_numeric(
                group["packets"]
            )
            .to_numpy()
        )

        syn_values = (
            _safe_numeric(
                group["syn_count"]
            )
            .to_numpy()
        )

        ports = (
            group["dst_port"]
            .astype(str)
            .to_numpy()
        )

        source_array = (
            sources.to_numpy()
        )

        destination_array = (
            destinations.to_numpy()
        )

        # ----------------------------------------------------
        # Source destination-port sets
        # ----------------------------------------------------

        source_port_sets = {
            node: set()
            for node in nodes
        }

        # ====================================================
        # AGGREGATE FLOWS
        # ====================================================

        for i in range(
            len(group)
        ):

            src = source_array[i]
            dst = destination_array[i]

            if src not in node_index:
                continue

            if dst not in node_index:
                continue

            src_idx = node_index[src]
            dst_idx = node_index[dst]

            flow_bytes = _safe_float(
                bytes_values[i]
            )

            flow_packets = _safe_float(
                packet_values[i]
            )

            flow_syn = _safe_float(
                syn_values[i]
            )

            port = ports[i]

            # ------------------------------------------------
            # Source node
            # ------------------------------------------------

            node_features[
                src_idx,
                0,
            ] += 1.0

            node_features[
                src_idx,
                2,
            ] += flow_bytes

            node_features[
                src_idx,
                3,
            ] += flow_packets

            node_features[
                src_idx,
                5,
            ] += flow_syn

            source_port_sets[
                src
            ].add(port)

            # ------------------------------------------------
            # Destination node
            # ------------------------------------------------

            node_features[
                dst_idx,
                1,
            ] += 1.0

            node_features[
                dst_idx,
                2,
            ] += flow_bytes

            node_features[
                dst_idx,
                3,
            ] += flow_packets

            # ------------------------------------------------
            # Directed edge
            # ------------------------------------------------

            adjacency[
                src_idx,
                dst_idx,
            ] += 1.0

        # ====================================================
        # UNIQUE DESTINATION PORTS
        # ====================================================

        for node, ports_set in (
            source_port_sets.items()
        ):

            idx = node_index[node]

            node_features[
                idx,
                4,
            ] = float(
                len(ports_set)
            )

        # ====================================================
        # NUMERICAL SAFETY
        # ====================================================

        node_features = np.nan_to_num(
            node_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        adjacency = np.nan_to_num(
            adjacency,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        graphs[window_id] = {
            "window_id": window_id,
            "nodes": nodes,
            "node_features": node_features,
            "adjacency": adjacency,
        }

    return graphs


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("ARJUN GRAPH BUILDER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Test 1:
    # Dataset with no IP columns.
    #
    # This specifically tests the CIC-IDS situation.
    # --------------------------------------------------------

    sample = pd.DataFrame(
        {
            "window_id": [
                0,
                0,
                0,
                1,
                1,
            ],
            "dst_port": [
                80,
                443,
                22,
                80,
                8080,
            ],
            "bytes": [
                1000,
                2000,
                500,
                3000,
                4000,
            ],
            "packets": [
                10,
                20,
                5,
                30,
                40,
            ],
            "syn_count": [
                1,
                1,
                1,
                2,
                3,
            ],
        }
    )

    graphs = build_window_graphs(
        sample
    )

    print(
        f"Graphs created : {len(graphs)}"
    )

    for window_id, graph in graphs.items():

        print()
        print(
            f"Window {window_id}"
        )

        print(
            f"Nodes: "
            f"{len(graph['nodes'])}"
        )

        print(
            f"Node features: "
            f"{graph['node_features'].shape}"
        )

        print(
            f"Adjacency: "
            f"{graph['adjacency'].shape}"
        )

        assert (
            graph["node_features"].shape[0]
            == len(graph["nodes"])
        )

        assert (
            graph["adjacency"].shape
            == (
                len(graph["nodes"]),
                len(graph["nodes"]),
            )
        )

        assert np.all(
            np.isfinite(
                graph["node_features"]
            )
        )

        assert np.all(
            np.isfinite(
                graph["adjacency"]
            )
        )

    # --------------------------------------------------------
    # Test 2:
    # Large pathological window.
    #
    # This verifies that the node safety cap works.
    # --------------------------------------------------------

    large_rows = 5000

    large_sample = pd.DataFrame(
        {
            "window_id": [99] * large_rows,
            "dst_port": (
                np.arange(
                    large_rows
                )
                % 500
            ),
            "bytes": [1000] * large_rows,
            "packets": [10] * large_rows,
            "syn_count": [1] * large_rows,
        }
    )

    large_graphs = build_window_graphs(
        large_sample
    )

    large_graph = large_graphs[99]

    print()
    print(
        "Large-window safety test:"
    )

    print(
        f"Nodes retained: "
        f"{len(large_graph['nodes'])}"
    )

    print(
        f"Adjacency shape: "
        f"{large_graph['adjacency'].shape}"
    )

    assert (
        len(large_graph["nodes"])
        <= MAX_GRAPH_NODES
    )

    assert np.all(
        np.isfinite(
            large_graph["adjacency"]
        )
    )

    print()
    print("=" * 70)
    print("GRAPH BUILDER TEST PASSED")
    print("=" * 70)