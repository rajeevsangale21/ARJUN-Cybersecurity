import numpy as np
import pandas as pd


class NetworkStateEncoder:
    """
    Converts window-level network telemetry into fixed-length
    network state vectors.

    ARJUN state:
        S_t ∈ R^33

    Important:
    - Available telemetry is used directly.
    - Missing telemetry is represented by zero because the
      33-dimensional model interface must remain fixed.
    - Packet-only fields are NOT fabricated from unrelated fields.
    - Flow-level CIC-IDS-2018 data therefore legitimately has
      zeros for unavailable packet-only measurements.
    """

    def __init__(self):

        self.feature_names = [

            # Flow
            "flow_count",
            "unique_src_ips",
            "unique_dst_ips",
            "unique_dst_ports",

            # Traffic
            "total_bytes",
            "total_packets",
            "avg_duration",
            "avg_packet_length",

            # TTL
            "avg_ttl",
            "ttl_variance",

            # TCP window
            "avg_tcp_window",
            "tcp_window_variance",

            # TCP flags
            "syn_count",
            "ack_count",
            "fin_count",
            "rst_count",
            "psh_count",
            "urg_count",

            # Packet behaviour
            "fragment_count",
            "retransmission_count",

            # Payload
            "avg_payload_size",
            "payload_variance",
            "max_payload_size",

            # Timing
            "avg_iat",
            "iat_variance",
            "avg_src_iat",
            "src_iat_variance",

            # Rates
            "avg_bytes_per_packet",
            "avg_packets_per_second",
            "avg_bytes_per_second",

            # Behaviour
            "avg_unique_dst_ports",
            "avg_unique_dst_hosts",

            # Scanning
            "port_scan_count",
        ]

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _numeric_series(group, column):
        """
        Return a numeric series.

        Missing columns are represented by zeros.
        Existing columns are converted safely.
        """

        if column not in group.columns:

            return pd.Series(
                0.0,
                index=group.index,
                dtype=np.float64
            )

        return pd.to_numeric(
            group[column],
            errors="coerce"
        ).fillna(0.0)

    @staticmethod
    def _mean(group, column):

        values = NetworkStateEncoder._numeric_series(
            group,
            column
        )

        if len(values) == 0:
            return 0.0

        return float(values.mean())

    @staticmethod
    def _sum(group, column):

        values = NetworkStateEncoder._numeric_series(
            group,
            column
        )

        if len(values) == 0:
            return 0.0

        return float(values.sum())

    @staticmethod
    def _variance(group, column):

        values = NetworkStateEncoder._numeric_series(
            group,
            column
        )

        if len(values) <= 1:
            return 0.0

        variance = values.var(ddof=0)

        if pd.isna(variance):
            return 0.0

        return float(variance)

    @staticmethod
    def _unique_count(group, column):

        if column not in group.columns:
            return 0.0

        values = group[column].dropna()

        if len(values) == 0:
            return 0.0

        return float(
            values.astype(str).nunique()
        )

    # ============================================================
    # Window encoding
    # ============================================================

    def encode_window(self, group):

        if group.empty:

            return np.zeros(
                len(self.feature_names),
                dtype=np.float32
            )

        # --------------------------------------------------------
        # Flow
        # --------------------------------------------------------

        flow_count = float(len(group))

        unique_src_ips = self._unique_count(
            group,
            "src_ip"
        )

        unique_dst_ips = self._unique_count(
            group,
            "dst_ip"
        )

        unique_dst_ports = self._unique_count(
            group,
            "dst_port"
        )

        # --------------------------------------------------------
        # Traffic volume
        # --------------------------------------------------------

        total_bytes = self._sum(
            group,
            "bytes"
        )

        total_packets = self._sum(
            group,
            "packets"
        )

        # --------------------------------------------------------
        # Duration
        # --------------------------------------------------------

        avg_duration = self._mean(
            group,
            "duration"
        )

        # --------------------------------------------------------
        # Packet length
        # --------------------------------------------------------

        avg_packet_length = self._mean(
            group,
            "packet_length"
        )

        # --------------------------------------------------------
        # TTL
        # --------------------------------------------------------

        avg_ttl = self._mean(
            group,
            "ttl"
        )

        ttl_variance = self._variance(
            group,
            "ttl"
        )

        # --------------------------------------------------------
        # TCP window
        # --------------------------------------------------------

        avg_tcp_window = self._mean(
            group,
            "tcp_window"
        )

        tcp_window_variance = self._variance(
            group,
            "tcp_window"
        )

        # --------------------------------------------------------
        # TCP flags
        # --------------------------------------------------------

        syn_count = self._sum(
            group,
            "flag_syn"
        )

        ack_count = self._sum(
            group,
            "flag_ack"
        )

        fin_count = self._sum(
            group,
            "flag_fin"
        )

        rst_count = self._sum(
            group,
            "flag_rst"
        )

        psh_count = self._sum(
            group,
            "flag_psh"
        )

        urg_count = self._sum(
            group,
            "flag_urg"
        )

        # --------------------------------------------------------
        # Fragmentation
        # --------------------------------------------------------

        fragment_count = self._sum(
            group,
            "fragment_flag"
        )

        # --------------------------------------------------------
        # Retransmissions
        # --------------------------------------------------------

        retransmission_count = self._sum(
            group,
            "retransmission"
        )

        # --------------------------------------------------------
        # Payload
        # --------------------------------------------------------

        avg_payload_size = self._mean(
            group,
            "payload_size"
        )

        payload_variance = self._variance(
            group,
            "payload_size"
        )

        if "payload_size" in group.columns:

            payload_values = pd.to_numeric(
                group["payload_size"],
                errors="coerce"
            ).fillna(0.0)

            if len(payload_values) > 0:
                max_payload_size = float(
                    payload_values.max()
                )
            else:
                max_payload_size = 0.0

        else:

            max_payload_size = 0.0

        # --------------------------------------------------------
        # Inter-arrival time
        # --------------------------------------------------------

        avg_iat = self._mean(
            group,
            "iat_seconds"
        )

        iat_variance = self._variance(
            group,
            "iat_seconds"
        )

        # --------------------------------------------------------
        # Source IAT
        # --------------------------------------------------------

        avg_src_iat = self._mean(
            group,
            "src_iat_seconds"
        )

        src_iat_variance = self._variance(
            group,
            "src_iat_seconds"
        )

        # --------------------------------------------------------
        # Rates
        # --------------------------------------------------------

        bytes_per_packet = self._mean(
            group,
            "bytes_per_packet"
        )

        packets_per_second = self._mean(
            group,
            "packets_per_second"
        )

        bytes_per_second = self._mean(
            group,
            "bytes_per_second"
        )

        # --------------------------------------------------------
        # Behaviour
        # --------------------------------------------------------

        avg_unique_dst_ports = self._mean(
            group,
            "unique_dst_ports_per_window"
        )

        avg_unique_dst_hosts = self._mean(
            group,
            "unique_dst_hosts_per_window"
        )

        # --------------------------------------------------------
        # Port scanning
        # --------------------------------------------------------

        port_scan_count = self._sum(
            group,
            "port_scan_signature"
        )

        # --------------------------------------------------------
        # Construct state
        # --------------------------------------------------------

        state = np.array(
            [

                flow_count,
                unique_src_ips,
                unique_dst_ips,
                unique_dst_ports,

                total_bytes,
                total_packets,
                avg_duration,
                avg_packet_length,

                avg_ttl,
                ttl_variance,

                avg_tcp_window,
                tcp_window_variance,

                syn_count,
                ack_count,
                fin_count,
                rst_count,
                psh_count,
                urg_count,

                fragment_count,
                retransmission_count,

                avg_payload_size,
                payload_variance,
                max_payload_size,

                avg_iat,
                iat_variance,

                avg_src_iat,
                src_iat_variance,

                bytes_per_packet,
                packets_per_second,
                bytes_per_second,

                avg_unique_dst_ports,
                avg_unique_dst_hosts,

                port_scan_count

            ],
            dtype=np.float32
        )

        state = np.nan_to_num(
            state,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        if len(state) != len(
            self.feature_names
        ):

            raise RuntimeError(
                "State dimension mismatch: "
                f"{len(state)} != "
                f"{len(self.feature_names)}"
            )

        return state

    # ============================================================
    # Dataset encoding
    # ============================================================

    def encode_dataset(self, df):

        if "window_id" not in df.columns:

            raise ValueError(
                "window_id column is required."
            )

        states = []
        window_ids = []

        for window_id, group in df.groupby(
            "window_id",
            sort=True
        ):

            state = self.encode_window(
                group
            )

            states.append(
                state
            )

            window_ids.append(
                int(window_id)
            )

        if not states:

            raise ValueError(
                "No network states could be created."
            )

        states = np.asarray(
            states,
            dtype=np.float32
        )

        window_ids = np.asarray(
            window_ids,
            dtype=np.int64
        )

        expected_dimension = len(
            self.feature_names
        )

        if states.ndim != 2:

            raise RuntimeError(
                "Network states must be 2-dimensional."
            )

        if states.shape[1] != expected_dimension:

            raise RuntimeError(
                "Network state dimension mismatch: "
                f"{states.shape[1]} != "
                f"{expected_dimension}"
            )

        states = np.nan_to_num(
            states,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return states, window_ids

    # ============================================================
    # Feature lookup
    # ============================================================

    def get_feature_index(self, feature_name):

        if feature_name not in self.feature_names:

            raise ValueError(
                f"Unknown feature: {feature_name}"
            )

        return self.feature_names.index(
            feature_name
        )

    # ============================================================
    # State description
    # ============================================================

    def describe_state(self, state):

        state = np.asarray(
            state
        )

        if len(state) != len(
            self.feature_names
        ):

            raise ValueError(
                "State dimension does not match "
                "feature definition."
            )

        return {
            feature: float(value)
            for feature, value in zip(
                self.feature_names,
                state
            )
        }


if __name__ == "__main__":

    print("=" * 70)
    print("ARJUN NETWORK STATE ENCODER TEST")
    print("=" * 70)

    encoder = NetworkStateEncoder()

    print(
        f"Feature count : "
        f"{len(encoder.feature_names)}"
    )

    print()

    sample = pd.DataFrame(
        {
            "src_ip": [
                "10.0.0.1",
                "10.0.0.2"
            ],
            "dst_ip": [
                "10.0.0.5",
                "10.0.0.5"
            ],
            "dst_port": [
                80,
                443
            ],
            "bytes": [
                1000,
                2000
            ],
            "packets": [
                10,
                20
            ],
            "duration": [
                1.0,
                2.0
            ],
            "packet_length": [
                100,
                100
            ],
            "ttl": [
                64,
                63
            ],
            "tcp_window": [
                64240,
                29200
            ],
            "flag_syn": [
                1,
                0
            ],
            "flag_ack": [
                0,
                5
            ],
            "flag_fin": [
                0,
                0
            ],
            "flag_rst": [
                0,
                0
            ],
            "flag_psh": [
                0,
                2
            ],
            "flag_urg": [
                0,
                0
            ],
            "fragment_flag": [
                0,
                0
            ],
            "retransmission": [
                0,
                0
            ],
            "payload_size": [
                80,
                80
            ],
            "iat_seconds": [
                0.1,
                0.2
            ],
            "src_iat_seconds": [
                0.1,
                0.2
            ],
            "bytes_per_packet": [
                100,
                100
            ],
            "packets_per_second": [
                10,
                10
            ],
            "bytes_per_second": [
                1000,
                1000
            ],
            "unique_dst_ports_per_window": [
                2,
                2
            ],
            "unique_dst_hosts_per_window": [
                1,
                1
            ],
            "port_scan_signature": [
                0,
                0
            ]
        }
    )

    sample["window_id"] = 0

    states, window_ids = encoder.encode_dataset(
        sample
    )

    print(
        f"State shape  : {states.shape}"
    )

    print(
        f"Window IDs   : {window_ids.tolist()}"
    )

    assert states.shape == (1, 33)
    assert window_ids.tolist() == [0]

    assert np.isfinite(states).all()

    print()
    print("STATE ENCODER TEST: PASSED")