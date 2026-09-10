import numpy as np
import pandas as pd


def _numeric(df, column, default=0.0):
    """
    Safely convert a column to numeric.
    """

    if column not in df.columns:

        return pd.Series(
            default,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


def _tcp_flag_features(df):
    """
    Extract TCP flag indicators from tcp_flags.

    Scapy commonly represents TCP flags as strings such as:
        S
        SA
        PA
        FA
        R
    """

    if "tcp_flags" not in df.columns:

        for column in [
            "flag_syn",
            "flag_ack",
            "flag_fin",
            "flag_rst",
            "flag_psh",
            "flag_urg",
        ]:

            df[column] = 0

        return df

    flags = (
        df["tcp_flags"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    df["flag_syn"] = (
        flags.str.contains(
            "S",
            regex=False,
        )
        .astype(int)
    )

    df["flag_ack"] = (
        flags.str.contains(
            "A",
            regex=False,
        )
        .astype(int)
    )

    df["flag_fin"] = (
        flags.str.contains(
            "F",
            regex=False,
        )
        .astype(int)
    )

    df["flag_rst"] = (
        flags.str.contains(
            "R",
            regex=False,
        )
        .astype(int)
    )

    df["flag_psh"] = (
        flags.str.contains(
            "P",
            regex=False,
        )
        .astype(int)
    )

    df["flag_urg"] = (
        flags.str.contains(
            "U",
            regex=False,
        )
        .astype(int)
    )

    return df


def _fragment_features(df):
    """
    Create a binary fragment indicator.

    IPv4:
        fragment_offset > 0
        OR more_fragments == True

    If fragment_flag was already calculated by the loader,
    preserve it.
    """

    if "fragment_flag" in df.columns:

        df["fragment_flag"] = (
            pd.to_numeric(
                df["fragment_flag"],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .astype(int)
        )

        return df

    if "fragment_offset" in df.columns:

        offset = _numeric(
            df,
            "fragment_offset",
        )

        fragment = offset.gt(0)

        if "more_fragments" in df.columns:

            more_fragments = (
                df["more_fragments"]
                .fillna(False)
                .astype(bool)
            )

            fragment = (
                fragment
                |
                more_fragments
            )

        df["fragment_flag"] = (
            fragment.astype(int)
        )

    else:

        df["fragment_flag"] = 0

    return df


def _retransmission_features(df):
    """
    Estimate TCP retransmissions.

    Preferred method:
        repeated TCP sequence numbers within the same
        directional connection.

    Required columns:
        src_ip
        dst_ip
        src_port
        dst_port
        tcp_seq

    If TCP sequence information is unavailable, the
    feature remains zero rather than incorrectly labeling
    every repeated flow as a retransmission.
    """

    df["retransmission"] = 0

    required = [
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "tcp_seq",
    ]

    if not all(
        column in df.columns
        for column in required
    ):

        return df

    tcp_seq = pd.to_numeric(
        df["tcp_seq"],
        errors="coerce",
    )

    valid = tcp_seq.notna()

    if not valid.any():

        return df

    connection = [
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "tcp_seq",
    ]

    duplicate_mask = (
        df.loc[valid]
        .duplicated(
            subset=connection,
            keep="first",
        )
    )

    df.loc[
        duplicate_mask.index[duplicate_mask],
        "retransmission"
    ] = 1

    return df


def add_packet_derived_features(df):
    """
    Add packet-level derived features required by ARJUN.

    For PCAP input this preserves genuine packet telemetry:

        - TTL
        - TCP window
        - TCP flags
        - payload size
        - fragmentation
        - retransmission estimate
        - port-scan signatures

    For flow CSV input, unavailable packet-level fields are
    deliberately left unavailable rather than fabricated.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "add_packet_derived_features expects "
            "a pandas DataFrame."
        )

    df = df.copy()

    # ==================================================
    # Numeric fields
    # ==================================================

    for column in [
        "ttl",
        "tcp_window",
        "packet_length",
        "payload_size",
        "fragment_offset",
        "tcp_seq",
        "src_port",
        "dst_port",
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # ==================================================
    # Packet length
    # ==================================================

    if "packet_length" in df.columns:

        df["packet_length"] = (
            df["packet_length"]
            .fillna(0)
            .clip(lower=0)
        )

    # ==================================================
    # Payload
    # ==================================================

    if "payload_size" not in df.columns:

        # Do NOT equate packet length with payload size.
        #
        # If this is flow data, payload size cannot be
        # reconstructed exactly from packet length alone.
        df["payload_size"] = 0.0

    else:

        df["payload_size"] = (
            df["payload_size"]
            .fillna(0)
            .clip(lower=0)
        )

    # ==================================================
    # TCP flags
    # ==================================================

    df = _tcp_flag_features(
        df
    )

    # ==================================================
    # Fragmentation
    # ==================================================

    df = _fragment_features(
        df
    )

    # ==================================================
    # TTL deviation
    # ==================================================

    if "ttl" in df.columns:

        ttl = pd.to_numeric(
            df["ttl"],
            errors="coerce",
        )

        median_ttl = ttl.median()

        if pd.isna(median_ttl):

            median_ttl = 0.0

        df["ttl_deviation"] = (
            ttl - median_ttl
        ).abs().fillna(0)

    else:

        df["ttl_deviation"] = 0.0

    # ==================================================
    # Retransmissions
    # ==================================================

    df = _retransmission_features(
        df
    )

    # ==================================================
    # Port scan signature
    # ==================================================

    if (
        "window_id" in df.columns
        and "src_ip" in df.columns
        and "dst_port" in df.columns
    ):

        port_count = (
            df.groupby(
                [
                    "window_id",
                    "src_ip",
                ]
            )["dst_port"]
            .transform("nunique")
        )

        if "dst_ip" in df.columns:

            host_count = (
                df.groupby(
                    [
                        "window_id",
                        "src_ip",
                    ]
                )["dst_ip"]
                .transform("nunique")
            )

        else:

            host_count = pd.Series(
                0,
                index=df.index,
            )

        df["port_scan_signature"] = (
            (
                (port_count >= 20)
                |
                (host_count >= 10)
            )
            .astype(int)
        )

    else:

        df["port_scan_signature"] = 0

    # ==================================================
    # Final numeric safety
    # ==================================================

    for column in [
        "ttl",
        "tcp_window",
        "packet_length",
        "payload_size",
        "ttl_deviation",
        "retransmission",
    ]:

        if column in df.columns:

            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .fillna(0)
            )

    return df


if __name__ == "__main__":

    sample = pd.DataFrame(
        {
            "src_ip": [
                "10.0.0.1",
                "10.0.0.1",
            ],
            "dst_ip": [
                "10.0.0.2",
                "10.0.0.2",
            ],
            "src_port": [1234, 1234],
            "dst_port": [80, 80],
            "ttl": [64, 64],
            "tcp_window": [64240, 64240],
            "tcp_flags": ["S", "A"],
            "packet_length": [60, 100],
            "payload_size": [0, 40],
            "fragment_offset": [0, 0],
            "tcp_seq": [1000, 1001],
        }
    )

    output = add_packet_derived_features(
        sample
    )

    required = [
        "flag_syn",
        "flag_ack",
        "fragment_flag",
        "retransmission",
        "port_scan_signature",
    ]

    missing = [
        column
        for column in required
        if column not in output.columns
    ]

    if missing:
        raise AssertionError(
            f"Missing derived features: {missing}"
        )

    print("PCAP PROCESSOR TEST: PASSED")
    print(
        output[
            [
                "ttl",
                "tcp_window",
                "payload_size",
                "flag_syn",
                "flag_ack",
                "fragment_flag",
                "retransmission",
            ]
        ]
    )