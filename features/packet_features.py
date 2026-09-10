import numpy as np
import pandas as pd


def add_packet_features(df):

    df = df.copy()

    # ==================================================
    # Numeric fields
    # ==================================================

    for column in [
        "ttl",
        "tcp_window",
        "payload_size",
        "packet_length"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


    # ==================================================
    # TTL variance per temporal window
    # ==================================================

    if (
        "window_id" in df.columns
        and "ttl" in df.columns
    ):

        df["ttl_window_variance"] = (
            df.groupby("window_id")["ttl"]
            .transform(
                lambda x:
                x.var(ddof=0)
            )
            .fillna(0)
        )

    else:

        df["ttl_window_variance"] = 0.0


    # ==================================================
    # TCP window statistics
    # ==================================================

    if (
        "window_id" in df.columns
        and "tcp_window" in df.columns
    ):

        df["tcp_window_mean"] = (
            df.groupby("window_id")[
                "tcp_window"
            ]
            .transform("mean")
            .fillna(0)
        )

        df["tcp_window_variance"] = (
            df.groupby("window_id")[
                "tcp_window"
            ]
            .transform(
                lambda x:
                x.var(ddof=0)
            )
            .fillna(0)
        )

    else:

        df["tcp_window_mean"] = 0.0
        df["tcp_window_variance"] = 0.0


    # ==================================================
    # Payload statistics
    # ==================================================

    if (
        "window_id" in df.columns
        and "payload_size" in df.columns
    ):

        df["payload_mean"] = (
            df.groupby("window_id")[
                "payload_size"
            ]
            .transform("mean")
            .fillna(0)
        )

        df["payload_variance"] = (
            df.groupby("window_id")[
                "payload_size"
            ]
            .transform(
                lambda x:
                x.var(ddof=0)
            )
            .fillna(0)
        )

        df["payload_max"] = (
            df.groupby("window_id")[
                "payload_size"
            ]
            .transform("max")
            .fillna(0)
        )

    else:

        df["payload_mean"] = 0.0
        df["payload_variance"] = 0.0
        df["payload_max"] = 0.0


    # ==================================================
    # Fragment statistics
    # ==================================================

    if "fragment_flag" in df.columns:

        df["fragment_flag"] = (
            pd.to_numeric(
                df["fragment_flag"],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )

    else:

        df["fragment_flag"] = 0


    # ==================================================
    # Retransmission statistics
    # ==================================================

    if "retransmission" in df.columns:

        df["retransmission"] = (
            pd.to_numeric(
                df["retransmission"],
                errors="coerce"
            )
            .fillna(0)
        )

    else:

        df["retransmission"] = 0


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
                ["window_id", "src_ip"]
            )["dst_port"]
            .transform("nunique")
        )

        host_count = (
            df.groupby(
                ["window_id", "src_ip"]
            )["dst_ip"]
            .transform("nunique")
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


    return df