import numpy as np


def add_flow_features(df):

    df = df.copy()

    # Average bytes per packet
    if (
        "bytes" in df.columns
        and
        "packets" in df.columns
    ):

        packets = (
            df["packets"]
            .replace(0, np.nan)
        )

        df["bytes_per_packet"] = (
            df["bytes"] / packets
        ).fillna(0)

    # Traffic rates
    if "duration" in df.columns:

        duration = (
            df["duration"]
            .replace(0, np.nan)
        )

        if "packets" in df.columns:

            df["packets_per_second"] = (
                df["packets"] /
                duration
            ).fillna(0)

        if "bytes" in df.columns:

            df["bytes_per_second"] = (
                df["bytes"] /
                duration
            ).fillna(0)

    return df