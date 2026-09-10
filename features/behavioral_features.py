def add_behavioral_features(df):

    df = df.copy()

    if (
        "src_ip" in df.columns
        and
        "dst_port" in df.columns
    ):

        df[
            "unique_dst_ports_per_window"
        ] = (
            df.groupby(
                [
                    "window_id",
                    "src_ip"
                ]
            )["dst_port"]
            .transform("nunique")
        )

    if (
        "src_ip" in df.columns
        and
        "dst_ip" in df.columns
    ):

        df[
            "unique_dst_hosts_per_window"
        ] = (
            df.groupby(
                [
                    "window_id",
                    "src_ip"
                ]
            )["dst_ip"]
            .transform("nunique")
        )

    return df