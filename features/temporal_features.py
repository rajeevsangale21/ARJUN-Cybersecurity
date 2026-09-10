def add_temporal_features(df):

    df = df.copy()

    # Time between consecutive network events
    df["iat_seconds"] = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
        .clip(lower=0)
    )

    # Time between events from the same source
    if "src_ip" in df.columns:

        df["src_iat_seconds"] = (
            df.groupby("src_ip")[
                "timestamp"
            ]
            .diff()
            .dt.total_seconds()
            .fillna(0)
            .clip(lower=0)
        )

    return df