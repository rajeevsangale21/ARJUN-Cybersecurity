def add_contextual_features(df):

    df = df.copy()

    if "src_ip" in df.columns:

        df["src_host_known"] = (
            df["src_ip"]
            .notna()
            .astype(int)
        )

    if "dst_ip" in df.columns:

        df["dst_host_known"] = (
            df["dst_ip"]
            .notna()
            .astype(int)
        )

    return df