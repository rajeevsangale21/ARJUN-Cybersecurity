import pandas as pd


def align_timestamps(
    df,
    window_seconds=5
):

    df = df.copy()

    if "timestamp" not in df.columns:

        raise ValueError(
            "ARJUN requires a timestamp column."
        )

    numeric_timestamp = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    # PCAP timestamps are normally Unix timestamps
    if numeric_timestamp.notna().mean() > 0.8:

        df["timestamp"] = pd.to_datetime(
            numeric_timestamp,
            unit="s",
            errors="coerce",
            utc=True
        )

    else:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
            utc=True
        )

    df = df.dropna(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    df = df.reset_index(
        drop=True
    )

    if df.empty:

        raise ValueError(
            "No valid timestamps found."
        )

    start_time = df["timestamp"].min()

    elapsed_seconds = (
        df["timestamp"] - start_time
    ).dt.total_seconds()

    # Assign each event to a time window
    df["window_id"] = (
        elapsed_seconds //
        window_seconds
    ).astype(int)

    df["window_start"] = (
        start_time
        +
        pd.to_timedelta(
            df["window_id"] *
            window_seconds,
            unit="s"
        )
    )

    return df