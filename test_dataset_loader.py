from pathlib import Path

import pandas as pd

from ingestion.csv_loader import (
    discover_csv_files,
    iter_csv,
)

from preprocessing.cleaner import (
    clean_data,
    create_binary_attack_label,
    create_attack_type,
    validate_required_columns,
)


# ============================================================
# ARJUN DATASET LOADER TEST
# ============================================================

DATASET_DIR = Path(
    "data/raw/dataset"
)

MAX_ROWS = 20_000
CHUNK_SIZE = 5_000


def main():

    print()
    print("=" * 70)
    print("ARJUN DATASET LOADER TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # FIND DATASET FILES
    # --------------------------------------------------------

    print(
        "\n[1/6] Searching for CSV files..."
    )

    files = discover_csv_files(
        DATASET_DIR
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in: "
            f"{DATASET_DIR.resolve()}"
        )

    print(
        f"Found {len(files)} CSV files."
    )

    for file in files:
        print(
            f"  - {file.name}"
        )

    # --------------------------------------------------------
    # SELECT FIRST FILE
    # --------------------------------------------------------

    file_path = files[0]

    print(
        "\n[2/6] Testing file:"
    )

    print(
        f"  {file_path}"
    )

    # --------------------------------------------------------
    # LOAD LIMITED DATA
    # --------------------------------------------------------

    print(
        "\n[3/6] Loading first "
        f"{MAX_ROWS:,} rows in chunks..."
    )

    chunks = []

    rows_loaded = 0

    for chunk in iter_csv(
        file_path,
        chunk_size=CHUNK_SIZE,
        nrows=MAX_ROWS,
    ):

        print(
            f"  Chunk: "
            f"{len(chunk):,} rows"
        )

        chunks.append(chunk)

        rows_loaded += len(chunk)

    if not chunks:
        raise RuntimeError(
            "No rows were loaded from the dataset."
        )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    print(
        f"\nTotal rows loaded: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    print(
        "\n[4/6] Cleaning data..."
    )

    before = len(df)

    df = clean_data(
        df
    )

    after = len(df)

    print(
        f"Rows before cleaning: "
        f"{before:,}"
    )

    print(
        f"Rows after cleaning : "
        f"{after:,}"
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    print(
        "\n[5/6] Creating ARJUN labels..."
    )

    if "Label" not in df.columns:
        raise ValueError(
            "The dataset does not contain "
            "the expected 'Label' column."
        )

    df = create_binary_attack_label(
        df
    )

    df = create_attack_type(
        df
    )

    print(
        "\nOriginal labels:"
    )

    print(
        df["Label"]
        .value_counts()
        .to_string()
    )

    print(
        "\nBinary labels:"
    )

    print(
        df["binary_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # REQUIRED COLUMN CHECK
    # --------------------------------------------------------

    print(
        "\n[6/6] Checking ARJUN feature columns..."
    )

    availability = (
        validate_required_columns(
            df
        )
    )

    for name, available in (
        availability.items()
    ):

        status = (
            "OK"
            if available
            else "MISSING"
        )

        print(
            f"  {name:22s}: {status}"
        )

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    print(
        "\nData types:"
    )

    print(
        df.dtypes.to_string()
    )

    # --------------------------------------------------------
    # TIMESTAMP CHECK
    # --------------------------------------------------------

    if "Timestamp" in df.columns:

        valid_timestamps = (
            df["Timestamp"]
            .notna()
            .sum()
        )

        print()
        print(
            f"Valid timestamps: "
            f"{valid_timestamps:,}"
        )

        print(
            f"Invalid timestamps: "
            f"{len(df) - valid_timestamps:,}"
        )

    # --------------------------------------------------------
    # NUMERIC VALIDATION
    # --------------------------------------------------------

    numeric_df = df.select_dtypes(
        include="number"
    )

    print()
    print(
        f"Numeric columns: "
        f"{len(numeric_df.columns)}"
    )

    if not numeric_df.empty:

        inf_count = (
            numeric_df
            .isin([
                float("inf"),
                float("-inf"),
            ])
            .sum()
            .sum()
        )

        nan_count = (
            numeric_df
            .isna()
            .sum()
            .sum()
        )

        print(
            f"NaN values: "
            f"{int(nan_count):,}"
        )

        print(
            f"Infinite values: "
            f"{int(inf_count):,}"
        )

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    print()
    print(
        "First 5 cleaned rows:"
    )

    display_columns = [
        column
        for column in [
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Dst Port",
            "Protocol",
            "Timestamp",
            "Flow Duration",
            "Tot Fwd Pkts",
            "Tot Bwd Pkts",
            "Label",
            "binary_label",
            "attack_type",
        ]
        if column in df.columns
    ]

    print(
        df[
            display_columns
        ].head().to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET LOADER TEST PASSED")
    print("=" * 70)

    print(
        "\nThe dataset can be read safely "
        "in chunks."
    )

    print(
        "Next step: large-scale preprocessing."
    )


if __name__ == "__main__":
    main()