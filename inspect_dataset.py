import os
import json
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# ARJUN - DATASET INSPECTION
# ============================================================

DATASET_DIR = Path("data/raw/dataset")
OUTPUT_FILE = Path("data/processed/dataset_inspection.json")

CHUNK_SIZE = 100_000
SAMPLE_ROWS = 5


# ============================================================
# HELPERS
# ============================================================

def normalize_column_name(name):
    """
    Normalize column names so that small formatting differences
    do not cause problems.
    """

    name = str(name)

    name = (
        name.replace("\ufeff", "")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )

    name = " ".join(name.split())

    return name


def find_column(columns, candidates):
    """
    Find a column using case-insensitive matching.
    """

    normalized = {
        normalize_column_name(col).lower(): col
        for col in columns
    }

    for candidate in candidates:

        candidate_key = (
            normalize_column_name(candidate)
            .lower()
        )

        if candidate_key in normalized:
            return normalized[candidate_key]

    return None


def find_timestamp_columns(columns):
    """
    Find columns that look like timestamps.
    """

    results = []

    keywords = [
        "timestamp",
        "time",
        "date",
    ]

    for column in columns:

        name = normalize_column_name(
            column
        ).lower()

        if any(
            keyword in name
            for keyword in keywords
        ):
            results.append(column)

    return results


# ============================================================
# INSPECT ONE CSV
# ============================================================

def inspect_csv(file_path):

    print()
    print("-" * 70)
    print(f"Inspecting: {file_path.name}")
    print("-" * 70)

    file_size_mb = (
        file_path.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"File size: {file_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # READ HEADER ONLY
    # --------------------------------------------------------

    try:

        header_df = pd.read_csv(
            file_path,
            nrows=0,
            low_memory=False,
        )

    except Exception as exc:

        print(
            f"ERROR reading header: {exc}"
        )

        return {
            "file": file_path.name,
            "error": str(exc),
        }

    original_columns = list(
        header_df.columns
    )

    columns = [
        normalize_column_name(col)
        for col in original_columns
    ]

    print(
        f"Number of columns: {len(columns)}"
    )

    print("\nColumns:")

    for index, column in enumerate(columns):

        print(
            f"  {index + 1:02d}. {column}"
        )

    # --------------------------------------------------------
    # FIND IMPORTANT COLUMNS
    # --------------------------------------------------------

    label_column = find_column(
        columns,
        [
            "Label",
            "label",
            "Attack",
            "Class",
        ],
    )

    timestamp_columns = (
        find_timestamp_columns(
            columns
        )
    )

    src_ip_column = find_column(
        columns,
        [
            "Src IP",
            "Source IP",
            "SourceIP",
            "src_ip",
        ],
    )

    dst_ip_column = find_column(
        columns,
        [
            "Dst IP",
            "Destination IP",
            "DestinationIP",
            "dst_ip",
        ],
    )

    protocol_column = find_column(
        columns,
        [
            "Protocol",
            "protocol",
        ],
    )

    src_port_column = find_column(
        columns,
        [
            "Src Port",
            "Source Port",
            "sport",
        ],
    )

    dst_port_column = find_column(
        columns,
        [
            "Dst Port",
            "Destination Port",
            "dport",
        ],
    )

    print("\nImportant columns found:")

    print(
        f"  Label      : {label_column}"
    )

    print(
        f"  Timestamp  : {timestamp_columns}"
    )

    print(
        f"  Source IP  : {src_ip_column}"
    )

    print(
        f"  Dest IP    : {dst_ip_column}"
    )

    print(
        f"  Protocol   : {protocol_column}"
    )

    print(
        f"  Source Port: {src_port_column}"
    )

    print(
        f"  Dest Port  : {dst_port_column}"
    )

    # --------------------------------------------------------
    # SAMPLE DATA
    # --------------------------------------------------------

    print(
        f"\nReading first {SAMPLE_ROWS} rows..."
    )

    try:

        sample_df = pd.read_csv(
            file_path,
            nrows=SAMPLE_ROWS,
            low_memory=False,
        )

        sample_df.columns = columns

        print("\nSample:")

        print(
            sample_df.to_string(
                max_cols=12
            )
        )

    except Exception as exc:

        print(
            f"ERROR reading sample: {exc}"
        )

        sample_df = pd.DataFrame()

    # --------------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------------

    label_counts = Counter()
    row_count = 0

    if label_column is not None:

        print(
            "\nScanning labels in chunks..."
        )

        try:

            for chunk in pd.read_csv(
                file_path,
                usecols=[
                    label_column
                ],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ):

                chunk.columns = [
                    label_column
                ]

                values = (
                    chunk[label_column]
                    .fillna("MISSING")
                    .astype(str)
                    .str.strip()
                )

                label_counts.update(
                    values.tolist()
                )

                row_count += len(chunk)

                print(
                    f"\rRows scanned: "
                    f"{row_count:,}",
                    end="",
                    flush=True,
                )

            print()

        except Exception as exc:

            print(
                f"\nERROR scanning labels: "
                f"{exc}"
            )

    else:

        print(
            "\nNo label column found."
        )

        try:

            for chunk in pd.read_csv(
                file_path,
                usecols=[columns[0]],
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ):

                row_count += len(chunk)

                print(
                    f"\rRows scanned: "
                    f"{row_count:,}",
                    end="",
                    flush=True,
                )

            print()

        except Exception as exc:

            print(
                f"\nCould not count rows: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # LABEL OUTPUT
    # --------------------------------------------------------

    print("\nLabel distribution:")

    for label, count in label_counts.most_common():

        print(
            f"  {label}: {count:,}"
        )

    # --------------------------------------------------------
    # BASIC NUMERIC CHECK
    # --------------------------------------------------------

    numeric_summary = {}

    numeric_candidates = [
        "Flow Duration",
        "Tot Fwd Pkts",
        "Tot Bwd Pkts",
        "TotLen Fwd Pkts",
        "TotLen Bwd Pkts",
        "Flow Byts/s",
        "Flow Pkts/s",
        "Fwd Pkts/s",
        "Bwd Pkts/s",
        "Packet Length Mean",
        "Packet Length Std",
        "Average Packet Size",
    ]

    available_numeric = []

    for candidate in numeric_candidates:

        found = find_column(
            columns,
            [candidate],
        )

        if found is not None:

            available_numeric.append(
                found
            )

    if available_numeric:

        print(
            "\nChecking numeric features..."
        )

        try:

            numeric_df = pd.read_csv(
                file_path,
                usecols=available_numeric,
                nrows=50_000,
                low_memory=False,
            )

            numeric_df.columns = [
                normalize_column_name(
                    col
                )
                for col in numeric_df.columns
            ]

            for column in numeric_df.columns:

                values = pd.to_numeric(
                    numeric_df[column],
                    errors="coerce",
                )

                numeric_summary[column] = {
                    "missing": int(
                        values.isna().sum()
                    ),
                    "infinite": int(
                        (~values.isfinite())
                        .sum()
                    ),
                    "min": (
                        float(values.min())
                        if values.notna().any()
                        else None
                    ),
                    "max": (
                        float(values.max())
                        if values.notna().any()
                        else None
                    ),
                }

                print(
                    f"  {column}: "
                    f"missing="
                    f"{numeric_summary[column]['missing']}, "
                    f"infinite="
                    f"{numeric_summary[column]['infinite']}"
                )

        except Exception as exc:

            print(
                f"Numeric inspection failed: "
                f"{exc}"
            )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {
        "file": file_path.name,
        "path": str(file_path),
        "size_mb": round(
            file_size_mb,
            2,
        ),
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "important_columns": {
            "label": label_column,
            "timestamp": timestamp_columns,
            "source_ip": src_ip_column,
            "destination_ip": dst_ip_column,
            "protocol": protocol_column,
            "source_port": src_port_column,
            "destination_port": dst_port_column,
        },
        "label_distribution": dict(
            label_counts
        ),
        "numeric_check": numeric_summary,
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ARJUN DATASET INSPECTION")
    print("=" * 70)

    print(
        f"\nDataset directory:"
        f" {DATASET_DIR.resolve()}"
    )

    if not DATASET_DIR.exists():

        print()
        print(
            "ERROR: Dataset directory does not exist."
        )

        print(
            "\nExpected:"
        )

        print(
            "ARJUN/data/raw/dataset/"
        )

        return

    # --------------------------------------------------------
    # FIND CSV FILES
    # --------------------------------------------------------

    csv_files = sorted(
        DATASET_DIR.rglob("*.csv")
    )

    if not csv_files:

        print()
        print(
            "ERROR: No CSV files found."
        )

        print(
            "\nMake sure the extracted Kaggle "
            "files are inside:"
        )

        print(
            DATASET_DIR.resolve()
        )

        return

    print(
        f"\nCSV files found: "
        f"{len(csv_files)}"
    )

    for file in csv_files:

        print(
            f"  - {file.name}"
        )

    # --------------------------------------------------------
    # INSPECT ALL FILES
    # --------------------------------------------------------

    results = []

    for file_path in csv_files:

        result = inspect_csv(
            file_path
        )

        results.append(result)

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_size = sum(
        result.get(
            "size_mb",
            0,
        )
        for result in results
    )

    total_rows = sum(
        result.get(
            "row_count",
            0,
        )
        for result in results
    )

    all_labels = Counter()

    for result in results:

        all_labels.update(
            result.get(
                "label_distribution",
                {},
            )
        )

    print()
    print("=" * 70)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 70)

    print(
        f"\nFiles: {len(csv_files)}"
    )

    print(
        f"Total CSV size: "
        f"{total_size:.2f} MB"
    )

    print(
        f"Total rows: "
        f"{total_rows:,}"
    )

    print(
        "\nCombined label distribution:"
    )

    for label, count in (
        all_labels.most_common()
    ):

        print(
            f"  {label}: {count:,}"
        )

    print(
        f"\nReport saved to:"
    )

    print(
        OUTPUT_FILE.resolve()
    )

    print()
    print(
        "ARJUN is ready for the next phase."
    )


if __name__ == "__main__":
    main()