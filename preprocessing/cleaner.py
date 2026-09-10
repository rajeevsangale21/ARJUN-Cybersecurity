"""
ARJUN - Data Cleaning Utilities

Cleans and validates network telemetry while preserving the
canonical column names used throughout the ARJUN pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Column utilities
# ============================================================

def _clean_column_name(name: str) -> str:
    """
    Convert a raw column name into a normalized snake_case name.
    """

    name = str(name).strip().lower()

    replacements = {
        " ": "_",
        "-": "_",
        "/": "_",
        "\\": "_",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        ".": "_",
        ":": "_",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataframe column names.
    """

    result = df.copy()

    result.columns = [
        _clean_column_name(column)
        for column in result.columns
    ]

    return result


# ============================================================
# Label utilities
# ============================================================

def _find_label_column(
    df: pd.DataFrame,
    label_column: str = "Label",
) -> str:
    """
    Locate the label column case-insensitively.

    Returns the actual dataframe column name.
    """

    requested = _clean_column_name(
        label_column
    )

    for column in df.columns:

        if _clean_column_name(column) == requested:
            return column

    # Common alternatives
    alternatives = [
        "label",
        "attack",
        "attack_type",
        "class",
        "target",
    ]

    for alternative in alternatives:

        for column in df.columns:

            if _clean_column_name(column) == alternative:
                return column

    raise ValueError(
        "Could not find a label column. "
        f"Requested: {label_column}"
    )


def _normalize_label_value(value) -> str:
    """
    Normalize an individual attack label.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# Main cleaning function
# ============================================================

def clean_data(
    df: pd.DataFrame,
    label_column: str = "Label",
) -> pd.DataFrame:
    """
    Clean a network telemetry dataframe.

    Operations:
        1. Normalize column names
        2. Remove repeated CSV headers
        3. Clean labels
        4. Parse timestamps
        5. Convert numeric columns
        6. Replace infinite values
        7. Fill numeric missing values
        8. Remove invalid rows
        9. Remove duplicate rows
    """

    if df is None:
        raise ValueError(
            "Input dataframe is None."
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "clean_data expects a pandas DataFrame."
        )

    if df.empty:
        return df.copy()

    result = _clean_column_names(df)

    # --------------------------------------------------------
    # Find label column AFTER column normalization
    # --------------------------------------------------------

    label_column_actual = _find_label_column(
        result,
        label_column,
    )

    # --------------------------------------------------------
    # Normalize label values
    # --------------------------------------------------------

    result[label_column_actual] = (
        result[label_column_actual]
        .map(_normalize_label_value)
    )

    # --------------------------------------------------------
    # Remove repeated CSV header rows.
    #
    # CIC-IDS files occasionally contain rows where Label itself
    # contains the literal string "Label".
    # --------------------------------------------------------

    repeated_header = (
        result[label_column_actual]
        .str.strip()
        .str.lower()
        == "label"
    )

    result = result.loc[
        ~repeated_header
    ].copy()

    # --------------------------------------------------------
    # Remove rows with no label
    # --------------------------------------------------------

    result = result.loc[
        result[label_column_actual].str.len() > 0
    ].copy()

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_column = None

    for candidate in [
        "timestamp",
        "time",
        "datetime",
        "date_time",
    ]:

        if candidate in result.columns:
            timestamp_column = candidate
            break

    if timestamp_column is not None:
        numeric_ts = pd.to_numeric(
            result[timestamp_column],
            errors="coerce",
        )

        if numeric_ts.notna().mean() > 0.8:
            result[timestamp_column] = pd.to_datetime(
                numeric_ts,
                unit="s",
                errors="coerce",
                utc=True,
            )
        else:
            result[timestamp_column] = pd.to_datetime(
                result[timestamp_column],
                errors="coerce",
                dayfirst=True,
            )

    # --------------------------------------------------------
    # Numeric conversion
    #
    # Only convert columns that are not obvious identifiers or
    # categorical values.
    # --------------------------------------------------------

    excluded = {
        label_column_actual,
        "label",
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "src_ip",
        "dst_ip",
        "source_ip",
        "destination_ip",
        "flow_id",
        "protocol",
        "attack_type",
    }

    for column in result.columns:

        if column in excluded:
            continue

        # Don't blindly convert object columns containing
        # meaningful identifiers.
        if (
            pd.api.types.is_object_dtype(
                result[column]
            )
        ):
            converted = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            valid_ratio = (
                converted.notna().mean()
                if len(converted) > 0
                else 0.0
            )

            # Convert only if most values are numeric.
            if valid_ratio >= 0.80:
                result[column] = converted

        elif pd.api.types.is_numeric_dtype(
            result[column]
        ):
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_columns) > 0:

        result[numeric_columns] = (
            result[numeric_columns]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    # --------------------------------------------------------
    # Fill numeric NaN values using column medians.
    #
    # If a column has no finite values, use zero.
    # --------------------------------------------------------

    for column in numeric_columns:

        series = result[column]

        if series.notna().any():

            median_value = series.median()

            if pd.isna(median_value):
                median_value = 0.0

        else:
            median_value = 0.0

        result[column] = (
            series.fillna(median_value)
        )

    # --------------------------------------------------------
    # Drop rows with invalid timestamps
    # --------------------------------------------------------

    if timestamp_column is not None:

        result = result.dropna(
            subset=[timestamp_column]
        )

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    result = result.drop_duplicates()

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    result = result.reset_index(
        drop=True
    )

    return result


# ============================================================
# Binary attack label
# ============================================================

def create_binary_attack_label(
    df: pd.DataFrame,
    label_column: str = "Label",
) -> pd.Series:
    """
    Convert attack labels into binary labels.

    Benign -> 0
    Any non-benign attack -> 1

    Returns:
        pandas Series aligned with df.index
    """

    if df is None:
        raise ValueError(
            "Input dataframe is None."
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        return pd.Series(
            dtype=np.int8,
            index=df.index,
            name="binary_label",
        )

    actual_column = _find_label_column(
        df,
        label_column,
    )

    labels = (
        df[actual_column]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
    )

    binary = (
        labels
        .ne("benign")
        .astype(np.int8)
    )

    binary.name = "binary_label"

    return binary


# ============================================================
# Attack type
# ============================================================

def create_attack_type(
    df: pd.DataFrame,
    label_column: str = "Label",
) -> pd.Series:
    """
    Return normalized attack-type labels.
    """

    if df is None:
        raise ValueError(
            "Input dataframe is None."
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        return pd.Series(
            dtype="string",
            index=df.index,
            name="attack_type",
        )

    actual_column = _find_label_column(
        df,
        label_column,
    )

    attack_type = (
        df[actual_column]
        .astype("string")
        .fillna("unknown")
        .str.strip()
    )

    attack_type = attack_type.mask(
        attack_type.str.len() == 0,
        "unknown",
    )

    attack_type.name = "attack_type"

    return attack_type


# ============================================================
# Required column validation
# ============================================================

def validate_required_columns(
    df: pd.DataFrame,
    required_columns,
) -> None:
    """
    Raise an informative error if required columns are missing.
    """

    if df is None:
        raise ValueError(
            "Input dataframe is None."
        )

    normalized = {
        _clean_column_name(column)
        for column in df.columns
    }

    missing = []

    for column in required_columns:

        normalized_required = _clean_column_name(
            column
        )

        if normalized_required not in normalized:
            missing.append(column)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                map(str, missing)
            )
        )


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    sample = pd.DataFrame(
        {
            "Label": [
                "Benign",
                "FTP-BruteForce",
                "DDOS attack-HOIC",
                "Label",
            ],
            "Flow Duration": [
                100,
                200,
                300,
                "Flow Duration",
            ],
        }
    )

    print(
        "Before cleaning:"
    )
    print(sample)

    cleaned = clean_data(
        sample,
        label_column="Label",
    )

    binary = create_binary_attack_label(
        cleaned,
        label_column="label",
    )

    attacks = create_attack_type(
        cleaned,
        label_column="label",
    )

    print()
    print(
        "After cleaning:"
    )
    print(cleaned)

    print()
    print(
        "Binary labels:"
    )
    print(binary)

    print()
    print(
        "Attack types:"
    )
    print(attacks)