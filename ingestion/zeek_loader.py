"""
ARJUN Zeek conn.log loader.

Supports:
    1. Classic Zeek TSV conn.log
    2. Zeek JSON Lines conn.log

Normalizes Zeek fields into the schema expected by
ARJUN's Zeek -> Network State pipeline.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# =====================================================================
# NORMALIZED SCHEMA
# =====================================================================

OUTPUT_COLUMNS = [
    "ts",
    "uid",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "protocol",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "conn_state",
    "history",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "missed_bytes",
]


# =====================================================================
# HELPERS
# =====================================================================

def _clean_value(value):
    """Convert Zeek missing-value markers to None."""
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    text = str(value).strip()

    if text in {"", "-", "(empty)", "null", "None"}:
        return None

    return value


def _numeric_series(series: pd.Series) -> pd.Series:
    """Convert a Zeek numeric column safely."""
    return pd.to_numeric(
        series.replace({"-": np.nan, "": np.nan}),
        errors="coerce",
    )


def _parse_timestamps(raw_ts: pd.Series) -> pd.Series:
    """
    Parse Zeek timestamps safely.

    Handles:
        - Unix epoch seconds
        - Unix epoch milliseconds
        - ISO-like datetime strings
        - fractional seconds

    Important:
    The result is explicitly constructed as datetime64[ns] so that
    fractional timestamps never trigger pandas dtype-casting errors.
    """

    raw = raw_ts.copy()

    # Always start with an object/string representation.
    text = raw.astype("string").str.strip()

    result = pd.Series(
        pd.NaT,
        index=raw.index,
        dtype="datetime64[ns]",
    )

    # ---------------------------------------------------------------
    # Numeric epoch timestamps
    # ---------------------------------------------------------------

    numeric = pd.to_numeric(text, errors="coerce")

    numeric_mask = numeric.notna()

    if numeric_mask.any():

        values = numeric.loc[numeric_mask]

        # Zeek timestamps are normally Unix epoch seconds.
        # Millisecond values are also accepted.
        magnitude = values.abs()

        seconds_mask = magnitude < 1e11
        millis_mask = ~seconds_mask

        if seconds_mask.any():
            parsed_seconds = pd.to_datetime(
                values.loc[seconds_mask],
                unit="s",
                errors="coerce",
                utc=False,
            )

            result.loc[parsed_seconds.index] = parsed_seconds

        if millis_mask.any():
            parsed_millis = pd.to_datetime(
                values.loc[millis_mask],
                unit="ms",
                errors="coerce",
                utc=False,
            )

            result.loc[parsed_millis.index] = parsed_millis

    # ---------------------------------------------------------------
    # String timestamps
    # ---------------------------------------------------------------

    string_mask = ~numeric_mask & text.notna()

    if string_mask.any():
        parsed_strings = pd.to_datetime(
            text.loc[string_mask],
            errors="coerce",
            utc=False,
            format="mixed",
        )

        result.loc[parsed_strings.index] = parsed_strings

    return result


# =====================================================================
# CLASSIC ZEEK TSV PARSER
# =====================================================================

def _read_zeek_tsv(path: Path) -> pd.DataFrame:
    """
    Read a classic Zeek TSV log.

    Zeek headers look like:

        #separator \x09
        #set_separator ,
        #empty_field (empty)
        #unset_field -
        #path conn
        #open ...
        #fields ts uid id.orig_h ...
        #types time string addr port ...
    """

    fields = None
    rows = []

    separator = "\t"

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as fh:

        for raw_line in fh:
            line = raw_line.rstrip("\r\n")

            if not line:
                continue

            # -------------------------------------------------------
            # Zeek metadata
            # -------------------------------------------------------

            if line.startswith("#separator"):
                parts = line.split(maxsplit=1)

                if len(parts) == 2:
                    value = parts[1]

                    if value == r"\x09":
                        separator = "\t"
                    else:
                        try:
                            separator = bytes(
                                value,
                                "utf-8",
                            ).decode(
                                "unicode_escape"
                            )
                        except Exception:
                            separator = "\t"

                continue

            if line.startswith("#fields"):
                parts = line.split(separator)

                if len(parts) == 1:
                    parts = line.split()

                fields = parts[1:]

                continue

            if line.startswith("#"):
                continue

            # -------------------------------------------------------
            # Data
            # -------------------------------------------------------

            if fields is None:
                # Allow simple TSV files without Zeek metadata.
                parts = line.split(separator)

                if not parts:
                    continue

                if not rows:
                    fields = [
                        f"field_{i}"
                        for i in range(len(parts))
                    ]

            parts = line.split(separator)

            if len(parts) < len(fields):
                parts.extend(
                    ["-"] * (len(fields) - len(parts))
                )

            if len(parts) > len(fields):
                parts = parts[:len(fields)]

            rows.append(parts)

    if fields is None or not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows,
        columns=fields,
    )


# =====================================================================
# JSON ZEEK PARSER
# =====================================================================

def _read_zeek_json(path: Path) -> pd.DataFrame:
    """Read Zeek JSON Lines."""
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as fh:

        for line in fh:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

                if isinstance(obj, dict):
                    rows.append(obj)

            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =====================================================================
# FORMAT DETECTION
# =====================================================================

def _detect_format(path: Path) -> str:
    """
    Detect classic Zeek TSV versus JSON Lines.
    """

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as fh:

        for line in fh:

            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                return "tsv"

            if stripped.startswith("{"):
                return "json"

            if "\t" in line:
                return "tsv"

            return "tsv"

    return "tsv"


# =====================================================================
# COLUMN ALIASES
# =====================================================================

ALIASES = {
    "ts": [
        "ts",
        "timestamp",
        "time",
    ],
    "uid": [
        "uid",
    ],
    "src_ip": [
        "id.orig_h",
        "orig_h",
        "src_ip",
        "src",
        "source_ip",
    ],
    "src_port": [
        "id.orig_p",
        "orig_p",
        "src_port",
        "source_port",
    ],
    "dst_ip": [
        "id.resp_h",
        "resp_h",
        "dst_ip",
        "dst",
        "destination_ip",
    ],
    "dst_port": [
        "id.resp_p",
        "resp_p",
        "dst_port",
        "destination_port",
    ],
    "protocol": [
        "proto",
        "protocol",
    ],
    "duration": [
        "duration",
    ],
    "orig_bytes": [
        "orig_bytes",
    ],
    "resp_bytes": [
        "resp_bytes",
    ],
    "orig_pkts": [
        "orig_pkts",
    ],
    "resp_pkts": [
        "resp_pkts",
    ],
    "conn_state": [
        "conn_state",
    ],
    "history": [
        "history",
    ],
    "orig_ip_bytes": [
        "orig_ip_bytes",
    ],
    "resp_ip_bytes": [
        "resp_ip_bytes",
    ],
    "missed_bytes": [
        "missed_bytes",
    ],
}


def _find_column(
    df: pd.DataFrame,
    candidates: list[str],
):
    """Find the first matching column."""
    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in lower_map:
            return lower_map[key]

    return None


# =====================================================================
# NORMALIZATION
# =====================================================================

def normalize_conn(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw Zeek connection records.
    """

    if raw is None or raw.empty:
        raise ValueError(
            "Zeek conn.log contains no rows."
        )

    output = pd.DataFrame(
        index=raw.index
    )

    # ---------------------------------------------------------------
    # Map columns
    # ---------------------------------------------------------------

    for target, candidates in ALIASES.items():

        source = _find_column(
            raw,
            candidates,
        )

        if source is None:
            output[target] = None
        else:
            output[target] = raw[source].map(
                _clean_value
            )

    # ---------------------------------------------------------------
    # Timestamp
    # ---------------------------------------------------------------

    output["ts"] = _parse_timestamps(
        output["ts"]
    )

    # ---------------------------------------------------------------
    # Numeric fields
    # ---------------------------------------------------------------

    numeric_columns = [
        "src_port",
        "dst_port",
        "duration",
        "orig_bytes",
        "resp_bytes",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        "missed_bytes",
    ]

    for column in numeric_columns:
        output[column] = _numeric_series(
            output[column]
        ).fillna(0.0)

    # ---------------------------------------------------------------
    # String fields
    # ---------------------------------------------------------------

    string_columns = [
        "uid",
        "src_ip",
        "dst_ip",
        "protocol",
        "conn_state",
        "history",
    ]

    for column in string_columns:
        output[column] = (
            output[column]
            .fillna("")
            .astype(str)
        )

    # ---------------------------------------------------------------
    # Remove unusable timestamps
    # ---------------------------------------------------------------

    output = output.dropna(
        subset=["ts"]
    ).copy()

    if output.empty:
        raise ValueError(
            "No valid timestamps found in Zeek conn.log."
        )

    # ---------------------------------------------------------------
    # Sort chronologically
    # ---------------------------------------------------------------

    output = output.sort_values(
        "ts",
        kind="mergesort",
    ).reset_index(drop=True)

    return output[
        OUTPUT_COLUMNS
    ]


# =====================================================================
# PUBLIC LOADER
# =====================================================================

def load_zeek_conn(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load and normalize a Zeek conn.log file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Zeek log not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Zeek input is not a file: {path}"
        )

    file_format = _detect_format(path)

    if file_format == "json":
        raw = _read_zeek_json(path)
    else:
        raw = _read_zeek_tsv(path)

    if raw.empty:
        raise ValueError(
            f"No records could be read from: {path}"
        )

    return normalize_conn(raw)


# =====================================================================
# STANDALONE TEST
# =====================================================================

def main():
    """
    Standalone loader test.

    Usage:
        python ingestion/zeek_loader.py

    Uses:
        data/raw/logs/synthetic_conn.log
    """

    root = Path(__file__).resolve().parent.parent

    default_input = (
        root
        / "data"
        / "raw"
        / "logs"
        / "synthetic_conn.log"
    )

    print("=" * 72)
    print("ARJUN ZEEK CONN.LOG LOADER")
    print("=" * 72)
    print(f"Input : {default_input}")

    df = load_zeek_conn(
        default_input
    )

    print(
        f"Rows loaded : {len(df)}"
    )

    print(
        f"Columns     : {len(df.columns)}"
    )

    print(
        f"Time range  : "
        f"{df['ts'].min()} -> {df['ts'].max()}"
    )

    print()
    print(
        "Timestamp dtype:",
        df["ts"].dtype,
    )

    print()
    print(
        df.head(3).to_string(
            index=False
        )
    )

    print()
    print(
        "ZEEK CONN LOADER: PASSED"
    )


if __name__ == "__main__":
    main()