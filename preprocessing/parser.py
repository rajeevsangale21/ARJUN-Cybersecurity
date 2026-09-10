"""
ARJUN - Robust CIC-IDS-2018 Parser

Purpose
-------
Convert raw CIC-IDS-2018 / flow / packet columns into the
canonical column names used by ARJUN.

Important
---------
CIC-IDS-2018 timestamps such as:

    02/14/2018 10:00:00

are MM/DD/YYYY.

This parser explicitly handles that format and does not allow
pandas to silently reinterpret the timestamp.

Features
--------
- MM/DD/YYYY timestamps
- DD/MM/YYYY timestamps
- Fractional seconds
- Already-parsed datetime values
- CIC duration conversion: microseconds -> seconds
- CIC IAT conversion: microseconds -> seconds
- Canonical column names
- Safe numeric conversion
- Packet/byte totals
- Timestamp validation
- No fabricated packet telemetry
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# ================================================================
# Canonical column aliases
# ================================================================

COLUMN_ALIASES = {

    # Network identity
    "src_ip": [
        "src ip",
        "source ip",
        "source_ip",
        "src_ip",
    ],

    "dst_ip": [
        "dst ip",
        "dest ip",
        "destination ip",
        "destination_ip",
        "dst_ip",
    ],

    "src_port": [
        "src port",
        "source port",
        "source_port",
        "src_port",
    ],

    "dst_port": [
        "dst port",
        "dest port",
        "destination port",
        "destination_port",
        "dst_port",
    ],

    # Basic network fields
    "protocol": [
        "protocol",
        "proto",
    ],

    "timestamp": [
        "timestamp",
        "time",
        "date",
        "datetime",
        "ts",
    ],

    # Flow duration
    "duration": [
        "duration",
        "flow duration",
        "flow_duration",
        "flow duration us",
        "flow duration (us)",
    ],

    # Packet counts
    "fwd_packets": [
        "tot fwd pkts",
        "total fwd packets",
        "total fwd pkts",
        "fwd packets",
        "fwd pkts",
        "tot_fwd_pkts",
        "fwd_packets",
    ],

    "bwd_packets": [
        "tot bwd pkts",
        "total bwd packets",
        "total bwd pkts",
        "bwd packets",
        "bwd pkts",
        "tot_bwd_pkts",
        "bwd_packets",
    ],

    # Byte counts
    "fwd_bytes": [
        "totlen fwd pkts",
        "total length of fwd packets",
        "total length of forward packets",
        "fwd bytes",
        "forward bytes",
        "tot_fwd_bytes",
        "fwd_bytes",
    ],

    "bwd_bytes": [
        "totlen bwd pkts",
        "total length of bwd packets",
        "total length of backward packets",
        "bwd bytes",
        "backward bytes",
        "tot_bwd_bytes",
        "bwd_bytes",
    ],

    # Packet length
    "packet_length": [
        "pkt len",
        "packet length",
        "packet_length",
        "pkt size",
        "pkt size avg",
        "packet size",
        "pkt_len",
        "pkt_size",
        "pkt_size_avg",
    ],

    # TTL
    "ttl": [
        "ttl",
        "ip ttl",
        "hop limit",
        "hop_limit",
    ],

    # TCP window
    "tcp_window": [
        "tcp window",
        "tcp_window",
        "init fwd win byts",
        "init bwd win byts",
        "initial tcp window",
    ],

    # Payload
    "payload_size": [
        "payload size",
        "payload_size",
        "tcp payload size",
        "udp payload size",
    ],

    # TCP flags
    "syn_count": [
        "syn flag cnt",
        "syn flag count",
        "syn_count",
        "syn count",
    ],

    "ack_count": [
        "ack flag cnt",
        "ack flag count",
        "ack_count",
        "ack count",
    ],

    "fin_count": [
        "fin flag cnt",
        "fin flag count",
        "fin_count",
        "fin count",
    ],

    "rst_count": [
        "rst flag cnt",
        "rst flag count",
        "rst_count",
        "rst count",
    ],

    "psh_count": [
        "psh flag cnt",
        "psh flag count",
        "psh_count",
        "psh count",
    ],

    "urg_count": [
        "urg flag cnt",
        "urg flag count",
        "urg_count",
        "urg count",
    ],

    # Fragmentation
    "fragment": [
        "fragment",
        "fragment flag",
        "fragment_flag",
        "fragment count",
        "fragment_count",
    ],

    # Retransmission
    "retransmission": [
        "retransmission",
        "retransmissions",
        "retransmission count",
        "retransmission_count",
    ],

    # IAT
    "iat_seconds": [
        "iat",
        "flow iat",
        "flow iat mean",
        "iat mean",
        "iat_seconds",
        "flow_iat_mean",
    ],

    "src_iat_seconds": [
        "src iat",
        "source iat",
        "src_iat",
        "src_iat_seconds",
    ],

    # Label
    "label": [
        "label",
        "attack",
        "class",
        "target",
    ],
}


# ================================================================
# Column-name normalization
# ================================================================

def normalize_column_name(name) -> str:
    """
    Normalize a raw column name.

    Examples
    --------
    Tot Fwd Pkts -> tot fwd pkts
    Flow Duration -> flow duration
    Src_IP -> src ip
    """

    text = str(name).strip().lower()

    text = text.replace("\ufeff", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_alias_lookup():
    """Build normalized alias -> canonical name lookup."""

    lookup = {}

    for canonical, aliases in COLUMN_ALIASES.items():

        for alias in aliases:

            lookup[
                normalize_column_name(alias)
            ] = canonical

    return lookup


ALIAS_LOOKUP = build_alias_lookup()


# ================================================================
# Safe numeric conversion
# ================================================================

def safe_numeric(series) -> pd.Series:
    """
    Safely convert a pandas Series to numeric.

    Invalid values become NaN.
    Infinite values become NaN.
    """

    if series is None:

        return pd.Series(
            dtype="float64"
        )

    result = series.copy()

    if result.dtype == object:

        result = (
            result
            .astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
        )

    result = pd.to_numeric(
        result,
        errors="coerce",
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


# ================================================================
# Timestamp helpers
# ================================================================

def _timestamp_component_order(
    series: pd.Series,
) -> str:
    """
    Determine the date ordering.

    Returns:
        "monthfirst"
        "dayfirst"

    Rules
    -----
    If first component > 12:
        DD/MM/YYYY

    If second component > 12:
        MM/DD/YYYY

    If ambiguous:
        MM/DD/YYYY

    CIC-IDS-2018 is treated as MM/DD/YYYY by default.
    """

    if series is None:

        return "monthfirst"

    values = (
        series
        .dropna()
        .astype(str)
        .str.strip()
    )

    if values.empty:

        return "monthfirst"

    first_values = []
    second_values = []

    for value in values.head(10000):

        match = re.match(
            r"^\s*(\d{1,2})/(\d{1,2})/\d{4}",
            value,
        )

        if match is None:
            continue

        first_values.append(
            int(match.group(1))
        )

        second_values.append(
            int(match.group(2))
        )

    if not first_values:

        return "monthfirst"

    if max(first_values) > 12:

        return "dayfirst"

    if max(second_values) > 12:

        return "monthfirst"

    # Ambiguous dates.
    #
    # CIC-IDS-2018 uses MM/DD/YYYY.
    return "monthfirst"


def parse_timestamp(
    series: pd.Series,
) -> pd.Series:
    """
    Parse timestamps safely.

    Supported formats:

        MM/DD/YYYY HH:MM:SS
        MM/DD/YYYY HH:MM:SS.sss
        DD/MM/YYYY HH:MM:SS
        DD/MM/YYYY HH:MM:SS.sss

    Already-datetime values are preserved.

    Invalid timestamps become NaT.

    The function deliberately avoids a blind:

        pd.to_datetime(..., dayfirst=True)

    because that can corrupt CIC-IDS-2018 dates.
    """

    if series is None:

        return pd.Series(
            dtype="datetime64[ns]"
        )

    # ------------------------------------------------------------
    # Already datetime or numeric epoch
    # ------------------------------------------------------------

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):

        return pd.to_datetime(
            series,
            errors="coerce",
        )

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric_series.notna().mean() > 0.8:
        median_val = float(numeric_series.dropna().median())
        if median_val > 1e8:
            return pd.to_datetime(
                numeric_series,
                unit="s",
                errors="coerce",
            )

    # ------------------------------------------------------------
    # Convert raw values to strings
    # ------------------------------------------------------------

    raw = (
        series
        .astype("string")
        .str.strip()
    )

    parsed = pd.Series(
        pd.NaT,
        index=raw.index,
        dtype="datetime64[ns]",
    )

    # ------------------------------------------------------------
    # Detect ordering
    # ------------------------------------------------------------

    order = _timestamp_component_order(
        raw
    )

    # ------------------------------------------------------------
    # Explicit formats
    # ------------------------------------------------------------

    if order == "monthfirst":

        formats = [
            "%m/%d/%Y %H:%M:%S.%f",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
        ]

    else:

        formats = [
            "%d/%m/%Y %H:%M:%S.%f",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
        ]

    for fmt in formats:

        missing = parsed.isna()

        if not missing.any():
            break

        candidate = pd.to_datetime(
            raw.loc[missing],
            format=fmt,
            errors="coerce",
        )

        parsed.loc[missing] = candidate

    # ------------------------------------------------------------
    # Row-level fallback
    #
    # Used only for values not understood by the explicit formats.
    # ------------------------------------------------------------

    missing = parsed.isna()

    if missing.any():

        fallback = pd.to_datetime(
            raw.loc[missing],
            errors="coerce",
            dayfirst=(order == "dayfirst"),
        )

        parsed.loc[missing] = fallback

    # ------------------------------------------------------------
    # Reject obviously corrupted years.
    # ------------------------------------------------------------

    years = parsed.dt.year

    invalid_years = (
        years.notna()
        & ~years.between(
            2000,
            2100,
        )
    )

    if invalid_years.any():

        parsed.loc[
            invalid_years
        ] = pd.NaT

    return parsed


# ================================================================
# Duration conversion
# ================================================================

def convert_duration_to_seconds(
    series: pd.Series,
) -> pd.Series:
    """
    Convert flow duration to seconds.

    CIC-IDS-2018 normally stores Flow Duration
    in microseconds.

    Example:

        1,000,000 -> 1 second
        2,000,000 -> 2 seconds
    """

    values = safe_numeric(
        series
    )

    if values.empty:

        return values

    values = values.clip(
        lower=0.0
    )

    nonzero = values[
        values > 0
    ]

    if nonzero.empty:

        return values.astype(
            np.float64
        )

    median_value = float(
        nonzero.median()
    )

    # CIC duration is microseconds.
    if median_value > 1000.0:

        values = (
            values
            / 1_000_000.0
        )

    values = (
        values
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return values.astype(
        np.float64
    )


# ================================================================
# IAT conversion
# ================================================================

def convert_iat_to_seconds(
    series: pd.Series,
) -> pd.Series:
    """
    Convert IAT-like values to seconds.

    CIC-IDS-2018 IAT statistics are normally
    represented in microseconds.
    """

    values = safe_numeric(
        series
    )

    if values.empty:

        return values

    values = values.clip(
        lower=0.0
    )

    nonzero = values[
        values > 0
    ]

    if nonzero.empty:

        return values.astype(
            np.float64
        )

    median_value = float(
        nonzero.median()
    )

    if median_value > 1000.0:

        values = (
            values
            / 1_000_000.0
        )

    values = (
        values
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return values.astype(
        np.float64
    )


# ================================================================
# Column discovery
# ================================================================

def find_canonical_columns(
    df: pd.DataFrame,
):
    """
    Find canonical ARJUN columns in a dataframe.

    Returns:

        {
            "canonical_name": "original column name"
        }
    """

    matches = {}

    for original_name in df.columns:

        normalized = (
            normalize_column_name(
                original_name
            )
        )

        canonical = ALIAS_LOOKUP.get(
            normalized
        )

        if canonical is None:
            continue

        if canonical not in matches:

            matches[
                canonical
            ] = original_name

    return matches


# ================================================================
# Standardization
# ================================================================

def standardize_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw flow/packet columns into
    ARJUN canonical names.

    Missing telemetry is not fabricated.
    """

    if df is None:

        return pd.DataFrame()

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "standardize_columns expects "
            "a pandas DataFrame."
        )

    if df.empty:

        return df.copy()

    result = df.copy()

    matches = find_canonical_columns(
        result
    )

    rename_map = {}

    for canonical, original in matches.items():

        if original != canonical:

            rename_map[
                original
            ] = canonical

    result = result.rename(
        columns=rename_map
    )

    # ------------------------------------------------------------
    # Total bytes
    # ------------------------------------------------------------

    if (
        "bytes" not in result.columns
        and (
            "fwd_bytes" in result.columns
            or "bwd_bytes" in result.columns
        )
    ):

        fwd = safe_numeric(
            result.get(
                "fwd_bytes",
                pd.Series(
                    0.0,
                    index=result.index,
                ),
            )
        ).fillna(0.0)

        bwd = safe_numeric(
            result.get(
                "bwd_bytes",
                pd.Series(
                    0.0,
                    index=result.index,
                ),
            )
        ).fillna(0.0)

        result["bytes"] = (
            fwd + bwd
        )

    # ------------------------------------------------------------
    # Total packets
    # ------------------------------------------------------------

    if (
        "packets" not in result.columns
        and (
            "fwd_packets" in result.columns
            or "bwd_packets" in result.columns
        )
    ):

        fwd = safe_numeric(
            result.get(
                "fwd_packets",
                pd.Series(
                    0.0,
                    index=result.index,
                ),
            )
        ).fillna(0.0)

        bwd = safe_numeric(
            result.get(
                "bwd_packets",
                pd.Series(
                    0.0,
                    index=result.index,
                ),
            )
        ).fillna(0.0)

        result["packets"] = (
            fwd + bwd
        )

    # ------------------------------------------------------------
    # Numeric columns
    # ------------------------------------------------------------

    numeric_columns = [

        "src_port",
        "dst_port",
        "protocol",

        "fwd_packets",
        "bwd_packets",

        "fwd_bytes",
        "bwd_bytes",

        "bytes",
        "packets",

        "packet_length",

        "ttl",
        "tcp_window",
        "payload_size",

        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "psh_count",
        "urg_count",

        "fragment",
        "retransmission",
    ]

    for column in numeric_columns:

        if column not in result.columns:
            continue

        result[column] = (
            safe_numeric(
                result[column]
            )
            .fillna(0.0)
        )

    # ------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------

    if "duration" in result.columns:

        result["duration"] = (
            convert_duration_to_seconds(
                result["duration"]
            )
        )

    # ------------------------------------------------------------
    # IAT
    # ------------------------------------------------------------

    if "iat_seconds" in result.columns:

        result["iat_seconds"] = (
            convert_iat_to_seconds(
                result["iat_seconds"]
            )
        )

    if "src_iat_seconds" in result.columns:

        result["src_iat_seconds"] = (
            convert_iat_to_seconds(
                result["src_iat_seconds"]
            )
        )

    # ------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------

    if "timestamp" in result.columns:

        result["timestamp"] = (
            parse_timestamp(
                result["timestamp"]
            )
        )

    # ------------------------------------------------------------
    # Label
    # ------------------------------------------------------------

    if "label" in result.columns:

        result["label"] = (
            result["label"]
            .astype("string")
            .str.strip()
        )

    return result


# ================================================================
# Derived flow fields
# ================================================================

def add_derived_flow_fields(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add packet and byte totals when absent.
    """

    result = df.copy()

    # ------------------------------------------------------------
    # Packets
    # ------------------------------------------------------------

    if "packets" not in result.columns:

        fwd = safe_numeric(
            result.get(
                "fwd_packets",
                pd.Series(
                    0.0,
                    index=result.index,
                ),
            )
        ).fillna(0.0)

        bwd = safe_numeric(
            result.get(
                "bwd_packets",
                pd.Series(
                    0.0,
                    index=result.index,
                ),
            )
        ).fillna(0.0)

        result["packets"] = (
            fwd + bwd
        )

    # ------------------------------------------------------------
    # Bytes
    # ------------------------------------------------------------

    if "bytes" not in result.columns:

        fwd = safe_numeric(
            result.get(
                "fwd_bytes",
                pd.Series(
                    0.0,
                    index=result.index,
                ),
            )
        ).fillna(0.0)

        bwd = safe_numeric(
            result.get(
                "bwd_bytes",
                pd.Series(
                    0.0,
                    index=result.index,
                ),
            )
        ).fillna(0.0)

        result["bytes"] = (
            fwd + bwd
        )

    return result


# ================================================================
# Complete parser
# ================================================================

def parse_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Complete ARJUN parsing pipeline.

    Pipeline:

        raw dataframe
              |
              v
        column standardization
              |
              v
        derived totals
              |
              v
        numeric cleanup
              |
              v
        duration conversion
              |
              v
        IAT conversion
              |
              v
        timestamp parsing
              |
              v
        final validation
    """

    if (
        df is None
        or df.empty
    ):

        return pd.DataFrame()

    result = standardize_columns(
        df
    )

    result = add_derived_flow_fields(
        result
    )

    # ------------------------------------------------------------
    # Numeric safety
    # ------------------------------------------------------------

    numeric_columns = [

        "duration",

        "bytes",
        "packets",

        "fwd_packets",
        "bwd_packets",

        "fwd_bytes",
        "bwd_bytes",

        "dst_port",
        "src_port",

        "protocol",

        "packet_length",

        "ttl",
        "tcp_window",
        "payload_size",

        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "psh_count",
        "urg_count",

        "fragment",
        "retransmission",

        "iat_seconds",
        "src_iat_seconds",
    ]

    for column in numeric_columns:

        if column not in result.columns:
            continue

        values = safe_numeric(
            result[column]
        )

        if column != "protocol":

            values = values.clip(
                lower=0.0
            )

        result[column] = (
            values
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

    # ------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------

    if "timestamp" in result.columns:

        result["timestamp"] = (
            parse_timestamp(
                result["timestamp"]
            )
        )

    return result


# ================================================================
# Timestamp validation
# ================================================================

def validate_timestamps(
    df: pd.DataFrame,
    expected_max_span_hours: Optional[
        float
    ] = None,
):
    """
    Validate timestamp quality.

    Does not modify the dataframe.
    """

    if (
        df is None
        or df.empty
        or "timestamp" not in df.columns
    ):

        return {
            "valid": False,
            "reason": (
                "No usable timestamp column."
            ),
        }

    timestamps = df[
        "timestamp"
    ]

    valid = timestamps.dropna()

    if valid.empty:

        return {
            "valid": False,
            "reason": (
                "All timestamps are invalid."
            ),
        }

    minimum = valid.min()
    maximum = valid.max()

    span_seconds = (
        maximum - minimum
    ).total_seconds()

    span_hours = (
        span_seconds / 3600.0
    )

    years = valid.dt.year

    valid_year_ratio = float(
        years.between(
            2000,
            2100,
        ).mean()
    )

    result = {

        "valid": True,

        "min_timestamp": str(
            minimum
        ),

        "max_timestamp": str(
            maximum
        ),

        "span_seconds": float(
            span_seconds
        ),

        "span_hours": float(
            span_hours
        ),

        "invalid_count": int(
            timestamps.isna().sum()
        ),

        "valid_year_ratio": (
            valid_year_ratio
        ),
    }

    if valid_year_ratio < 0.95:

        result["valid"] = False

        result["reason"] = (
            "More than 5% of timestamps "
            "are outside the expected "
            "2000-2100 range."
        )

        return result

    if (
        expected_max_span_hours is not None
        and span_hours
        > expected_max_span_hours
    ):

        result["valid"] = False

        result["reason"] = (
            "Timestamp span exceeds "
            "expected limit."
        )

    return result


# ================================================================
# Standalone self-test
# ================================================================

def _self_test():

    # ------------------------------------------------------------
    # MM/DD/YYYY test
    # ------------------------------------------------------------

    raw = pd.DataFrame(
        {
            "Dst Port": [
                443,
                80,
            ],

            "Protocol": [
                6,
                6,
            ],

            "Timestamp": [
                "02/14/2018 10:00:00",
                "02/14/2018 10:00:02",
            ],

            "Flow Duration": [
                1_000_000,
                2_000_000,
            ],

            "Tot Fwd Pkts": [
                10,
                5,
            ],

            "Tot Bwd Pkts": [
                8,
                3,
            ],

            "TotLen Fwd Pkts": [
                5000,
                2000,
            ],

            "TotLen Bwd Pkts": [
                4000,
                1000,
            ],

            "Pkt Size Avg": [
                500,
                300,
            ],

            "Init Fwd Win Byts": [
                64240,
                29200,
            ],

            "SYN Flag Cnt": [
                1,
                0,
            ],

            "ACK Flag Cnt": [
                8,
                5,
            ],

            "Label": [
                "Benign",
                "Attack",
            ],
        }
    )

    parsed = parse_dataframe(
        raw
    )

    # ------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------

    required = [

        "dst_port",
        "protocol",
        "timestamp",

        "duration",

        "fwd_packets",
        "bwd_packets",

        "fwd_bytes",
        "bwd_bytes",

        "packets",
        "bytes",

        "packet_length",

        "tcp_window",

        "syn_count",
        "ack_count",

        "label",
    ]

    missing = [
        column
        for column in required
        if column not in parsed.columns
    ]

    if missing:

        raise AssertionError(
            "Parser test missing columns: "
            f"{missing}"
        )

    # ------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------

    expected_timestamp = pd.Timestamp(
        "2018-02-14 10:00:00"
    )

    if (
        parsed.loc[
            0,
            "timestamp"
        ]
        != expected_timestamp
    ):

        raise AssertionError(
            "MM/DD/YYYY timestamp "
            "parsing failed."
        )

    # ------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------

    if not np.isclose(
        parsed.loc[
            0,
            "duration"
        ],
        1.0,
    ):

        raise AssertionError(
            "Duration conversion failed."
        )

    if not np.isclose(
        parsed.loc[
            1,
            "duration"
        ],
        2.0,
    ):

        raise AssertionError(
            "Duration conversion failed."
        )

    # ------------------------------------------------------------
    # Packet total
    # ------------------------------------------------------------

    if (
        parsed.loc[
            0,
            "packets"
        ]
        != 18
    ):

        raise AssertionError(
            "Packet total calculation failed."
        )

    # ------------------------------------------------------------
    # Byte total
    # ------------------------------------------------------------

    if (
        parsed.loc[
            0,
            "bytes"
        ]
        != 9000
    ):

        raise AssertionError(
            "Byte total calculation failed."
        )

    # ------------------------------------------------------------
    # DD/MM/YYYY test
    # ------------------------------------------------------------

    day_first_test = pd.DataFrame(
        {
            "Timestamp": [
                "23/02/2018 10:00:00",
                "23/02/2018 10:00:05",
            ]
        }
    )

    parsed_day_first = (
        parse_dataframe(
            day_first_test
        )
    )

    expected_day_first = pd.Timestamp(
        "2018-02-23 10:00:00"
    )

    if (
        parsed_day_first.loc[
            0,
            "timestamp"
        ]
        != expected_day_first
    ):

        raise AssertionError(
            "DD/MM/YYYY timestamp "
            "parsing failed."
        )

    # ------------------------------------------------------------
    # Fractional seconds test
    # ------------------------------------------------------------

    fractional_test = pd.DataFrame(
        {
            "Timestamp": [
                "02/14/2018 10:00:00.123",
            ]
        }
    )

    parsed_fractional = (
        parse_dataframe(
            fractional_test
        )
    )

    if (
        parsed_fractional.loc[
            0,
            "timestamp"
        ]
        != pd.Timestamp(
            "2018-02-14 10:00:00.123"
        )
    ):

        raise AssertionError(
            "Fractional timestamp "
            "parsing failed."
        )

    # ------------------------------------------------------------
    # Timestamp validation
    # ------------------------------------------------------------

    validation = validate_timestamps(
        parsed,
        expected_max_span_hours=1,
    )

    if not validation["valid"]:

        raise AssertionError(
            "Timestamp validation failed: "
            f"{validation}"
        )

    # ------------------------------------------------------------
    # Negative duration
    # ------------------------------------------------------------

    negative_test = raw.copy()

    negative_test.loc[
        0,
        "Flow Duration"
    ] = -500000

    parsed_negative = (
        parse_dataframe(
            negative_test
        )
    )

    if (
        parsed_negative.loc[
            0,
            "duration"
        ]
        != 0.0
    ):

        raise AssertionError(
            "Negative duration was "
            "not clamped."
        )

    # ------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------

    print("=" * 70)
    print(
        "ARJUN PARSER TEST"
    )
    print("=" * 70)

    print(
        parsed[
            [
                "dst_port",
                "protocol",
                "timestamp",
                "duration",
                "packets",
                "bytes",
                "packet_length",
                "syn_count",
                "ack_count",
                "tcp_window",
            ]
        ]
    )

    print()
    print(
        "Timestamp validation:"
    )

    print(
        validation
    )

    print()
    print(
        "PARSER TEST: PASSED"
    )


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":

    _self_test()