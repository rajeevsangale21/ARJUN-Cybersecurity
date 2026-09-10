"""
ARJUN - Unseen Attack Generalization Preparation

Purpose
-------
Creates a leakage-safe attack-family label for every ARJUN 5-second state
window using the ORIGINAL CIC-IDS-2018 CSV Label column, then aligns those
labels to the already-built ARJUN state/sequence datasets.

This is the first step of a rigorous unseen-attack experiment:

    training attacks != held-out attack family

The script deliberately does NOT claim unseen-attack generalization yet.
It prepares and audits the split so a fresh World Model can be trained on
non-held-out families and evaluated on the held-out family.

Default held-out family:
    Infilteration

The CIC dataset itself spells this label "Infilteration".

Run from ARJUN project root:
    python evaluation\\unseen_attack_generalization.py

Optional:
    python evaluation\\unseen_attack_generalization.py --held-out "SQL Injection"
    python evaluation\\unseen_attack_generalization.py --dataset-dir data\\raw\\dataset
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import re
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = ROOT / "data" / "raw" / "dataset"
PROCESSED_DIR = ROOT / "data" / "processed"
STATE_FILE = PROCESSED_DIR / "training_states.npz"
SEQUENCE_FILE = PROCESSED_DIR / "graph_sequences_states.npz"
OUTPUT_FILE = PROCESSED_DIR / "unseen_attack_state_labels.npz"
REPORT_FILE = ROOT / "evaluation" / "unseen_attack_split.json"

WINDOW_SECONDS = 5
CHUNK_SIZE = 200_000
DEFAULT_HELD_OUT = "Infilteration"

BENIGN_NAMES = {
    "",
    "benign",
    "normal",
    "0",
    "0.0",
    "nan",
    "none",
}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def clean_label(value) -> str:
    if pd.isna(value):
        return "Benign"
    text = str(value).strip()
    if not text or text.lower() in BENIGN_NAMES:
        return "Benign"
    return text


def is_benign(value: str) -> bool:
    return clean_label(value).lower() == "benign"


def normalize_family(value: str) -> str:
    """Normalize only whitespace/case for matching, not the displayed label."""
    return " ".join(str(value).strip().split()).casefold()


# ---------------------------------------------------------------------------
# CIC timestamp handling
# ---------------------------------------------------------------------------

def parse_cic_timestamp(series: pd.Series, source_name: str) -> pd.Series:
    """
    Robustly parse CIC-IDS-2018 timestamps.

    CIC files in this copy are not consistent enough for one global
    day-first/month-first setting.  Some rows use DD/MM/YYYY while rows such
    as 02/14/2018 are necessarily MM/DD/YYYY.

    Strategy:
      1. Treat the source values as strings so numeric-looking values are
         never silently interpreted as Unix nanoseconds.
      2. Parse common CIC timestamp layouts explicitly.
      3. Use the date components to resolve slash ambiguity:
           first component > 12 -> DD/MM
           second component > 12 -> MM/DD
      4. Retry remaining values with pandas' mixed-format parser.
      5. Reject implausible years rather than allowing 1970-era corruption.
    """
    raw = series.astype("string").str.strip()

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    # Explicit formats first.  CIC commonly has seconds, optional fractional
    # seconds, and occasionally AM/PM.
    formats = [
        "%d/%m/%Y %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M:%S %p",
    ]

    remaining = raw.notna() & raw.ne("")
    for fmt in formats:
        if not bool(remaining.any()):
            break
        parsed = pd.to_datetime(raw.loc[remaining], format=fmt, errors="coerce")
        good = parsed.notna()
        if bool(good.any()):
            idx = parsed.index[good]
            result.loc[idx] = parsed.loc[idx]
            remaining.loc[idx] = False

    # Component-aware parsing for slash dates that did not match the explicit
    # formats. This handles both DD/MM/YYYY and MM/DD/YYYY without a global
    # dayfirst assumption.
    if bool(remaining.any()):
        candidates = raw.loc[remaining]

        def parse_one(value):
            if value is None or pd.isna(value):
                return pd.NaT
            s = str(value).strip()
            if not s:
                return pd.NaT

            # Numeric epoch values are accepted only when they are clearly
            # seconds/milliseconds since 2000, preventing accidental 1970 data.
            if re.fullmatch(r"\d+(?:\.\d+)?", s):
                try:
                    number = float(s)
                    if number >= 946684800:
                        unit = "ms" if number > 1e11 else "s"
                        return pd.to_datetime(number, unit=unit, errors="coerce")
                except Exception:
                    pass

            match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})(.*)$", s)
            if match:
                first = int(match.group(1))
                second = int(match.group(2))
                if first > 12 and second <= 12:
                    dayfirst = True
                elif second > 12 and first <= 12:
                    dayfirst = False
                else:
                    # Ambiguous slash dates in the source are resolved by
                    # the actual calendar validity.  If both interpretations
                    # are valid, prefer the conventional MM/DD form used by
                    # the 02-14 file; explicit day-first rows are recovered
                    # by the first>12 rule above.
                    dayfirst = False

                try:
                    return pd.to_datetime(s, dayfirst=dayfirst, errors="coerce")
                except Exception:
                    return pd.NaT

            try:
                return pd.to_datetime(s, errors="coerce")
            except Exception:
                return pd.NaT

        parsed = candidates.map(parse_one)
        good = parsed.notna()
        if bool(good.any()):
            idx = parsed.index[good]
            result.loc[idx] = parsed.loc[idx]

    # Final mixed-format fallback for unusual but legitimate timestamp text.
    remaining = result.isna() & raw.notna() & raw.ne("")
    if bool(remaining.any()):
        parsed = pd.to_datetime(
            raw.loc[remaining],
            format="mixed",
            errors="coerce",
            dayfirst=False,
        )
        good = parsed.notna()
        if bool(good.any()):
            idx = parsed.index[good]
            result.loc[idx] = parsed.loc[idx]

    # Strictly reject corrupted dates outside a reasonable historical range.
    invalid_year = result.notna() & (
        (result.dt.year < 2000) | (result.dt.year > 2100)
    )
    result.loc[invalid_year] = pd.NaT

    return result


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_csvs(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"CIC dataset directory does not exist:\n{dataset_dir}"
        )

    files = sorted(dataset_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in:\n{dataset_dir}"
        )

    return files


# ---------------------------------------------------------------------------
# Load ARJUN state metadata
# ---------------------------------------------------------------------------

def load_state_metadata() -> Dict[str, np.ndarray]:
    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"ARJUN state dataset not found:\n{STATE_FILE}"
        )

    with np.load(STATE_FILE, allow_pickle=True) as data:
        required = [
            "states",
            "labels",
            "source_files",
            "local_window_ids",
            "window_starts",
            "window_ends",
        ]

        missing = [key for key in required if key not in data]
        if missing:
            raise RuntimeError(
                "training_states.npz is missing required metadata: "
                + ", ".join(missing)
            )

        result = {
            "states": np.asarray(data["states"], dtype=np.float32),
            "labels": np.asarray(data["labels"], dtype=np.int64),
            "source_files": np.asarray(data["source_files"], dtype=str),
            "local_window_ids": np.asarray(data["local_window_ids"], dtype=np.int64),
            "window_starts": np.asarray(data["window_starts"], dtype=str),
            "window_ends": np.asarray(data["window_ends"], dtype=str),
        }

    n = len(result["states"])
    for key, value in result.items():
        if len(value) != n:
            raise RuntimeError(
                f"State metadata length mismatch for {key}: {len(value)} != {n}"
            )

    return result


# ---------------------------------------------------------------------------
# Raw label -> 5-second window map
# ---------------------------------------------------------------------------

def scan_file_labels(csv_path: Path) -> Tuple[Dict[int, Counter], Dict[str, int]]:
    """
    Scan only Timestamp and Label columns.

    Memory stays bounded because the 6.5 GB CIC dataset is never loaded as a
    complete DataFrame.
    """
    print()
    print(f"Scanning labels: {csv_path.name}")

    header = pd.read_csv(csv_path, nrows=0)
    columns = list(header.columns)

    timestamp_column = next(
        (c for c in columns if str(c).strip().casefold() == "timestamp"),
        None,
    )
    label_column = next(
        (c for c in columns if str(c).strip().casefold() == "label"),
        None,
    )

    if timestamp_column is None:
        raise RuntimeError(f"Timestamp column not found in {csv_path.name}")
    if label_column is None:
        raise RuntimeError(f"Label column not found in {csv_path.name}")

    window_labels: Dict[int, Counter] = defaultdict(Counter)
    label_totals: Counter = Counter()
    total_rows = 0
    valid_rows = 0
    file_min = None

    reader = pd.read_csv(
        csv_path,
        usecols=[timestamp_column, label_column],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        timestamps = parse_cic_timestamp(chunk[timestamp_column], csv_path.name)
        labels = chunk[label_column].map(clean_label)

        valid = timestamps.notna()
        if not bool(valid.any()):
            total_rows += len(chunk)
            continue

        valid_ts = timestamps.loc[valid]
        if file_min is None:
            file_min = valid_ts.min()
        else:
            candidate = valid_ts.min()
            if candidate < file_min:
                file_min = candidate

        total_rows += len(chunk)
        valid_rows += int(valid.sum())

        # We cannot finalize window IDs until the true file minimum is known.
        # Keep only the timestamp + label pairs for this chunk in a temporary
        # list; each chunk is bounded. A second pass below computes exact IDs.
        print(
            f"  chunk {chunk_number:03d}: rows={total_rows:,}",
            end="\r",
            flush=True,
        )

    print()

    if file_min is None:
        raise RuntimeError(f"No valid timestamps found in {csv_path.name}")

    # Exact second pass. This still reads only two columns and keeps the
    # accumulated structure bounded by number of 5-second windows.
    window_labels.clear()
    total_rows = 0
    valid_rows = 0

    reader = pd.read_csv(
        csv_path,
        usecols=[timestamp_column, label_column],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        timestamps = parse_cic_timestamp(chunk[timestamp_column], csv_path.name)
        labels = chunk[label_column].map(clean_label)
        valid = timestamps.notna()

        if bool(valid.any()):
            elapsed = (timestamps.loc[valid] - file_min).dt.total_seconds()
            window_ids = np.floor(elapsed / WINDOW_SECONDS).astype(np.int64)

            valid_labels = labels.loc[valid]
            for window_id, label in zip(window_ids.to_numpy(), valid_labels.to_numpy()):
                label = clean_label(label)
                window_labels[int(window_id)][label] += 1
                label_totals[label] += 1

            valid_rows += int(valid.sum())

        total_rows += len(chunk)
        print(
            f"  map chunk {chunk_number:03d}: rows={total_rows:,}",
            end="\r",
            flush=True,
        )

    print()
    print(f"  File minimum time : {file_min}")
    print(f"  Raw rows          : {total_rows:,}")
    print(f"  Valid timestamps  : {valid_rows:,}")
    print(f"  Windows observed  : {len(window_labels):,}")

    return dict(window_labels), dict(label_totals)


# ---------------------------------------------------------------------------
# State-family alignment
# ---------------------------------------------------------------------------

def choose_window_family(counter: Counter) -> str:
    """
    Assign one family to a state window.

    Rules:
      - benign-only -> Benign
      - attack-only -> most frequent attack family
      - mixed benign/attack -> most frequent attack family
      - ties are deterministic alphabetically
    """
    attack_counts = [
        (label, count)
        for label, count in counter.items()
        if not is_benign(label)
    ]

    if not attack_counts:
        return "Benign"

    attack_counts.sort(key=lambda item: (-item[1], normalize_family(item[0])))
    return attack_counts[0][0]


def build_state_families(
    state_meta: Dict[str, np.ndarray],
    dataset_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    source_files = state_meta["source_files"]
    local_window_ids = state_meta["local_window_ids"]

    unique_sources = []
    seen = set()
    for source in source_files:
        source = str(source)
        if source not in seen:
            seen.add(source)
            unique_sources.append(source)

    state_families = np.full(
        len(source_files),
        "UNMAPPED",
        dtype=object,
    )

    audit: Dict[str, Dict[str, int]] = {}

    for source_name in unique_sources:
        csv_path = dataset_dir / source_name
        if not csv_path.exists():
            raise FileNotFoundError(
                f"State dataset references missing raw file:\n{csv_path}"
            )

        window_labels, raw_counts = scan_file_labels(csv_path)

        source_mask = source_files == source_name
        state_indices = np.flatnonzero(source_mask)

        mapped = 0
        unmapped = 0
        families = Counter()

        for state_index in state_indices:
            local_id = int(local_window_ids[state_index])
            counter = window_labels.get(local_id)

            if not counter:
                unmapped += 1
                continue

            family = choose_window_family(Counter(counter))
            state_families[state_index] = family
            families[family] += 1
            mapped += 1

        audit[source_name] = {
            "raw_rows": int(sum(raw_counts.values())),
            "states": int(len(state_indices)),
            "mapped_states": int(mapped),
            "unmapped_states": int(unmapped),
            "attack_states": int(sum(v for k, v in families.items() if not is_benign(k))),
        }

        print()
        print(
            f"{source_name}: mapped {mapped:,}/{len(state_indices):,} states"
        )

    return state_families, audit


# ---------------------------------------------------------------------------
# Leakage-safe sequence split
# ---------------------------------------------------------------------------

def build_sequence_split(
    state_families: np.ndarray,
    held_out: str,
    sequence_length: int,
) -> Dict[str, np.ndarray]:
    if not SEQUENCE_FILE.exists():
        raise FileNotFoundError(
            f"Sequence dataset not found:\n{SEQUENCE_FILE}"
        )

    with np.load(SEQUENCE_FILE, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(
            data["targets"] if "targets" in data else data["y"],
            dtype=np.float32,
        )

    if len(state_families) < sequence_length + 1:
        raise RuntimeError("Not enough state windows for sequence construction")

    expected_sequences = len(state_families) - sequence_length
    if len(X) != expected_sequences:
        raise RuntimeError(
            "Sequence/state alignment mismatch: "
            f"X has {len(X)} sequences but expected {expected_sequences} "
            f"from {len(state_families)} states and sequence length {sequence_length}."
        )

    held_norm = normalize_family(held_out)
    target_families = state_families[sequence_length:]
    history_families = np.asarray(
        [
            state_families[i : i + sequence_length]
            for i in range(expected_sequences)
        ],
        dtype=object,
    )

    target_is_held = np.asarray(
        [normalize_family(x) == held_norm for x in target_families],
        dtype=bool,
    )

    # Training sequences may not contain the held-out family anywhere in the
    # 10-state history or in the target. This prevents temporal leakage.
    history_has_held = np.asarray(
        [
            any(normalize_family(x) == held_norm for x in row)
            for row in history_families
        ],
        dtype=bool,
    )

    train_mask = (~target_is_held) & (~history_has_held)
    test_mask = target_is_held & (~history_has_held)
    excluded_context_mask = target_is_held & history_has_held

    return {
        "train_mask": train_mask,
        "test_mask": test_mask,
        "excluded_context_mask": excluded_context_mask,
        "target_families": target_families,
        "history_families": history_families,
        "X": X,
        "y": y,
    }


# ---------------------------------------------------------------------------
# Save and report
# ---------------------------------------------------------------------------

def save_outputs(
    state_meta: Dict[str, np.ndarray],
    state_families: np.ndarray,
    split: Dict[str, np.ndarray],
    held_out: str,
    audit: Dict[str, Dict[str, int]],
    sequence_length: int,
) -> None:
    np.savez_compressed(
        OUTPUT_FILE,
        state_families=np.asarray(state_families, dtype=str),
        source_files=state_meta["source_files"],
        local_window_ids=state_meta["local_window_ids"],
        window_starts=state_meta["window_starts"],
        window_ends=state_meta["window_ends"],
        train_mask=split["train_mask"],
        test_mask=split["test_mask"],
        excluded_context_mask=split["excluded_context_mask"],
        target_families=np.asarray(split["target_families"], dtype=str),
    )

    family_counts = Counter(str(x) for x in state_families)
    unknown = family_counts.pop("UNMAPPED", 0)

    report = {
        "experiment": "ARJUN unseen attack family generalization",
        "status": "SPLIT_PREPARED",
        "held_out_attack": held_out,
        "sequence_length": int(sequence_length),
        "window_seconds": WINDOW_SECONDS,
        "state_count": int(len(state_families)),
        "mapped_states": int(len(state_families) - unknown),
        "unmapped_states": int(unknown),
        "state_family_distribution": dict(sorted(family_counts.items())),
        "sequence_count": int(len(split["X"])),
        "train_sequences": int(split["train_mask"].sum()),
        "held_out_test_sequences": int(split["test_mask"].sum()),
        "excluded_held_out_context_sequences": int(split["excluded_context_mask"].sum()),
        "audit_by_source": audit,
        "leakage_rule": (
            "A training sequence is excluded if the held-out attack family "
            "appears in any of its 10 historical states or its target state."
        ),
        "next_step": (
            "Train a fresh World Model only on train_sequences and evaluate "
            "next-state prediction on held_out_test_sequences."
        ),
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 72)
    print("ARJUN UNSEEN-ATTACK SPLIT PREPARATION")
    print("=" * 72)
    print(f"Held-out attack family      : {held_out}")
    print(f"State windows               : {len(state_families):,}")
    print(f"Mapped state windows        : {len(state_families) - unknown:,}")
    print(f"Unmapped state windows      : {unknown:,}")
    print(f"Sequences                   : {len(split['X']):,}")
    print(f"Leakage-safe train sequences: {int(split['train_mask'].sum()):,}")
    print(f"Held-out test sequences     : {int(split['test_mask'].sum()):,}")
    print(
        "Excluded held-out-context  : "
        f"{int(split['excluded_context_mask'].sum()):,}"
    )
    print()
    print("STATE FAMILY DISTRIBUTION")
    print("-" * 72)
    for family, count in sorted(family_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{family:<35} {count:>8,}")
    print()
    print(f"Saved mapping : {OUTPUT_FILE}")
    print(f"Saved report  : {REPORT_FILE}")

    if unknown:
        print()
        print("WARNING: Some state windows could not be mapped to raw labels.")
        print("Do not train the final unseen-attack experiment until this is 0.")
    elif int(split["test_mask"].sum()) == 0:
        print()
        print("WARNING: No leakage-safe held-out test sequences were found.")
        print("Choose a different attack family with sufficient isolated windows.")
    else:
        print()
        print("UNSEEN-ATTACK SPLIT PREPARATION: PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a leakage-safe ARJUN unseen-attack family split."
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Directory containing CIC-IDS-2018 CSV files.",
    )
    parser.add_argument(
        "--held-out",
        default=DEFAULT_HELD_OUT,
        help="Attack label/family to hold out from World Model training.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Historical state count used by the current ARJUN sequence dataset.",
    )
    args = parser.parse_args()

    if args.sequence_length < 1:
        raise ValueError("--sequence-length must be >= 1")

    dataset_dir = Path(args.dataset_dir)
    csv_files = discover_csvs(dataset_dir)

    print("=" * 72)
    print("ARJUN - UNSEEN ATTACK GENERALIZATION PREPARATION")
    print("=" * 72)
    print(f"Dataset directory : {dataset_dir}")
    print(f"CSV files         : {len(csv_files)}")
    print(f"Held-out family   : {args.held_out}")
    print(f"Window size       : {WINDOW_SECONDS}s")
    print(f"Sequence length   : {args.sequence_length}")

    state_meta = load_state_metadata()
    print(f"Existing states   : {len(state_meta['states']):,}")
    print(f"State dimension   : {state_meta['states'].shape[1]}")

    state_families, audit = build_state_families(
        state_meta,
        dataset_dir,
    )

    split = build_sequence_split(
        state_families,
        args.held_out,
        args.sequence_length,
    )

    save_outputs(
        state_meta,
        state_families,
        split,
        args.held_out,
        audit,
        args.sequence_length,
    )


if __name__ == "__main__":
    main()
