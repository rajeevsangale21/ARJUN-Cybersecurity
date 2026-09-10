from pathlib import Path
from typing import Callable, Iterator, Optional, Union

import pandas as pd


DEFAULT_CHUNK_SIZE = 100_000


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean CSV column names without changing their meaning.
    """

    df = df.copy()

    cleaned = []

    for column in df.columns:

        name = str(column)

        name = (
            name.replace("\ufeff", "")
            .replace("\n", " ")
            .replace("\r", " ")
            .strip()
        )

        name = " ".join(name.split())

        cleaned.append(name)

    df.columns = cleaned

    return df


def _remove_repeated_headers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove rows accidentally containing CSV header
    values as actual data.

    Example:

        Label

    appearing inside the Label column is a repeated
    header row, not an attack class.
    """

    if df.empty:
        return df

    df = df.copy()

    if "Label" in df.columns:

        mask = (
            df["Label"]
            .astype(str)
            .str.strip()
            .str.lower()
            != "label"
        )

        df = df.loc[mask]

    return df


def _normalize_label(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize the Label column.

    Original attack names are preserved.
    """

    if "Label" not in df.columns:
        return df

    df = df.copy()

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    return df


def prepare_chunk(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply lightweight cleaning to one CSV chunk.

    Heavy feature engineering is intentionally NOT done here.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = _clean_column_names(df)

    df = _remove_repeated_headers(df)

    df = _normalize_label(df)

    return df.reset_index(drop=True)


def iter_csv(
    path: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usecols: Optional[list] = None,
    nrows: Optional[int] = None,
    dtype=None,
    encoding: str = "utf-8",
) -> Iterator[pd.DataFrame]:
    """
    Stream a CSV file chunk-by-chunk.

    This is the preferred method for ARJUN's large
    CSE-CIC-IDS2018 dataset.

    Parameters
    ----------
    path:
        CSV file path.

    chunk_size:
        Number of rows loaded into memory at a time.

    usecols:
        Optional columns to read.

    nrows:
        Optional maximum number of rows.

    dtype:
        Optional pandas dtype configuration.

    encoding:
        CSV encoding.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"CSV path is not a file: {path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, got: {path}"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    reader = pd.read_csv(
        path,
        chunksize=chunk_size,
        usecols=usecols,
        nrows=nrows,
        dtype=dtype,
        encoding=encoding,
        low_memory=True,
    )

    for chunk in reader:

        chunk = prepare_chunk(chunk)

        if not chunk.empty:
            yield chunk


def load_csv(
    path: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    nrows: Optional[int] = None,
    usecols: Optional[list] = None,
    dtype=None,
) -> pd.DataFrame:
    """
    Load a CSV into a DataFrame.

    IMPORTANT:
    For very large files, prefer iter_csv().

    This function exists for compatibility with the
    existing ARJUN pipeline and for smaller CSV files.
    """

    chunks = []

    for chunk in iter_csv(
        path=path,
        chunk_size=chunk_size,
        nrows=nrows,
        usecols=usecols,
        dtype=dtype,
    ):

        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    return pd.concat(
        chunks,
        ignore_index=True,
    )


def load_csv_sample(
    path: Union[str, Path],
    rows: int = 10_000,
) -> pd.DataFrame:
    """
    Load a limited sample from a CSV.

    Useful for testing and development.
    """

    if rows <= 0:
        raise ValueError(
            "rows must be greater than zero."
        )

    return load_csv(
        path,
        chunk_size=min(
            rows,
            DEFAULT_CHUNK_SIZE,
        ),
        nrows=rows,
    )


def process_csv_chunks(
    path: Union[str, Path],
    processor: Callable[[pd.DataFrame], object],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    usecols: Optional[list] = None,
    dtype=None,
):
    """
    Apply a processing function to every CSV chunk.

    The processor receives one DataFrame at a time.

    This avoids requiring the entire dataset to fit
    into memory.
    """

    results = []

    for chunk in iter_csv(
        path=path,
        chunk_size=chunk_size,
        usecols=usecols,
        dtype=dtype,
    ):

        result = processor(chunk)

        if result is not None:
            results.append(result)

    return results


def discover_csv_files(
    directory: Union[str, Path],
) -> list[Path]:
    """
    Recursively find CSV files in a directory.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    files = sorted(
        directory.rglob("*.csv")
    )

    return files