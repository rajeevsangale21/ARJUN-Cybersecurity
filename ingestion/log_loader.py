from pathlib import Path
import pandas as pd


def load_generic_log(path):
    """
    Load a generic CSV/TSV-style security log.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    separators = [
        ",",
        "\t",
        r"\s+"
    ]

    for separator in separators:

        try:

            df = pd.read_csv(
                path,
                sep=separator,
                engine="python",
                comment="#"
            )

            if not df.empty and df.shape[1] > 1:

                return df

        except Exception:
            pass

    raise ValueError(
        f"Could not parse log file: {path}"
    )