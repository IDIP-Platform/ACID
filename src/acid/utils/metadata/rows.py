import logging

import pandas as pd

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def get_required_filename(metadata_row: pd.Series, column_name: str) -> str:
    """Return a required filename stored in one metadata row.

    Args:
        metadata_row: One row of a metadata dataframe.
        column_name: Column holding the filename.

    Returns:
        The filename without surrounding whitespace.

    Raises:
        ValueError: If the column is missing or the cell is empty.
    """
    filename = metadata_row.get(column_name)

    if pd.isna(filename) or str(filename).strip() == "":
        raise ValueError(f"Missing filename in column {column_name!r}")

    return str(filename).strip()
