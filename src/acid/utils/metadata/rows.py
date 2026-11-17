"""Read values from single rows of a metadata dataframe."""

import logging

import pandas as pd

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def get_required_filename(metadata_row: pd.Series, column_name: str) -> str:
    """Return a file name that must be present in one metadata row.

    Each stage reads its input file name from a different column, e.g.
    `fov_column_name` for extracted fields of view or
    `illum_correct_df_file_name_clm_name` for background-corrected ones; the
    caller passes the column name taken from its stage configuration.

    Args:
        metadata_row (pd.Series): One row of a metadata dataframe.
        column_name (str): Name of the column holding the file name.

    Returns:
        str: The file name with surrounding whitespace removed.

    Raises:
        ValueError: If the column does not exist, or the cell is `NaN`, `None`
            or contains only whitespace.
    """
    filename = metadata_row.get(column_name)

    if pd.isna(filename) or str(filename).strip() == "":
        raise ValueError(f"Missing filename in column {column_name!r}")

    return str(filename).strip()
