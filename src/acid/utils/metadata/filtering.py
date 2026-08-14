import logging

import pandas as pd

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------

def filter_metadata_by_splits(
    metadata_df: pd.DataFrame,
    selected_splits: str | list[str] = "train",
    split_config: dict | None = None,
) -> pd.DataFrame:
    """Filter metadata rows by one or more dataset splits.

    Args:
        metadata_df: Metadata dataframe to filter.
        selected_splits: Dataset split name or split names to keep. For example,
            "train" or ["train", "validation"].
        split_config: Dataset split configuration dictionary. Expected keys are
            "column" and "labels". The "column" value identifies the metadata
            column containing split labels, and "labels" maps split names to the
            values stored in that column.

    Returns:
        A copy of metadata_df containing only rows that belong to the selected
        dataset splits.

    Raises:
        ValueError: If split_config is None.
        KeyError: If split_config is missing required keys or if a selected split
            is not defined in split_config["labels"].
    """
    if split_config is None:
        raise ValueError("split_config must be provided.")

    if isinstance(selected_splits, str):
        selected_splits = [selected_splits]

    split_column = split_config["column"]
    split_labels = split_config["labels"]

    selected_values = [
        split_labels[split_name]
        for split_name in selected_splits
    ]

    return metadata_df[
        metadata_df[split_column].isin(selected_values)
    ].copy()

# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------
