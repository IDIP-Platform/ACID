"""Per-row processing results shared by the pipeline stages."""

import logging

import pandas as pd
from tqdm.auto import tqdm

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def make_success_result(
    row_index, input_file, output_file, metadata_values: dict, **extra_fields
) -> dict:
    """Build the result record of a successfully processed metadata row.

    Args:
        row_index: Index of the processed row in the metadata dataframe.
        input_file: Name of the file that was processed.
        output_file: Name of the file that was written.
        metadata_values: Stage-specific metadata columns and their values.
        **extra_fields: Additional identifiers, e.g. ``segmentation_file``.
    """
    return {
        "row_index": row_index,
        "input_file": input_file,
        **extra_fields,
        "output_file": output_file,
        "success": True,
        "stage": None,
        "error_type": None,
        "error_message": None,
        **metadata_values,
    }


def make_failure_result(
    row_index,
    input_file,
    error: Exception,
    metadata_columns,
    null_value,
    stage: str | None = None,
    **extra_fields,
) -> dict:
    """Build the result record of a metadata row whose processing failed.

    All ``metadata_columns`` are set to ``null_value``.
    """
    return {
        "row_index": row_index,
        "input_file": input_file,
        **extra_fields,
        "output_file": None,
        "success": False,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        **dict.fromkeys(metadata_columns, null_value),
    }


def update_metadata_with_results(
    metadata_df: pd.DataFrame, results: list[dict], metadata_columns, copy_dataframe=True
) -> pd.DataFrame:
    """Write the metadata columns of per-row results back into the metadata.

    Raises:
        KeyError: If a metadata column is missing from the results.
    """
    metadata_columns = list(metadata_columns)

    if copy_dataframe:
        metadata_df = metadata_df.copy()

    if not results:
        return metadata_df

    results_df = pd.DataFrame.from_records(results).set_index("row_index")

    missing_result_columns = [
        column for column in metadata_columns if column not in results_df.columns
    ]
    if missing_result_columns:
        raise KeyError(f"Results are missing metadata columns: {missing_result_columns}")

    missing_metadata_columns = [
        column for column in metadata_columns if column not in metadata_df.columns
    ]
    metadata_df = metadata_df.assign(
        **{column: pd.NA for column in missing_metadata_columns}
    )
    metadata_df = metadata_df.astype(dict.fromkeys(metadata_columns, "object"))
    metadata_df.loc[results_df.index, metadata_columns] = results_df[
        metadata_columns
    ].to_numpy()

    return metadata_df


def process_rows(metadata_df: pd.DataFrame, process_row, description: str, max_rows=None):
    """Call ``process_row(row_index, metadata_row)`` for each row and collect results.

    Args:
        metadata_df: Rows to process.
        process_row: Callable returning one result dict per row.
        description: Progress-bar label.
        max_rows: Process only the first ``max_rows`` rows when given.
    """
    row_indices = metadata_df.index
    if max_rows is not None:
        row_indices = row_indices[:max_rows]

    return [
        process_row(row_index, metadata_df.loc[row_index])
        for row_index in tqdm(row_indices, desc=description)
    ]
