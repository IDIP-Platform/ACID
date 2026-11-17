"""Per-row processing results shared by the pipeline stages.

Stages that process one field of view per metadata row (background
correction, segmentation, feature extraction) return one result record per
row instead of raising on the first bad file. Every record has the same common
keys:

- `row_index`: index of the row in the metadata dataframe,
- `input_file`: file name that was processed (`None` if it could not be read
  from the row),
- `output_file`: file name that was written (`None` on failure),
- `success`: `True` or `False`,
- `stage`: name of the step that failed (`None` on success),
- `error_type`, `error_message`: exception class name and message on failure,

followed by the stage-specific metadata columns that
`update_metadata_with_results` writes back into the metadata dataframe.
"""

import logging
from collections.abc import Callable, Hashable, Iterable
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def make_success_result(
    row_index: Hashable,
    input_file: str,
    output_file: str,
    metadata_values: dict[str, Any],
    **extra_fields,
) -> dict:
    """Build the result record of a successfully processed metadata row.

    Args:
        row_index (Hashable): Index of the processed row in the metadata
            dataframe.
        input_file (str): File name that was processed.
        output_file (str): File name that was written.
        metadata_values (dict[str, Any]): Stage-specific metadata column names
            mapped to the values recorded for this row, e.g. processing date,
            output file name and processing parameters.
        **extra_fields: Additional identifiers placed right after `input_file`,
            e.g. `segmentation_file` in feature extraction.

    Returns:
        dict: Result record with the common keys described in the module
        docstring, `success=True`, followed by `metadata_values`.
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
    row_index: Hashable,
    input_file: str | None,
    error: Exception,
    metadata_columns: Iterable[str],
    null_value: Any,
    stage: str | None = None,
    **extra_fields,
) -> dict:
    """Build the result record of a metadata row whose processing failed.

    Args:
        row_index (Hashable): Index of the row in the metadata dataframe.
        input_file (str | None): File name that was being processed, or `None`
            if it could not be read from the row.
        error (Exception): The exception that stopped the processing.
        metadata_columns (Iterable[str]): Stage-specific metadata column names;
            each is set to `null_value`.
        null_value (Any): Value written for missing results, usually
            `metadata.dataframe_columns.null_value` (`NaN` by default).
        stage (str | None): Name of the step that failed, e.g.
            `"load_field_of_view"`.
        **extra_fields: Additional identifiers placed right after `input_file`.

    Returns:
        dict: Result record with `success=False`, `output_file=None`, the
        error type and message, and every metadata column set to `null_value`.
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
    metadata_df: pd.DataFrame,
    results: list[dict],
    metadata_columns: Iterable[str],
    copy_dataframe: bool = True,
) -> pd.DataFrame:
    """Write the metadata columns of per-row results into the metadata dataframe.

    Rows are matched by `row_index`. Columns that do not exist yet are added
    and filled with `pd.NA` for rows without a result. All `metadata_columns`
    are cast to `object` dtype so that numbers, strings and `NaN` can coexist.

    Args:
        metadata_df (pd.DataFrame): Metadata dataframe whose rows were processed.
        results (list[dict]): Result records from `make_success_result` or
            `make_failure_result`.
        metadata_columns (Iterable[str]): Names of the columns to copy from the
            results.
        copy_dataframe (bool): If `True`, `metadata_df` is left unchanged and a
            modified copy is returned.

    Returns:
        pd.DataFrame: Metadata with the result columns filled in. If `results`
        is empty, the (copied) input is returned unchanged.

    Raises:
        KeyError: If a name in `metadata_columns` is missing from the results.
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


def process_rows(
    metadata_df: pd.DataFrame,
    process_row: Callable[[Hashable, pd.Series], dict],
    description: str,
    max_rows: int | None = None,
) -> list[dict]:
    """Process metadata rows one by one and collect their result records.

    Shows a progress bar (`tqdm.auto`, so it works in notebooks and scripts).
    `process_row` is expected to catch its own errors and return a failure
    record; an exception it raises stops the whole loop.

    Args:
        metadata_df (pd.DataFrame): Rows to process, in index order.
        process_row (Callable[[Hashable, pd.Series], dict]): Called as
            `process_row(row_index, metadata_row)`; returns one result record.
        description (str): Label of the progress bar.
        max_rows (int | None): If given, only the first `max_rows` rows are
            processed, e.g. for a quick test run.

    Returns:
        list[dict]: One result record per processed row, in row order.
    """
    row_indices = metadata_df.index
    if max_rows is not None:
        row_indices = row_indices[:max_rows]

    return [
        process_row(row_index, metadata_df.loc[row_index])
        for row_index in tqdm(row_indices, desc=description)
    ]
