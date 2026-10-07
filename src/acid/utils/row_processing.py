"""Per-row processing results shared by the pipeline stages."""

import logging

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
