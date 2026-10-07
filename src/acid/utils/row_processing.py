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
