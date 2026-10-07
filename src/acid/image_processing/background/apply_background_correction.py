"""Apply a background function to every field of view in the metadata."""

import logging
from pathlib import Path

from acid.image_processing.background.load_background_function import (
    BackgroundFunctionStrategy,
)

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

CORRECTION_METADATA_COLUMN_CONFIG_KEYS = (
    "illum_correct_df_date_clm_name",
    "illum_correct_df_file_name_clm_name",
    "illum_correct_df_method_clm_name",
    "illum_correct_df_offset_clm_name",
    "illum_correct_df_rescale_clm_name",
    "illum_correct_df_clipping_clm_name",
    "illum_correct_df_clip_min_value_clm_name",
    "illum_correct_df_clip_max_value_clm_name",
    "illum_correct_df_offset_background_clm_name",
)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def get_correction_metadata_columns(dataframe_columns) -> list[str]:
    """Return the metadata column names written by background correction."""
    return [dataframe_columns[key] for key in CORRECTION_METADATA_COLUMN_CONFIG_KEYS]


def get_background_for_fov(metadata_row, backgrounds, config):
    strategy = config.background_function_strategy

    if strategy == BackgroundFunctionStrategy.DATASET:
        return backgrounds

    if strategy == BackgroundFunctionStrategy.WELL:
        well = metadata_row[config.well_column_name]
        return backgrounds[well]

    if strategy == BackgroundFunctionStrategy.GRID_POSITION:
        grid_position = metadata_row[config.gridpos_column_name]
        return backgrounds[grid_position]

    raise ValueError(
        f"Invalid background_function_strategy: {strategy}. "
        "Please select either 1, 2 or 3."
    )


def make_output_filename(field_of_view_file, config):
    """Create the output filename for a background-corrected field of view."""
    field_of_view_file = Path(field_of_view_file)

    suffix = config.ome_suffix
    stem = field_of_view_file.name.removesuffix(suffix)
    logger.debug("Stem: %s", stem)

    return (
        f"{stem}"
        f"{config.save_file_name_separator}"
        f"{config.fov_illumin_corrected_savingword}"
        f"{suffix}"
    )
