"""Apply a background function to every field of view in the metadata."""

import logging
from pathlib import Path

from acid.image_processing.background.load_background_function import (
    BackgroundFunctionStrategy,
)
from acid.image_processing.correct_background import correct_background

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


def correct_background_image(image, background, config):
    """Apply background correction to one field-of-view image."""
    correction_kwargs = {
        "channel_axis": config.channel_axis,
        "method": config.method,
        "offset": config.offset,
        "epsilon": config.epsilon,
        "output_dtype": config.output_dtype,
        "working_dtype": config.working_dtype,
        "rescale_background": config.rescale_background,
        "clip_corrected_image": config.clip_corrected_image,
        "min_clip_value": config.min_clip_value,
        "max_clip_value": config.max_clip_value,
        "offset_background": config.offset_background,
        "zero_kwargs": config.zero_kwargs,
        "verbose": config.verbose,
    }

    return correct_background(
        image=image,
        background=background,
        **correction_kwargs,
    )
