"""Apply a background function to every field of view in the metadata."""

import logging
from datetime import datetime
from pathlib import Path

from acid.image_processing.background.load_background_function import (
    BackgroundFunctionStrategy,
)
from acid.image_processing.correct_background import correct_background
from acid.image_processing.extract_metadata import extract_ometif_imagej_metadata
from acid.image_processing.make_imagej_metadata import imagej_compatible_metadata_dict
from acid.io.image_loading import load_field_of_view
from acid.utils.metadata.rows import get_required_filename
from acid.utils.row_processing import (
    make_failure_result,
    make_success_result,
    process_rows,
    update_metadata_with_results,
)
from acid.utils.save_image import tifffile_save_ometiff

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


def build_image_metadata(field_of_view_file, fov_directory, config):
    field_of_view_path = Path(fov_directory) / str(field_of_view_file)

    image_metadata = extract_ometif_imagej_metadata(field_of_view_path)

    metadata_cfg = config.metadata
    image_metadata_cfg = metadata_cfg.image_metadata
    processing_cfg = config.processing

    processing_metadata = {
        image_metadata_cfg.proc_img_meta_date_name: datetime.now().strftime(
            image_metadata_cfg.processing_date_format
        ),
        image_metadata_cfg.proc_img_meta_dtype_name: processing_cfg.output_dtype,
    }

    background_correction_metadata = {
        image_metadata_cfg.illum_corr_method_metadata_entry: processing_cfg.method,
        image_metadata_cfg.illum_corr_offset_metadata_entry: processing_cfg.offset,
        image_metadata_cfg.illum_corr_rescale_metadata_entry: processing_cfg.rescale_background,
        image_metadata_cfg.illum_corr_clipping_metadata_entry: processing_cfg.clip_corrected_image,
        image_metadata_cfg.illum_corr_clip_min_value_metadata_entry: processing_cfg.min_clip_value,
        image_metadata_cfg.illum_corr_clip_max_value_metadata_entry: processing_cfg.max_clip_value,
        image_metadata_cfg.illum_corr_offset_background_metadata_entry: processing_cfg.offset_background,
    }

    image_metadata.update(imagej_compatible_metadata_dict(processing_metadata))
    image_metadata.update(
        imagej_compatible_metadata_dict(background_correction_metadata)
    )

    return image_metadata


def save_corrected_image(output_filename, corrected_image, image_metadata, config):
    """Save one background-corrected field-of-view image."""

    output_path = Path(config.directory) / str(output_filename)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile_save_ometiff(
        output_path,
        data=corrected_image,
        imagej=config.save_imagej_compatible,
        photometric=config.photometric,
        metadata=image_metadata,
    )

    return output_path


def make_correction_success_result(row_index, field_of_view_file, output_file, config):
    """Build the result of one successfully corrected field of view."""
    columns = config.metadata.dataframe_columns
    processing = config.processing

    return make_success_result(
        row_index=row_index,
        input_file=field_of_view_file,
        output_file=output_file,
        metadata_values={
            columns.illum_correct_df_date_clm_name: datetime.now().strftime(
                columns.illum_correct_df_meta_date_format
            ),
            columns.illum_correct_df_file_name_clm_name: output_file,
            columns.illum_correct_df_method_clm_name: processing.method,
            columns.illum_correct_df_offset_clm_name: processing.offset,
            columns.illum_correct_df_rescale_clm_name: processing.rescale_background,
            columns.illum_correct_df_clipping_clm_name: processing.clip_corrected_image,
            columns.illum_correct_df_clip_min_value_clm_name: processing.min_clip_value,
            columns.illum_correct_df_clip_max_value_clm_name: processing.max_clip_value,
            columns.illum_correct_df_offset_background_clm_name: processing.offset_background,
        },
    )


def apply_background_correction_for_fov(
    row_index, metadata_row, backgrounds, background_correction_config, paths
):
    """Correct, save and describe one field of view.

    Every failing step returns a failure result naming that step in ``stage``
    instead of raising, so one bad file does not stop the batch.
    """
    config = background_correction_config
    columns = config.metadata.dataframe_columns
    field_of_view_file = None

    def failure(stage, error):
        return make_failure_result(
            row_index=row_index,
            input_file=field_of_view_file,
            error=error,
            metadata_columns=get_correction_metadata_columns(columns),
            null_value=columns.null_value,
            stage=stage,
        )

    try:
        field_of_view_file = get_required_filename(metadata_row, columns.fov_column_name)
    except Exception as error:
        return failure("get_field_of_view_file", error)

    try:
        field_of_view = load_field_of_view(field_of_view_file, paths.extracted_fov_dir)
    except Exception as error:
        return failure("load_field_of_view", error)

    try:
        image_metadata = build_image_metadata(
            field_of_view_file=field_of_view_file,
            fov_directory=paths.extracted_fov_dir,
            config=config,
        )
    except Exception as error:
        return failure("build_image_metadata", error)

    try:
        background = get_background_for_fov(
            metadata_row=metadata_row,
            backgrounds=backgrounds,
            config=config.background_function_selection,
        )
    except Exception as error:
        return failure("get_background_for_fov", error)

    try:
        corrected_image = correct_background_image(
            image=field_of_view, background=background, config=config.processing
        )
    except Exception as error:
        return failure("correct_background_image", error)

    output_filename = make_output_filename(field_of_view_file, config.image_saving)
    logger.info("Output filename: %s", output_filename)

    try:
        save_corrected_image(
            output_filename=output_filename,
            corrected_image=corrected_image,
            image_metadata=image_metadata,
            config=config.image_saving,
        )
    except Exception as error:
        return failure("save_corrected_image", error)

    return make_correction_success_result(
        row_index, field_of_view_file, output_filename, config
    )


def apply_background_correction_batch(
    metadata_df, backgrounds, config, paths, max_rows=None
):
    """Apply background correction to each metadata row; return per-row results."""
    return process_rows(
        metadata_df,
        lambda row_index, metadata_row: apply_background_correction_for_fov(
            row_index, metadata_row, backgrounds, config, paths
        ),
        description="Applying background correction",
        max_rows=max_rows,
    )


def update_metadata_with_correction_results(
    metadata_df, results, dataframe_columns, copy_dataframe=True
):
    """Write background-correction results into the metadata dataframe."""
    return update_metadata_with_results(
        metadata_df,
        results,
        get_correction_metadata_columns(dataframe_columns),
        copy_dataframe=copy_dataframe,
    )
