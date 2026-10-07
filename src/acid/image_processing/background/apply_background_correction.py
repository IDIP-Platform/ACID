"""Apply a background function to every field of view in the metadata.

This module is the background-correction stage (`part4` notebook). For each
metadata row it loads the extracted field of view, picks the matching
background function, corrects the image, saves it as an OME-TIFF with the
correction settings in its ImageJ metadata, and returns one result record
(see `acid.utils.row_processing`). The records are then written into the
metadata columns named in `background_correction.metadata.dataframe_columns`.

Most functions take a section of the `background_correction` configuration;
each docstring names the section and the keys it reads.
"""

import logging
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

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

#: Keys of `background_correction.metadata.dataframe_columns` whose values are
#: the metadata columns written by this stage, in output order.
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


def get_correction_metadata_columns(dataframe_columns: DictConfig) -> list[str]:
    """Return the metadata column names written by background correction.

    Args:
        dataframe_columns (DictConfig): The
            `background_correction.metadata.dataframe_columns` section. Reads
            the keys listed in `CORRECTION_METADATA_COLUMN_CONFIG_KEYS`.

    Returns:
        list[str]: Column names, e.g. `illumination_correction_date`,
        `illumination_correction_file_name`, ... (9 columns).
    """
    return [dataframe_columns[key] for key in CORRECTION_METADATA_COLUMN_CONFIG_KEYS]


def get_background_for_fov(
    metadata_row: pd.Series,
    backgrounds: np.ndarray | dict[Hashable, np.ndarray],
    config: DictConfig,
) -> np.ndarray:
    """Select the background function that belongs to one field of view.

    Args:
        metadata_row (pd.Series): Metadata row of the field of view.
        backgrounds (np.ndarray | dict[Hashable, np.ndarray]): Output of
            `load_background_function`: a single array for strategy `1`, or a
            dict keyed by well (strategy `2`) or grid position (strategy `3`).
        config (DictConfig): The
            `background_correction.background_function_selection` section.
            Reads `background_function_strategy` and, depending on it,
            `well_column_name` or `gridpos_column_name`.

    Returns:
        np.ndarray: Background function with the same shape as the field of
        view.

    Raises:
        KeyError: If the row's well or grid position has no background.
        ValueError: If `background_function_strategy` is not 1, 2 or 3.
    """
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


def make_output_filename(field_of_view_file: str, config: DictConfig) -> str:
    """Create the file name of a background-corrected field of view.

    The saving word is inserted before the OME suffix, e.g. `a.ome.tif` becomes
    `a_bg.ome.tif` with the default configuration.

    Args:
        field_of_view_file (str): File name (or path) of the extracted field of
            view; only the name is used.
        config (DictConfig): The `background_correction.image_saving` section.
            Reads `ome_suffix`, `save_file_name_separator` and
            `fov_illumin_corrected_savingword`.

    Returns:
        str: Output file name without a directory.
    """
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


def correct_background_image(
    image: np.ndarray, background: np.ndarray, config: DictConfig
) -> np.ndarray:
    """Apply background correction to one field-of-view image.

    Thin wrapper that passes the configured settings to
    `acid.image_processing.correct_background.correct_background`.

    Args:
        image (np.ndarray): Field of view, e.g. `(channels, y, x)`.
        background (np.ndarray): Background function with the same shape as
            `image`.
        config (DictConfig): The `background_correction.processing` section.
            Reads `channel_axis`, `method` (`"division"` or `"subtraction"`),
            `offset`, `epsilon`, `output_dtype`, `working_dtype`,
            `rescale_background`, `clip_corrected_image`, `min_clip_value`,
            `max_clip_value`, `offset_background`, `zero_kwargs` and `verbose`.

    Returns:
        np.ndarray: Corrected image with the shape of `image` and dtype
        `output_dtype`.

    Raises:
        ValueError: If the settings are inconsistent with the data, e.g. when
            `offset` is not smaller than the minimum pixel value of the image
            or (with `offset_background`) of the background.
    """
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


def build_image_metadata(
    field_of_view_file: str, fov_directory: str | Path, config: DictConfig
) -> dict:
    """Build the ImageJ metadata of a corrected field of view.

    Starts from the ImageJ metadata stored in the extracted field of view and
    adds the processing date, output dtype and all correction settings. Added
    keys get the `custom_` prefix (see `imagej_compatible_metadata_dict`).

    Args:
        field_of_view_file (str): File name of the extracted field of view.
        fov_directory (str | Path): Directory of the extracted fields of view,
            usually `shared.paths.extracted_fov_dir`.
        config (DictConfig): The whole `background_correction` section. Reads
            the entry names in `metadata.image_metadata`
            (`proc_img_meta_date_name`, `processing_date_format`,
            `proc_img_meta_dtype_name`, `illum_corr_*_metadata_entry`) and
            their values from `processing` (`output_dtype`, `method`,
            `offset`, `rescale_background`, `clip_corrected_image`,
            `min_clip_value`, `max_clip_value`, `offset_background`).

    Returns:
        dict: ImageJ-compatible metadata for `tifffile_save_ometiff`.

    Raises:
        FileNotFoundError: If the field of view does not exist.
    """
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


def save_corrected_image(
    output_filename: str,
    corrected_image: np.ndarray,
    image_metadata: dict,
    config: DictConfig,
) -> Path:
    """Save one background-corrected field of view as an OME-TIFF.

    Creates the output directory when it does not exist and overwrites an
    existing file with the same name.

    Args:
        output_filename (str): File name of the corrected image, usually from
            `make_output_filename`.
        corrected_image (np.ndarray): Corrected image, channels on the
            configured channel axis.
        image_metadata (dict): ImageJ-compatible metadata written into the
            file, usually from `build_image_metadata`.
        config (DictConfig): The `background_correction.image_saving` section.
            Reads `directory`, `save_imagej_compatible` and `photometric`.

    Returns:
        Path: Full path of the written file.
    """

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


def make_correction_success_result(
    row_index: Hashable,
    field_of_view_file: str,
    output_file: str,
    config: DictConfig,
) -> dict:
    """Build the result record of one successfully corrected field of view.

    Args:
        row_index (Hashable): Index of the row in the metadata dataframe.
        field_of_view_file (str): File name of the extracted field of view.
        output_file (str): File name of the corrected image.
        config (DictConfig): The whole `background_correction` section. Reads
            the column names in `metadata.dataframe_columns` (including
            `illum_correct_df_meta_date_format`) and the recorded settings in
            `processing`.

    Returns:
        dict: Result record (see `acid.utils.row_processing`) with the
        correction date, output file name and correction settings in the
        metadata columns.
    """
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
    row_index: Hashable,
    metadata_row: pd.Series,
    backgrounds: np.ndarray | dict[Hashable, np.ndarray],
    background_correction_config: DictConfig,
    paths: DictConfig,
) -> dict:
    """Correct, save and describe one field of view.

    Runs these steps in order; the name in brackets is the `stage` reported if
    the step fails: read the file name (`get_field_of_view_file`), load the
    image (`load_field_of_view`), build its metadata (`build_image_metadata`),
    select the background (`get_background_for_fov`), correct it
    (`correct_background_image`), save it (`save_corrected_image`).
    A failing step returns a failure record instead of raising, so one bad
    file does not stop the batch.

    Args:
        row_index (Hashable): Index of the row in the metadata dataframe.
        metadata_row (pd.Series): The row; its file name is read from
            `metadata.dataframe_columns.fov_column_name`.
        backgrounds (np.ndarray | dict[Hashable, np.ndarray]): Output of
            `load_background_function` (see `get_background_for_fov`).
        background_correction_config (DictConfig): The whole
            `background_correction` section.
        paths (DictConfig): The `shared.paths` section. Reads
            `extracted_fov_dir`.

    Returns:
        dict: Success record from `make_correction_success_result`, or failure
        record with the correction metadata columns set to
        `metadata.dataframe_columns.null_value`.
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
    metadata_df: pd.DataFrame,
    backgrounds: np.ndarray | dict[Hashable, np.ndarray],
    config: DictConfig,
    paths: DictConfig,
    max_rows: int | None = None,
) -> list[dict]:
    """Apply background correction to every row of the metadata dataframe.

    Args:
        metadata_df (pd.DataFrame): Rows to process, usually the training
            split selected by `filter_metadata_by_splits`.
        backgrounds (np.ndarray | dict[Hashable, np.ndarray]): Output of
            `load_background_function`.
        config (DictConfig): The whole `background_correction` section.
        paths (DictConfig): The `shared.paths` section.
        max_rows (int | None): If given, only the first `max_rows` rows are
            processed, e.g. for a quick test run.

    Returns:
        list[dict]: One result record per row (see
        `apply_background_correction_for_fov`).
    """
    return process_rows(
        metadata_df,
        lambda row_index, metadata_row: apply_background_correction_for_fov(
            row_index, metadata_row, backgrounds, config, paths
        ),
        description="Applying background correction",
        max_rows=max_rows,
    )


def update_metadata_with_correction_results(
    metadata_df: pd.DataFrame,
    results: list[dict],
    dataframe_columns: DictConfig,
    copy_dataframe: bool = True,
) -> pd.DataFrame:
    """Write background-correction results into the metadata dataframe.

    Args:
        metadata_df (pd.DataFrame): Metadata whose rows were processed.
        results (list[dict]): Output of `apply_background_correction_batch`.
        dataframe_columns (DictConfig): The
            `background_correction.metadata.dataframe_columns` section.
        copy_dataframe (bool): If `True`, `metadata_df` is left unchanged.

    Returns:
        pd.DataFrame: Metadata with the correction columns filled in; failed
        rows hold `null_value`, rows without a result hold `pd.NA`.

    Raises:
        KeyError: If the results lack one of the correction columns.
    """
    return update_metadata_with_results(
        metadata_df,
        results,
        get_correction_metadata_columns(dataframe_columns),
        copy_dataframe=copy_dataframe,
    )
