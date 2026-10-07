"""Segment objects in every background-corrected field of view.

This module is the object-segmentation stage (`part5` notebook). For each
metadata row it loads the corrected field of view, prepares the two-channel
model input (`acid.image_processing.segmentation_preprocessing`), runs the
segmentation model, resizes the label mask back to the full image size, saves
it as an OME-TIFF and returns one result record (see
`acid.utils.row_processing`).

The model is passed in as an argument (any object with
`eval(image, **kwargs) -> (masks, flows, styles)` and a `version` attribute,
e.g. from `acid.segmentation.registry.create_segmentation_model`), so this
module does not import a model backend itself.

Most functions take the `object_segmentation` configuration section or one of
its subsections; each docstring names the section and the keys it reads.
"""

import logging
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from skimage.transform import resize

from acid.image_processing.extract_metadata import extract_ometif_imagej_metadata
from acid.image_processing.make_imagej_metadata import imagej_compatible_metadata_dict
from acid.image_processing.segmentation_preprocessing import (
    preprocess_image_for_segmentation,
)
from acid.io.image_loading import load_field_of_view
from acid.utils.metadata.rows import get_required_filename
from acid.utils.row_processing import make_failure_result, make_success_result
from acid.utils.save_image import tifffile_save_ometiff

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

#: Keys of `object_segmentation.metadata.dataframe_columns` whose values are
#: the metadata columns written by this stage, in output order.
SEGMENTATION_METADATA_COLUMN_CONFIG_KEYS = (
    "metadata_df_date_clm_name",
    "metadata_df_file_name_clm_name",
    "metadata_df_method_clm_name",
    "metadata_df_method_version_clm_name",
    "metadata_df_diameter_clm_name",
    "metadata_df_flow_threshold_clm_name",
    "metadata_df_cellprob_threshold_clm_name",
    "metadata_df_downsampling_factor_clm_name",
    "metadata_df_nucleus_med_filter_size_name",
    "metadata_df_concactin_merge_med_filter_size_name",
    "metadata_df_resize_order_name",
    "metadata_df_output_dtype_name",
)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def get_segmentation_metadata_columns(dataframe_columns: DictConfig) -> list[str]:
    """Return the metadata column names written by object segmentation.

    Args:
        dataframe_columns (DictConfig): The
            `object_segmentation.metadata.dataframe_columns` section. Reads the
            keys listed in `SEGMENTATION_METADATA_COLUMN_CONFIG_KEYS`.

    Returns:
        list[str]: The 12 column names (date, mask file name, method, method
        version and the segmentation settings).
    """
    return [dataframe_columns[key] for key in SEGMENTATION_METADATA_COLUMN_CONFIG_KEYS]


def get_channel_shape(image: np.ndarray, channel_axis: int) -> tuple[int, ...]:
    """Return the image shape without its channel axis.

    Used as the target shape when the downsampled mask is resized back to
    full resolution.

    Args:
        image (np.ndarray): Image with a channel axis, e.g. `(channels, y, x)`.
        channel_axis (int): Index of the channel axis; must be non-negative.

    Returns:
        tuple[int, ...]: Shape of one channel, e.g. `(y, x)`.
    """
    return tuple(size for axis, size in enumerate(image.shape) if axis != channel_axis)


def cast_mask_to_output_dtype(
    mask: np.ndarray, output_dtype: str | np.dtype | None
) -> np.ndarray:
    """Cast a label mask to the configured output dtype.

    Args:
        mask (np.ndarray): Label mask.
        output_dtype (str | np.dtype | None): Target dtype, usually
            `object_segmentation.processing.output_dtype` (e.g. `"uint16"`).
            If `None`, the mask is returned unchanged.

    Returns:
        np.ndarray: The mask in `output_dtype`, or the same object when
        `output_dtype` is `None`. Labels above the dtype's maximum overflow, so
        the dtype must hold the largest label.
    """
    if output_dtype is None:
        return mask

    return mask.astype(output_dtype)


def segment_objects(preprocessed_image: np.ndarray, model, config: DictConfig) -> tuple:
    """Run the segmentation model on a preprocessed image.

    Args:
        preprocessed_image (np.ndarray): Output of
            `preprocess_image_for_segmentation`, e.g. `(2, y / f, x / f)`.
        model: Segmentation model with `eval(image, **kwargs)`, e.g.
            `CellposeSegmentationModel`. Runs on GPU when the model was
            created with one.
        config (DictConfig): The whole `object_segmentation` section. Reads
            `processing.flow_threshold`, `processing.cellprob_threshold`,
            `processing.diameter` and `processing.channel_axis`.

    Returns:
        tuple: The model output `(masks, flows, styles)`; `masks` is a label
        image on the downsampled grid, `0` being background.
    """
    return model.eval(
        preprocessed_image,
        flow_threshold=config.processing.flow_threshold,
        cellprob_threshold=config.processing.cellprob_threshold,
        diameter=config.processing.diameter,
        channel_axis=config.processing.channel_axis,
    )


def resize_segmentation_mask(
    mask: np.ndarray, output_shape: tuple[int, ...], config: DictConfig
) -> np.ndarray:
    """Resize a label mask from the downsampled grid to full resolution.

    Args:
        mask (np.ndarray): Label mask returned by the model.
        output_shape (tuple[int, ...]): Target shape, usually from
            `get_channel_shape`.
        config (DictConfig): The whole `object_segmentation` section. Reads
            `processing.order` (`0` = nearest neighbour, keeps labels intact),
            `processing.preserve_range` and `processing.anti_aliasing` (keep
            `False` for label images), passed to `skimage.transform.resize`.

    Returns:
        np.ndarray: Resized mask with the dtype of `mask`.
    """
    resized_mask = resize(
        mask,
        output_shape=output_shape,
        order=config.processing.order,
        preserve_range=config.processing.preserve_range,
        anti_aliasing=config.processing.anti_aliasing,
    )

    return resized_mask.astype(mask.dtype, copy=False)


def make_segmentation_output_filename(
    field_of_view_file: str, config: DictConfig
) -> str:
    """Create the file name of a segmentation mask.

    The saving word is inserted before the OME suffix, e.g. `a_bg.ome.tif`
    becomes `a_bg_segmentation.ome.tif` with the default configuration.

    Args:
        field_of_view_file (str): File name (or path) of the corrected field of
            view; only the name is used.
        config (DictConfig): The whole `object_segmentation` section. Reads
            `image_saving.ome_suffix`, `image_saving.save_file_name_separator`
            and `image_saving.segmentation_savingword`.

    Returns:
        str: Mask file name without a directory.
    """
    field_of_view_file = Path(field_of_view_file)
    suffix = config.image_saving.ome_suffix
    stem = field_of_view_file.name.removesuffix(suffix)

    return (
        f"{stem}"
        f"{config.image_saving.save_file_name_separator}"
        f"{config.image_saving.segmentation_savingword}"
        f"{suffix}"
    )


def save_segmentation_mask(
    output_filename: str,
    mask: np.ndarray,
    image_metadata: dict,
    config: DictConfig,
    output_directory: str | Path,
) -> Path:
    """Save one segmentation mask as an OME-TIFF.

    Creates `output_directory` when it does not exist and overwrites an
    existing file with the same name.

    Args:
        output_filename (str): Mask file name, usually from
            `make_segmentation_output_filename`.
        mask (np.ndarray): Full-resolution label mask, `(y, x)`.
        image_metadata (dict): ImageJ-compatible metadata written into the
            file, usually from `build_segmentation_image_metadata`.
        config (DictConfig): The whole `object_segmentation` section. Reads
            `image_saving.save_imagej_compatible` and `image_saving.photometric`.
        output_directory (str | Path): Target directory, usually
            `shared.paths.segmentation_masks_dir`.

    Returns:
        Path: Full path of the written file.
    """
    output_path = Path(output_directory) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tifffile_save_ometiff(
        output_path,
        data=mask,
        imagej=config.image_saving.save_imagej_compatible,
        photometric=config.image_saving.photometric,
        metadata=image_metadata,
    )

    return output_path


def copy_selected_field_of_view_metadata(
    segmentation_metadata: dict, field_of_view_metadata: dict, config: DictConfig
) -> dict:
    """Copy provenance and pixel-size entries of the field of view into the mask metadata.

    An entry is copied when its key contains one of the configured entry
    names (raw file name, scene file name, physical pixel size and its unit
    in x and y). All other field-of-view entries are dropped.

    Args:
        segmentation_metadata (dict): Mask metadata to extend; modified in
            place.
        field_of_view_metadata (dict): ImageJ metadata of the corrected field
            of view.
        config (DictConfig): The whole `object_segmentation` section. Reads
            `metadata.image_metadata.preproc_img_meta_raw_file_name_entry`,
            `preproc_img_meta_scene_file_name_entry`,
            `preproc_img_meta_x_physic_px_size_entry`,
            `preproc_img_meta_y_physic_px_size_entry`,
            `preproc_img_meta_x_physic_px_size_unit_entry` and
            `preproc_img_meta_y_physic_px_size_unit_entry`.

    Returns:
        dict: `segmentation_metadata` with the selected entries added.
    """
    metadata_keywords = [
        config.metadata.image_metadata.preproc_img_meta_raw_file_name_entry,
        config.metadata.image_metadata.preproc_img_meta_scene_file_name_entry,
        config.metadata.image_metadata.preproc_img_meta_x_physic_px_size_entry,
        config.metadata.image_metadata.preproc_img_meta_y_physic_px_size_entry,
        config.metadata.image_metadata.preproc_img_meta_x_physic_px_size_unit_entry,
        config.metadata.image_metadata.preproc_img_meta_y_physic_px_size_unit_entry,
    ]

    for key, value in field_of_view_metadata.items():
        if any(keyword in key for keyword in metadata_keywords):
            segmentation_metadata[key] = value

    return segmentation_metadata


def build_segmentation_image_metadata(
    field_of_view_path: str | Path, mask: np.ndarray, model, config: DictConfig
) -> dict:
    """Build the ImageJ metadata of a segmentation mask.

    Records the segmentation date, method and version, all segmentation
    settings and the mask dtype, and copies provenance and pixel-size entries
    from the field of view (see `copy_selected_field_of_view_metadata`).
    Keys get the `custom_` prefix.

    Args:
        field_of_view_path (str | Path): Full path of the corrected field of
            view whose ImageJ metadata is read.
        mask (np.ndarray): Final mask; only its dtype is recorded.
        model: Segmentation model; its `version` attribute is recorded
            (`None` if missing).
        config (DictConfig): The whole `object_segmentation` section. Reads the
            entry names in `metadata.image_metadata` (`segmented_img_meta_*`,
            `segmentation_method_version_name`, `processing_date_format`),
            `metadata.segmentation_method_name` and the settings in
            `processing` (`diameter`, `flow_threshold`, `cellprob_threshold`,
            `downsampling_factor`, `med_filter_nucleus`,
            `med_filter_concactin_merge`, `order`, `output_dtype`).

    Returns:
        dict: ImageJ-compatible metadata for `save_segmentation_mask`.

    Raises:
        FileNotFoundError: If `field_of_view_path` does not exist.
    """
    field_of_view_metadata = extract_ometif_imagej_metadata(field_of_view_path)

    processing_steps = (
        config.metadata.image_metadata.segmented_img_meta_processing_steps
    )

    if config.processing.output_dtype is not None:
        processing_steps = f"{processing_steps} change output data type to {config.processing.output_dtype} for saving."

    segmentation_metadata = {
        config.metadata.image_metadata.segmented_img_meta_date_name: datetime.now().strftime(
            config.metadata.image_metadata.processing_date_format
        ),
        config.metadata.image_metadata.segmented_img_meta_method_name: config.metadata.segmentation_method_name,
        config.metadata.image_metadata.segmentation_method_version_name: getattr(
            model, "version", None
        ),
        config.metadata.image_metadata.segmented_img_meta_diameter_name: config.processing.diameter,
        config.metadata.image_metadata.segmented_img_meta_flow_threshold_name: config.processing.flow_threshold,
        config.metadata.image_metadata.segmented_img_meta_cellprob_threshold_name: config.processing.cellprob_threshold,
        config.metadata.image_metadata.segmented_img_meta_downsampling_factor_name: config.processing.downsampling_factor,
        config.metadata.image_metadata.segmented_img_meta_nucleus_med_filter_size_name: config.processing.med_filter_nucleus,
        config.metadata.image_metadata.segmented_img_meta_concactin_merge_med_filter_size_name: (
            config.processing.med_filter_concactin_merge
        ),
        config.metadata.image_metadata.segmented_img_meta_resize_order_name: config.processing.order,
        config.metadata.image_metadata.segmented_img_meta_processing_name: processing_steps,
        config.metadata.image_metadata.segmented_img_meta_dtype_name: str(mask.dtype),
    }

    imagej_metadata = imagej_compatible_metadata_dict(segmentation_metadata)

    return copy_selected_field_of_view_metadata(
        segmentation_metadata=imagej_metadata,
        field_of_view_metadata=field_of_view_metadata,
        config=config,
    )


def make_segmentation_success_result(
    row_index: Hashable,
    input_file: str,
    output_file: str,
    mask_dtype: np.dtype,
    model,
    config: DictConfig,
) -> dict:
    """Build the result record of one successfully segmented field of view.

    Args:
        row_index (Hashable): Index of the row in the metadata dataframe.
        input_file (str): File name of the corrected field of view.
        output_file (str): File name of the saved mask.
        mask_dtype (np.dtype): Dtype of the saved mask.
        model: Segmentation model; its `version` attribute is recorded
            (`None` if missing).
        config (DictConfig): The whole `object_segmentation` section. Reads the
            column names in `metadata.dataframe_columns` (including
            `metadata_df_meta_date_format`), `metadata.segmentation_method_name`
            and the recorded settings in `processing`.

    Returns:
        dict: Result record (see `acid.utils.row_processing`) with the
        segmentation date, mask file name, method, version, settings and mask
        dtype in the 12 segmentation metadata columns.
    """
    columns = config.metadata.dataframe_columns
    processing = config.processing

    return make_success_result(
        row_index=row_index,
        input_file=input_file,
        output_file=output_file,
        metadata_values={
            columns.metadata_df_date_clm_name: datetime.now().strftime(
                columns.metadata_df_meta_date_format
            ),
            columns.metadata_df_file_name_clm_name: output_file,
            columns.metadata_df_method_clm_name: config.metadata.segmentation_method_name,
            columns.metadata_df_method_version_clm_name: getattr(model, "version", None),
            columns.metadata_df_diameter_clm_name: processing.diameter,
            columns.metadata_df_flow_threshold_clm_name: processing.flow_threshold,
            columns.metadata_df_cellprob_threshold_clm_name: processing.cellprob_threshold,
            columns.metadata_df_downsampling_factor_clm_name: processing.downsampling_factor,
            columns.metadata_df_nucleus_med_filter_size_name: processing.med_filter_nucleus,
            columns.metadata_df_concactin_merge_med_filter_size_name: (
                processing.med_filter_concactin_merge
            ),
            columns.metadata_df_resize_order_name: processing.order,
            columns.metadata_df_output_dtype_name: str(mask_dtype),
        },
    )


def apply_segmentation_for_fov(
    row_index: Hashable,
    metadata_row: pd.Series,
    model,
    config: DictConfig,
    paths: DictConfig,
) -> dict:
    """Segment, resize, save and describe one field of view.

    Runs these steps in order; the name in brackets is the `stage` reported if
    the step fails: read the file name (`get_field_of_view_file`), load the
    corrected image (`load_field_of_view`), build the model input
    (`preprocess_image_for_segmentation`), run the model
    (`segment_preprocessed_image`), resize the mask to full resolution
    (`resize_segmentation_mask`), build its metadata
    (`build_segmentation_image_metadata`), save it (`save_segmentation_mask`).
    The mask is cast to `processing.output_dtype` before saving.
    A failing step returns a failure record instead of raising, so one bad
    file does not stop the batch.

    Args:
        row_index (Hashable): Index of the row in the metadata dataframe.
        metadata_row (pd.Series): The row; its file name is read from
            `metadata.dataframe_columns.illum_correct_df_file_name_clm_name`
            (the output of background correction).
        model: Segmentation model, see `segment_objects`.
        config (DictConfig): The whole `object_segmentation` section.
        paths (DictConfig): The `shared.paths` section. Reads
            `corrected_fov_dir` and `segmentation_masks_dir`.

    Returns:
        dict: Success record from `make_segmentation_success_result`, or
        failure record with the segmentation metadata columns set to
        `metadata.dataframe_columns.null_value`.
    """
    columns = config.metadata.dataframe_columns
    field_of_view_file = None

    def failure(stage, error):
        return make_failure_result(
            row_index=row_index,
            input_file=field_of_view_file,
            error=error,
            metadata_columns=get_segmentation_metadata_columns(columns),
            null_value=columns.null_value,
            stage=stage,
        )

    try:
        field_of_view_file = get_required_filename(
            metadata_row, columns.illum_correct_df_file_name_clm_name
        )
    except Exception as error:
        return failure("get_field_of_view_file", error)

    try:
        image = load_field_of_view(field_of_view_file, paths.corrected_fov_dir)
    except Exception as error:
        return failure("load_field_of_view", error)

    try:
        preprocessed_image = preprocess_image_for_segmentation(
            image=image, config=config.processing
        )
    except Exception as error:
        return failure("preprocess_image_for_segmentation", error)
    logger.debug("Preprocessing image shape: %s", preprocessed_image.shape)

    try:
        masks, _flows, _styles = segment_objects(
            preprocessed_image=preprocessed_image, model=model, config=config
        )
    except Exception as error:
        return failure("segment_preprocessed_image", error)
    logger.debug("Number of segmented objects: %d", np.unique(masks).size - 1)

    channel_shape = get_channel_shape(image, config.processing.channel_axis)
    logger.info("Channel shape: %s", channel_shape)

    try:
        mask = resize_segmentation_mask(
            mask=masks, output_shape=channel_shape, config=config
        )
    except Exception as error:
        return failure("resize_segmentation_mask", error)

    mask = cast_mask_to_output_dtype(mask, config.processing.output_dtype)

    try:
        image_metadata = build_segmentation_image_metadata(
            field_of_view_path=Path(paths.corrected_fov_dir) / field_of_view_file,
            mask=mask,
            model=model,
            config=config,
        )
    except Exception as error:
        return failure("build_segmentation_image_metadata", error)

    try:
        output_filename = make_segmentation_output_filename(field_of_view_file, config)
        save_segmentation_mask(
            output_filename=output_filename,
            mask=mask,
            image_metadata=image_metadata,
            config=config,
            output_directory=paths.segmentation_masks_dir,
        )
    except Exception as error:
        logger.error("Error saving masks")
        return failure("save_segmentation_mask", error)
    logger.info("Output file: %s", output_filename)

    return make_segmentation_success_result(
        row_index=row_index,
        input_file=field_of_view_file,
        output_file=output_filename,
        mask_dtype=mask.dtype,
        model=model,
        config=config,
    )
