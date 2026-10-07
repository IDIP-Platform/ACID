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

import numpy as np
from omegaconf import DictConfig
from skimage.transform import resize

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
