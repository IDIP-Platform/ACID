"""Extract per-object features from every segmented field of view.

This module is the feature-extraction stage (`part6` notebook). For each
metadata row it loads the background-corrected field of view and its
segmentation mask, smooths the image, drops objects touching the image edge,
measures region properties plus Hessian and structure-tensor eigenvalue
features per object, saves one CSV per field of view and returns one result
record (see `acid.utils.row_processing`).

Most functions take the `feature_extraction` configuration section; each
docstring names the keys it reads.
"""

import logging

import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter

from acid.utils.label_image_utils import exclude_label_on_edge

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

#: Keys of `feature_extraction.metadata.dataframe_columns` whose values are the
#: metadata columns written by this stage, in output order.
FEATURE_METADATA_COLUMN_CONFIG_KEYS = (
    "metadata_df_date_clm_name",
    "metadata_df_file_name_clm_name",
    "metadata_df_method_clm_name",
)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def get_feature_metadata_columns(dataframe_columns: DictConfig) -> list[str]:
    """Return the metadata column names written by feature extraction.

    Args:
        dataframe_columns (DictConfig): The
            `feature_extraction.metadata.dataframe_columns` section. Reads the
            keys listed in `FEATURE_METADATA_COLUMN_CONFIG_KEYS`.

    Returns:
        list[str]: Column names for the extraction date, the feature table
        file name and the preprocessing description.
    """
    return [dataframe_columns[key] for key in FEATURE_METADATA_COLUMN_CONFIG_KEYS]


def preprocess_field_of_view(field_of_view: np.ndarray, config: DictConfig) -> np.ndarray:
    """Prepare a field of view for `skimage.measure.regionprops_table`.

    Moves the channel axis to the last position (the layout `regionprops`
    expects for multichannel intensity images) and applies a Gaussian filter.

    Args:
        field_of_view (np.ndarray): Background-corrected field of view, e.g.
            `(channels, y, x)`.
        config (DictConfig): The whole `feature_extraction` section. Reads
            `processing.channel_axis`, `processing.sigma` and
            `processing.axes`, the axes the filter runs along after the channel
            axis has been moved last. With the default `axes: -1` the filter
            smooths across channels, not across pixels.

    Returns:
        np.ndarray: Smoothed image with channels last, e.g. `(y, x, channels)`,
        in the dtype of `field_of_view`.
    """
    preprocessed = np.moveaxis(field_of_view, config.processing.channel_axis, -1)

    return gaussian_filter(
        preprocessed,
        sigma=config.processing.sigma,
        axes=config.processing.axes,
    )


def preprocess_segmentation_mask(segmentation: np.ndarray) -> np.ndarray:
    """Remove segmented objects that touch the image edge.

    Edge objects are cut off by the field-of-view border, so their shape and
    intensity features would be biased.

    Args:
        segmentation (np.ndarray): Label mask, `(y, x)`; `0` is background.

    Returns:
        np.ndarray: Copy of the mask with edge-touching labels set to `0`; the
        input is not modified.
    """
    return exclude_label_on_edge(segmentation)
