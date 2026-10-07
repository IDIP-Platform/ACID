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
import pandas as pd
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops_table

from acid.feature_extraction.measure_hessian_matrix import MeasureHessianMatrix
from acid.feature_extraction.measure_structure_tensor import MeasureStructureTensor
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


def extract_regionprops_features(
    label_image: np.ndarray,
    intensity_image: np.ndarray,
    properties: list[str],
    extra_properties: list,
) -> pd.DataFrame:
    """Measure standard and custom region properties of every object.

    Args:
        label_image (np.ndarray): Label mask, `(y, x)`; `0` is background.
        intensity_image (np.ndarray): Image with channels last, `(y, x, c)`,
            usually from `preprocess_field_of_view`.
        properties (list[str]): `regionprops` property names, usually
            `default_regionpros_props()`; include `"label"` so tables can be
            merged.
        extra_properties (list): Custom property functions, usually
            `regionpros_extra_props()`.

    Returns:
        pd.DataFrame: One row per object. Multichannel properties become one
        column per channel, e.g. `intensity_mean-0`.
    """
    return pd.DataFrame(
        regionprops_table(
            label_image,
            intensity_image=intensity_image,
            properties=properties,
            extra_properties=extra_properties,
        )
    )


def extract_hessian_features(
    label_image: np.ndarray, intensity_image: np.ndarray
) -> pd.DataFrame:
    """Measure Hessian-matrix eigenvalue statistics of every object.

    Uses `MeasureHessianMatrix` with its defaults: the eigenvalues are
    computed per channel and summarised (mean, max, min, std) inside each
    object after eroding the mask with a disk of radius 9. Objects smaller
    than about 19 pixels across vanish in the erosion and get no row.

    Args:
        label_image (np.ndarray): Label mask, `(y, x)`; `0` is background.
        intensity_image (np.ndarray): Image with channels last, `(y, x, c)`.

    Returns:
        pd.DataFrame: One row per remaining object with a `label` column and
        columns such as `hessian_eigv_1_intensity_mean-0`.
    """
    hessian_measurer = MeasureHessianMatrix(intensity_image)

    return hessian_measurer.measure_obj_hessian_matrix_eigenval(
        label_image=label_image,
        axis=-1,
    )


def extract_structure_tensor_features(
    label_image: np.ndarray, intensity_image: np.ndarray
) -> pd.DataFrame:
    """Measure structure-tensor eigenvalue statistics of every object.

    Uses `MeasureStructureTensor` with its defaults: the eigenvalues are
    computed per channel and summarised (mean, max, min, std) inside each
    object after eroding the mask. As with `extract_hessian_features`, small
    objects can vanish in the erosion and get no row.

    Args:
        label_image (np.ndarray): Label mask, `(y, x)`; `0` is background.
        intensity_image (np.ndarray): Image with channels last, `(y, x, c)`.

    Returns:
        pd.DataFrame: One row per remaining object with a `label` column and
        columns such as `str_tens_eigv_1_intensity_mean-0`.
    """
    structure_tensor_measurer = MeasureStructureTensor(intensity_image)

    return structure_tensor_measurer.measure_obj_struct_tensor_eigenval(
        label_image=label_image,
        axis=-1,
    )
