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
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops_table

from acid.feature_extraction.measure_hessian_matrix import MeasureHessianMatrix
from acid.feature_extraction.measure_structure_tensor import MeasureStructureTensor
from acid.utils.label_image_utils import exclude_label_on_edge
from acid.utils.row_processing import make_success_result

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


def merge_feature_tables(
    regionprops_features_df: pd.DataFrame,
    hessian_features: pd.DataFrame,
    structure_tensor_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the per-object feature tables on the `label` column.

    Uses right joins, so only objects present in the structure-tensor table
    (and the Hessian table) are kept; objects that vanished in their erosion
    step are dropped from the result.

    Args:
        regionprops_features_df (pd.DataFrame): Output of
            `extract_regionprops_features`.
        hessian_features (pd.DataFrame): Output of `extract_hessian_features`.
        structure_tensor_features (pd.DataFrame): Output of
            `extract_structure_tensor_features`.

    Returns:
        pd.DataFrame: One row per kept object with all feature columns.
    """
    merged_features = regionprops_features_df.merge(
        hessian_features,
        on="label",
        how="right",
    )

    return merged_features.merge(
        structure_tensor_features,
        on="label",
        how="right",
    )


def make_features_output_filename(field_of_view_file: str, config: DictConfig) -> str:
    """Create the file name of the feature table of one field of view.

    The OME suffix is replaced by the output suffix, e.g. `a_bg.ome.tif`
    becomes `a_bg.csv` with the default configuration.

    Args:
        field_of_view_file (str): File name of the corrected field of view.
        config (DictConfig): The whole `feature_extraction` section. Reads
            `features_saving.ome_suffix` and `features_saving.output_suffix`.

    Returns:
        str: Feature table file name without a directory.
    """
    stem = str(field_of_view_file).removesuffix(config.features_saving.ome_suffix)

    return f"{stem}{config.features_saving.output_suffix}"


def save_features_dataframe(
    features_df: pd.DataFrame,
    output_filename: str,
    config: DictConfig,
    output_directory: str | Path,
) -> Path:
    """Save the feature table of one field of view as CSV.

    Creates `output_directory` when it does not exist and overwrites an
    existing file with the same name.

    Args:
        features_df (pd.DataFrame): Merged feature table, usually from
            `merge_feature_tables`.
        output_filename (str): File name, usually from
            `make_features_output_filename`.
        config (DictConfig): The whole `feature_extraction` section. Reads
            `features_saving.save_csv_index`.
        output_directory (str | Path): Target directory, usually
            `shared.paths.feature_tables_dir`.

    Returns:
        Path: Full path of the written CSV file.
    """
    output_path = Path(output_directory) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features_df.to_csv(
        output_path,
        index=config.features_saving.save_csv_index,
    )

    return output_path


def make_feature_success_result(
    row_index: Hashable,
    field_of_view_file: str,
    segmentation_file: str,
    output_file: str,
    config: DictConfig,
) -> dict:
    """Build the result record of one field of view whose features were saved.

    Args:
        row_index (Hashable): Index of the row in the metadata dataframe.
        field_of_view_file (str): File name of the corrected field of view.
        segmentation_file (str): File name of the segmentation mask; stored
            under the extra key `segmentation_file`.
        output_file (str): File name of the saved feature table.
        config (DictConfig): The whole `feature_extraction` section. Reads the
            column names in `metadata.dataframe_columns`,
            `metadata_df_meta_date_format` and `preprocessing_steps` (the
            recorded preprocessing description).

    Returns:
        dict: Result record (see `acid.utils.row_processing`) with the
        extraction date, feature table file name and preprocessing description
        in the metadata columns.
    """
    columns = config.metadata.dataframe_columns

    return make_success_result(
        row_index=row_index,
        input_file=field_of_view_file,
        output_file=output_file,
        metadata_values={
            columns.metadata_df_date_clm_name: datetime.now().strftime(
                columns.metadata_df_meta_date_format
            ),
            columns.metadata_df_file_name_clm_name: output_file,
            columns.metadata_df_method_clm_name: columns.preprocessing_steps,
        },
        segmentation_file=segmentation_file,
    )
