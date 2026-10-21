"""Measure and flag the image quality of every field of view.

This module is the quality-control stage (`part2` notebook). For each training
field of view it measures, per channel, the power log-log slope (PLLS), the
normalized intensity skewness and the mean-over-standard-deviation ratio, and
flags fields of view whose mean-over-std lies outside configured thresholds.

Functions take the `quality_control` configuration section (`cfg`); each
docstring names the keys it reads.
"""

import logging
import os

import numpy as np
import pandas as pd
import tifffile
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import median_filter

from acid.image_measurement.measure_glob_image_stat import (
    axis_image_stat,
    normalized_intensity_skewness,
)
from acid.image_quality_control.flag_column import add_flag_column
from acid.image_quality_control.measure_mean_over_std import mean_over_std
from acid.image_quality_control.measure_plls import compute_plls
from acid.utils.fov_axis_utils import get_fov_ch_shape

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def flag_quality(
    metadata_df: pd.DataFrame, columns: list[str], cfg: DictConfig
) -> pd.DataFrame:
    """Flag fields of view whose QC values lie outside the thresholds.

    A row is flagged when any of `columns` is above its low-pass threshold or
    below its high-pass threshold (see `add_flag_column`).

    Args:
        metadata_df (pd.DataFrame): Metadata with QC measurement columns,
            usually from `measure_quality`.
        columns (list[str]): QC columns to check, one per channel, e.g. the
            mean-over-std columns returned by `measure_quality`.
        cfg (DictConfig): The `quality_control` section. Reads
            `processing.mean_over_std_lowpass_thresholds` and
            `processing.mean_over_std_highpass_thresholds` (one value per
            column, in column order), `processing.flag_value`,
            `processing.ok_value` and
            `metadata.dataframe_columns.flag_column_name`.

    Returns:
        pd.DataFrame: Metadata with the flag column added.

    Raises:
        ValueError: If the number of thresholds differs from the number of
            columns.
    """
    upper = list(cfg.processing.mean_over_std_lowpass_thresholds)
    lower = list(cfg.processing.mean_over_std_highpass_thresholds)
    if len(upper) != len(columns) or len(lower) != len(columns):
        raise ValueError("QC threshold counts must match the number of channels")
    return add_flag_column(
        df=metadata_df,
        lowpass_thres=upper,
        highpass_thres=lower,
        cols=columns,
        flag_col=cfg.metadata.dataframe_columns.flag_column_name,
        flag_value=cfg.processing.flag_value,
        ok_value=cfg.processing.ok_value,
    )



def measure_quality(
    metadata_df: pd.DataFrame, cfg: DictConfig
) -> tuple[pd.DataFrame, int, list[str], list[str]]:
    """Measure per-channel QC statistics of every field of view.

    For each row the field of view is read from `processing.fov_directory`
    and three statistics are computed per channel: the power log-log slope
    (`compute_plls`), the normalized intensity skewness of the
    (optionally median-smoothed) image and the mean over standard deviation.
    A file that cannot be read gives `null_value` for all three statistics; a
    single failing statistic gives `null_value` for that statistic only. The
    loop continues and every failure is logged as a warning.

    The number of channels is taken from the first readable field of view
    (`get_fov_ch_shape`); all fields of view must have the same number.

    Args:
        metadata_df (pd.DataFrame): Metadata of the fields of view to measure.
        cfg (DictConfig): The `quality_control` section. Reads
            `processing.fov_directory`, `processing.channel_axis`,
            `processing.median_smooth`, `processing.median_size`,
            `processing.median_kwargs_axes_name`, `processing.mask`,
            `processing.normalize`, `processing.ravel_kwargs` and, in
            `metadata.dataframe_columns`, `fov_column_name`, `null_value`,
            `plls_column_name`, `skewness_column_name`,
            `mean_over_std_column_name` and `channel_measurement_separator`.

    Returns:
        tuple[pd.DataFrame, int, list[str], list[str]]: A copy of
        `metadata_df` with columns `<statistic><separator><channel>` added
        (e.g. `plls-0`, `normalized_intensity_skewness-0`,
        `mean_over_std-0`), the number of channels, the mean-over-std column
        names and the skewness column names.
    """
    metadata_df = metadata_df.copy()
    _fov_shape, num_channels, _shape_of_channels = get_fov_ch_shape(
        metadata_df,
        cfg.processing.fov_directory,
        fov_clm=cfg.metadata.dataframe_columns.fov_column_name,
        channel_axis=cfg.processing.channel_axis,
        null_value=cfg.metadata.dataframe_columns.null_value,
    )
    glob_plls_collection_list = []
    glob_skewness_collection_list = []
    glob_mean_over_std_collection_list = []
    for i in metadata_df.index:
        field_of_view_file = metadata_df.loc[
            i, cfg.metadata.dataframe_columns.fov_column_name
        ]
        logger.info("Working on %s", field_of_view_file)
        try:
            field_of_view = tifffile.imread(
                os.path.join(cfg.processing.fov_directory, str(field_of_view_file))
            )
        except Exception as error:
            logger.warning(
                "QC calculation failed: %s", error, exc_info=True
            )
            glob_plls_collection_list.append(
                _null_measurements(num_channels, cfg.metadata.dataframe_columns.null_value)
            )
            glob_skewness_collection_list.append(
                _null_measurements(num_channels, cfg.metadata.dataframe_columns.null_value)
            )
            glob_mean_over_std_collection_list.append(
                _null_measurements(num_channels, cfg.metadata.dataframe_columns.null_value)
            )
            continue
        try:
            fov_plls = compute_plls(
                image=field_of_view, axis=cfg.processing.channel_axis
            )
            glob_plls_collection_list.append(fov_plls)
        except Exception as error:
            logger.warning(
                "QC calculation failed: %s", error, exc_info=True
            )
            glob_plls_collection_list.append(
                _null_measurements(num_channels, cfg.metadata.dataframe_columns.null_value)
            )
        try:
            if cfg.processing.median_smooth:
                median_kwargs = {
                    cfg.processing.median_kwargs_axes_name: tuple(
                        [
                            c
                            for c in range(len(field_of_view.shape))
                            if c != cfg.processing.channel_axis
                        ]
                    )
                }
                smoothed_field_of_view = median_filter(
                    field_of_view, size=cfg.processing.median_size, **median_kwargs
                )
            else:
                smoothed_field_of_view = field_of_view
            fov_skewness = axis_image_stat(
                image=smoothed_field_of_view,
                stat=normalized_intensity_skewness,
                axis=cfg.processing.channel_axis,
                mask=cfg.processing.mask,
                normalize=tuple(cfg.processing.normalize),
                ravel_kwargs=OmegaConf.to_container(
                    cfg.processing.ravel_kwargs, resolve=True
                ),
            )
            glob_skewness_collection_list.append(fov_skewness)
        except Exception as error:
            logger.warning(
                "QC calculation failed: %s", error, exc_info=True
            )
            glob_skewness_collection_list.append(
                _null_measurements(num_channels, cfg.metadata.dataframe_columns.null_value)
            )
        try:
            fov_mean_over_std = mean_over_std(
                image=field_of_view,
                axis=cfg.processing.channel_axis,
                null_val=cfg.metadata.dataframe_columns.null_value,
                inf_val=cfg.metadata.dataframe_columns.null_value,
            )
            glob_mean_over_std_collection_list.append(fov_mean_over_std)
        except Exception as error:
            logger.warning(
                "QC calculation failed: %s", error, exc_info=True
            )
            glob_mean_over_std_collection_list.append(
                _null_measurements(num_channels, cfg.metadata.dataframe_columns.null_value)
            )
    glob_plls = np.stack(glob_plls_collection_list, axis=0)
    glob_skewness = np.stack(glob_skewness_collection_list, axis=0)
    glob_mean_over_std = np.stack(glob_mean_over_std_collection_list, axis=0)
    plls_column_names = [
        f"{cfg.metadata.dataframe_columns.plls_column_name}{cfg.metadata.dataframe_columns.channel_measurement_separator}{ch}"
        for ch in range(num_channels)
    ]
    skewness_column_names = [
        f"{cfg.metadata.dataframe_columns.skewness_column_name}{cfg.metadata.dataframe_columns.channel_measurement_separator}{ch}"
        for ch in range(num_channels)
    ]
    mean_over_std_column_names = [
        f"{cfg.metadata.dataframe_columns.mean_over_std_column_name}{cfg.metadata.dataframe_columns.channel_measurement_separator}{ch}"
        for ch in range(num_channels)
    ]
    metadata_df[plls_column_names] = pd.DataFrame(glob_plls, index=metadata_df.index)
    metadata_df[skewness_column_names] = pd.DataFrame(
        glob_skewness, index=metadata_df.index
    )
    metadata_df[mean_over_std_column_names] = pd.DataFrame(
        glob_mean_over_std, index=metadata_df.index
    )
    return metadata_df, num_channels, mean_over_std_column_names, skewness_column_names


# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------


def _null_measurements(num_channels, null_value):
    return [null_value] * num_channels
