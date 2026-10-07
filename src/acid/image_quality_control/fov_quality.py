"""Measure and flag the image quality of every field of view.

This module is the quality-control stage (`part2` notebook). For each training
field of view it measures, per channel, the power log-log slope (PLLS), the
normalized intensity skewness and the mean-over-standard-deviation ratio, and
flags fields of view whose mean-over-std lies outside configured thresholds.

Functions take the `quality_control` configuration section (`cfg`); each
docstring names the keys it reads.
"""

import logging

import pandas as pd
from omegaconf import DictConfig

from acid.image_quality_control.flag_column import add_flag_column

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

