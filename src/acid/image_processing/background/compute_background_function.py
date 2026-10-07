"""Estimate, fit and save background functions from unflagged fields of view.

This module is the background-function stage (`part3` notebook). It averages
stacks of unflagged training fields of view into one background per group
(the whole dataset, each well, or each grid position), smooths or fits it,
saves it as an OME-TIFF for the background-correction stage and records the
settings in the metadata.

Functions take the `background_function_calculation` configuration section
(`cfg`); each docstring names the keys it reads.
"""

import logging
from collections.abc import Sequence
from datetime import datetime

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def update_background_metadata(metadata_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Record the background-function settings in every metadata row.

    Adds the processing date, the averaging method and the fit parameters.
    List-valued parameters (one value per channel) become one column per
    channel, `<name><channel_name_separator><channel>`.

    Args:
        metadata_df (pd.DataFrame): Metadata to annotate (all rows, not only
            the ones used for the background).
        cfg (DictConfig): The `background_function_calculation` section. Reads
            `processing.background_function_avg_method`,
            `processing.background_fit_method` and, for `"polyfit"`,
            `processing.polynomial_order_x`/`_y`, otherwise
            `processing.ball_radius`, `white_background` and `gau_smooth`;
            column names and `background_df_meta_date_format`,
            `channel_name_separator` from `metadata.dataframe_columns`.

    Returns:
        pd.DataFrame: Copy of `metadata_df` with the background columns added.
    """
    updated = metadata_df.copy()
    proc, columns = cfg.processing, cfg.metadata.dataframe_columns
    updated[columns.background_df_date_clm_name] = (
        datetime.now()
        .astimezone()
        .strftime(columns.background_df_meta_date_format)
    )
    updated[columns.background_df_avg_method_clm_name] = (
        proc.background_function_avg_method
    )
    params = OmegaConf.to_container(proc, resolve=True)
    if proc.background_fit_method == "polyfit":
        fields = [
            (columns.background_df_poly_order_x_clm_name, params["polynomial_order_x"]),
            (columns.background_df_poly_order_y_clm_name, params["polynomial_order_y"]),
        ]
    else:
        fields = [
            (columns.background_df_meta_ballradius_name, params["ball_radius"]),
            (columns.background_df_meta_whitebg_name, params["white_background"]),
            (columns.background_df_meta_gausmooth_name, params["gau_smooth"]),
        ]
    for name, value in fields:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for channel, item in enumerate(value):
                updated[f"{name}{columns.channel_name_separator}{channel}"] = item
        else:
            updated[name] = value
    return updated

