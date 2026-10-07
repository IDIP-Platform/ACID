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
from collections.abc import Hashable, Sequence
from datetime import datetime

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from acid.image_processing.calculate_background_function import (
    calculate_background_function,
    compute_simple_background,
    get_polyfit_bg_funct_channel,
    import_fov,
)

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



def fit_background(background: np.ndarray, cfg: DictConfig) -> np.ndarray:
    """Smooth or fit an averaged background function.

    With `"polyfit"` a 2D polynomial is fitted to every channel
    (`get_polyfit_bg_funct_channel`); with `"simple"` every channel is
    flattened with a rolling-ball background estimate and Gaussian smoothing
    (`compute_simple_background`), in parallel over `n_workers` processes.

    Args:
        background (np.ndarray): Averaged background, e.g. `(channels, y, x)`,
            from `calculate_backgrounds`.
        cfg (DictConfig): The `background_function_calculation` section. Reads
            `processing.background_fit_method` (`"polyfit"` or `"simple"`),
            `processing.channel_axis`, `processing.verbose_calc_bg`; for
            `"polyfit"` `processing.polynomial_order_x`/`_y` (one order per
            channel); for `"simple"` `processing.ball_radius`,
            `white_background` (one flag per channel), `rolling_ball_kwargs`,
            `invert_kwargs`, `gau_smooth`, `gaussian_kwargs`,
            `convolve_kwargs`, `dtype`, `n_workers` and `map_kwargs`.

    Returns:
        np.ndarray: Fitted background with the shape of `background`.

    Raises:
        ValueError: If `background_fit_method` is neither `"polyfit"` nor
            `"simple"`.
    """
    proc = cfg.processing
    if proc.background_fit_method == "polyfit":
        return get_polyfit_bg_funct_channel(
            background_function=background,
            channel_axis=proc.channel_axis,
            kx=tuple(proc.polynomial_order_x),
            ky=tuple(proc.polynomial_order_y),
            verbose=proc.verbose_calc_bg,
        )
    if proc.background_fit_method != "simple":
        raise ValueError("background_fit_method must be polyfit or simple")
    params = OmegaConf.to_container(proc, resolve=True)
    return compute_simple_background(
        background,
        ball_radius=params["ball_radius"],
        white_background=params["white_background"],
        _rb_kwargs=params["rolling_ball_kwargs"],
        invert_kwargs=params["invert_kwargs"],
        gau_smooth=params["gau_smooth"],
        gaussian_kwargs=params["gaussian_kwargs"],
        convolve_kwargs=params["convolve_kwargs"],
        dtype=params["dtype"],
        axis=proc.channel_axis,
        n_workers=proc.n_workers,
        map_kwargs=params["map_kwargs"],
    )



def calculate_backgrounds(
    metadata_df: pd.DataFrame, fov_shape: tuple[int, ...], cfg: DictConfig
) -> dict[Hashable | None, np.ndarray]:
    """Average fields of view into one background function per group.

    The grouping follows `processing.background_function_strategy`:
    `1` uses all rows as one group (key `None`), `2` groups by well and `3`
    by grid position. For every group the fields of view are stacked
    (optionally sampled) and averaged pixel-wise.

    Args:
        metadata_df (pd.DataFrame): Rows to use, usually unflagged training
            fields of view.
        fov_shape (tuple[int, ...]): Shape of one field of view, e.g.
            `(channels, y, x)`, from `get_fov_ch_shape`.
        cfg (DictConfig): The `background_function_calculation` section. Reads
            `processing.background_function_strategy`,
            `processing.fov_directory`, `processing.axis_calc_bg` (must be
            `-1` or `len(fov_shape)`, the appended stack axis),
            `processing.background_function_avg_method` (e.g. `"median"`),
            `processing.verbose_calc_bg`, `processing.np_zero_kwargs`,
            `processing.sample_df`, `processing.sample_fraction`,
            `processing.sample_kwargs` and, in `metadata.dataframe_columns`,
            `fov_column_name`, `well_column_name` and `gridpos_column_name`.

    Returns:
        dict[Hashable | None, np.ndarray]: Averaged background with the shape
        `fov_shape` for each group key.

    Raises:
        ValueError: If the strategy is not 1, 2 or 3, a grouping value is
            missing, `axis_calc_bg` is invalid, or sampling selects no image
            for a group.
    """
    proc = cfg.processing
    columns = cfg.metadata.dataframe_columns
    strategy = proc.background_function_strategy
    if strategy == 1:
        groups = [(None, metadata_df)]
    elif strategy in (2, 3):
        column = (
            columns.well_column_name if strategy == 2 else columns.gridpos_column_name
        )
        if metadata_df[column].isna().any():
            raise ValueError(f"Missing background grouping values in {column}")
        groups = metadata_df.groupby(column, sort=False)
    else:
        raise ValueError("background_function_strategy must be 1, 2 or 3")
    if proc.axis_calc_bg not in (-1, len(fov_shape)):
        raise ValueError("axis_calc_bg must select the appended image-stack axis")
    backgrounds = {}
    for key, rows in groups:
        stack = import_fov(
            df=rows,
            fov_dir=proc.fov_directory,
            fov_clm=columns.fov_column_name,
            fov_shape=fov_shape,
            verbose=proc.verbose_calc_bg,
            np_zero_kwargs=OmegaConf.to_container(proc.np_zero_kwargs)
            if proc.np_zero_kwargs is not None
            else None,
            sample_df=proc.sample_df,
            sample_fraction=proc.sample_fraction,
            sample_kwargs=OmegaConf.to_container(proc.sample_kwargs),
        )
        if stack.shape[-1] == 0:
            raise ValueError(f"Sampling selected no images for background group {key}")
        backgrounds[key] = calculate_background_function(
            stack,
            method=proc.background_function_avg_method,
            axis=-1,
            verbose=proc.verbose_calc_bg,
        )
    return backgrounds

