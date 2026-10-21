"""Select the rows of a stage and store per-field-of-view results.

The Snakemake workflow runs one job per field of view (FOV). Every job and
the merge job that follows it must agree on which metadata rows a stage
processes and on how a FOV is named, so that the result records can be
matched back to their rows. This module holds those shared rules:

- a FOV is named by its extracted file name without the OME suffix, e.g.
  `H7_DENV2_MOI1_40h_fixed_stained_well6_A07p4_G6` (`fov_name`); the file
  name is read from `quality_control.metadata.dataframe_columns.fov_column_name`,
  which every later metadata file keeps,
- `select_stage_rows` applies the same split filter as the stage notebook,
- one result record per FOV is written as JSON (`write_record`). Its
  `row_index` is the row's index in the selected metadata, as in the batch
  functions.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from acid.utils.metadata.filtering import filter_metadata_by_splits

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

# Stages whose notebook keeps only the training split; the others use every row.
TRAINING_SPLIT_STAGES = (
    "quality_control",
    "background_function_calculation",
    "background_correction",
    "object_segmentation",
)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def fov_column(config: DictConfig) -> str:
    """Return the metadata column holding the extracted FOV file name.

    Args:
        config (DictConfig): The whole configuration. Reads
            `quality_control.metadata.dataframe_columns.fov_column_name`.

    Returns:
        str: Column name, `ome_tif_file_name` by default.
    """
    return config.quality_control.metadata.dataframe_columns.fov_column_name


def fov_name(field_of_view_file: str, config: DictConfig) -> str:
    """Return the FOV name used as Snakemake wildcard and record file name.

    Args:
        field_of_view_file (str): Extracted FOV file name, e.g. `a_A1.ome.tif`.
        config (DictConfig): The whole configuration. Reads
            `shared.naming.ome_suffix`.

    Returns:
        str: The file name without the OME suffix, e.g. `a_A1`.

    Raises:
        ValueError: If the file name does not end with the OME suffix.
    """
    suffix = config.shared.naming.ome_suffix
    if not field_of_view_file.endswith(suffix):
        raise ValueError(f"{field_of_view_file!r} does not end with {suffix!r}")
    return field_of_view_file[: -len(suffix)]


def select_stage_rows(
    metadata_df: pd.DataFrame, stage: str, config: DictConfig
) -> pd.DataFrame:
    """Select the metadata rows a stage processes, as its notebook does.

    Stages in `TRAINING_SPLIT_STAGES` keep the training split
    (`filter_metadata_by_splits` with `<stage>.dataset_split`); feature
    extraction keeps every row of its input. The index is left unchanged.

    Args:
        metadata_df (pd.DataFrame): Input metadata of the stage.
        stage (str): Configuration section name of the stage, e.g.
            `"background_correction"`.
        config (DictConfig): The whole configuration.

    Returns:
        pd.DataFrame: A copy of the selected rows.
    """
    if stage in TRAINING_SPLIT_STAGES:
        return filter_metadata_by_splits(
            metadata_df,
            selected_splits="train",
            split_config=config[stage].dataset_split,
        )
    return metadata_df.copy()


def list_fov_names(metadata_df: pd.DataFrame, config: DictConfig) -> list[str]:
    """Return the FOV names of the metadata rows, in row order.

    Args:
        metadata_df (pd.DataFrame): Metadata with the FOV file name column
            (see `fov_column`).
        config (DictConfig): The whole configuration.

    Returns:
        list[str]: One name per row (see `fov_name`).

    Raises:
        ValueError: If two rows have the same FOV name.
    """
    names = [fov_name(str(name), config) for name in metadata_df[fov_column(config)]]
    if len(set(names)) != len(names):
        raise ValueError("FOV file names in the metadata are not unique")
    return names


def select_fov_row(
    metadata_df: pd.DataFrame, fov: str, config: DictConfig
) -> pd.DataFrame:
    """Return the one-row dataframe of a FOV, keeping its index.

    Args:
        metadata_df (pd.DataFrame): Selected metadata of the stage.
        fov (str): FOV name (see `fov_name`).
        config (DictConfig): The whole configuration.

    Returns:
        pd.DataFrame: The row of `fov` as a one-row dataframe.

    Raises:
        KeyError: If the FOV is not in the metadata exactly once.
    """
    names = pd.Series(list_fov_names(metadata_df, config), index=metadata_df.index)
    matches = names.index[names == fov]
    if len(matches) != 1:
        raise KeyError(f"FOV {fov!r} not found in the metadata")
    return metadata_df.loc[matches]


def write_record(record: dict, path: str | Path) -> Path:
    """Write one result record as JSON.

    NumPy scalars and arrays are converted to Python numbers and lists; `NaN`
    is written as JSON `NaN`, which `read_record` reads back.

    Args:
        record (dict): Result record, e.g. from `make_success_result`.
        path (str | Path): Output file; its directory is created.

    Returns:
        Path: The written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, default=_to_json, indent=1))
    return path


def read_record(path: str | Path) -> dict:
    """Read a result record written by `write_record`.

    Args:
        path (str | Path): Record file.

    Returns:
        dict: The record.
    """
    return json.loads(Path(path).read_text())


def read_records(paths: list[str | Path]) -> list[dict]:
    """Read several result records.

    Args:
        paths (list[str | Path]): Record files.

    Returns:
        list[dict]: The records, in the order of `paths`.
    """
    return [read_record(path) for path in paths]


# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------


def _to_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is pd.NA:
        return None
    raise TypeError(f"Cannot write {type(value).__name__} to a result record")
