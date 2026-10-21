"""Select the rows of a stage, split them into batches and store results.

The Snakemake workflow processes the fields of view (FOVs) of a stage in
batches, one job per batch. Every batch job and the merge job that follows
must agree on which metadata rows a stage processes, how a FOV is named and
which FOVs form a batch, so that the result records can be matched back to
their rows. This module holds those shared rules:

- a FOV is named by its extracted file name without the OME suffix, e.g.
  `H7_DENV2_MOI1_40h_fixed_stained_well6_A07p4_G6` (`fov_name`); the file
  name is read from `quality_control.metadata.dataframe_columns.fov_column_name`,
  which every later metadata file keeps,
- `select_stage_rows` applies the same split filter as the stage notebook,
- `make_batches` splits the FOV list, in row order, into batches,
- the result records of a batch (one per FOV) are written as one JSON list
  (`write_records`). A record's `row_index` is the row's index in the
  selected metadata, as in the notebooks' batch functions.
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


def select_fov_rows(
    metadata_df: pd.DataFrame, fovs: list[str], config: DictConfig
) -> pd.DataFrame:
    """Return the rows of some FOVs, in the order of `fovs`, keeping the index.

    Args:
        metadata_df (pd.DataFrame): Selected metadata of the stage.
        fovs (list[str]): FOV names (see `fov_name`).
        config (DictConfig): The whole configuration.

    Returns:
        pd.DataFrame: One row per FOV.

    Raises:
        KeyError: If a FOV is not in the metadata.
    """
    index_by_name = dict(
        zip(list_fov_names(metadata_df, config), metadata_df.index, strict=True)
    )
    missing = [fov for fov in fovs if fov not in index_by_name]
    if missing:
        raise KeyError(f"FOVs not found in the metadata: {missing}")
    return metadata_df.loc[[index_by_name[fov] for fov in fovs]]


def make_batches(fovs: list[str], batch_size: int) -> list[list[str]]:
    """Split FOV names into consecutive batches.

    Args:
        fovs (list[str]): FOV names in row order.
        batch_size (int): Maximum number of FOVs per batch (at least 1).

    Returns:
        list[list[str]]: Batches in order; only the last one may be shorter.

    Raises:
        ValueError: If `batch_size` is smaller than 1.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    return [
        fovs[start : start + batch_size] for start in range(0, len(fovs), batch_size)
    ]


def write_records(records: list[dict], path: str | Path) -> Path:
    """Write the result records of a batch as one JSON list.

    NumPy scalars and arrays are converted to Python numbers and lists; `NaN`
    is written as JSON `NaN`, which `read_records` reads back.

    Args:
        records (list[dict]): Result records, e.g. from `make_success_result`.
        path (str | Path): Output file; its directory is created.

    Returns:
        Path: The written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(records), default=_to_json, indent=1))
    return path


def read_records(paths: list[str | Path]) -> list[dict]:
    """Read the records of several batch files.

    Args:
        paths (list[str | Path]): Files written by `write_records`.

    Returns:
        list[dict]: All records, in file order and, within a file, in batch
        order.
    """
    return [record for path in paths for record in json.loads(Path(path).read_text())]


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
