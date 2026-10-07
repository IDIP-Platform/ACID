"""Save metadata dataframes under the standard ACID file name.

Every stage writes its updated metadata as
`<date><sep><project><sep><savingword><sep><suffix>`, e.g.
`20261007_ACID_metadata_part_4b.csv`, into the stage's `metadata.directory`.
The next stage finds it through its `metadata.file_selection`.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def build_metadata_dataframe_filename(
    metadata_saving_config: DictConfig,
    project_name: str,
    timestamp: datetime | None = None,
) -> str:
    """Build the standard ACID metadata dataframe file name.

    Args:
        metadata_saving_config (DictConfig): The `<stage>.metadata.saving`
            section. Reads `save_file_name_separator`, `metadata_date_format`,
            `metadata_savingword` and `metadata_file_suffix`; the suffix may
            contain the placeholder `{save_file_name_separator}`.
        project_name (str): Project name, usually
            `project_identity.project_name`.
        timestamp (datetime | None): Time used for the date prefix. If `None`,
            the current local time is used.

    Returns:
        str: File name such as `20261007_ACID_metadata_part_4b.csv`.
    """
    if timestamp is None:
        timestamp = datetime.now()

    separator = metadata_saving_config.save_file_name_separator
    metadata_file_suffix = metadata_saving_config.metadata_file_suffix.format(
        save_file_name_separator=separator
    )

    return separator.join(
        [
            timestamp.strftime(metadata_saving_config.metadata_date_format),
            project_name,
            metadata_saving_config.metadata_savingword,
            metadata_file_suffix,
        ]
    )


def build_metadata_dataframe_path(
    metadata_config: DictConfig,
    project_name: str,
    timestamp: datetime | None = None,
) -> Path:
    """Build the full save path of a stage's metadata dataframe.

    Args:
        metadata_config (DictConfig): The `<stage>.metadata` section. Reads
            `directory` and the `saving` subsection (see
            `build_metadata_dataframe_filename`).
        project_name (str): Project name inserted into the file name.
        timestamp (datetime | None): Time used for the date prefix; `None`
            means now.

    Returns:
        Path: `metadata_config.directory / <standard file name>`.
    """
    metadata_filename = build_metadata_dataframe_filename(
        metadata_saving_config=metadata_config.saving,
        project_name=project_name,
        timestamp=timestamp,
    )

    return Path(metadata_config.directory) / metadata_filename


def save_metadata_dataframe(
    metadata_df: pd.DataFrame,
    metadata_config: DictConfig,
    project_name: str,
    timestamp: datetime | None = None,
) -> Path:
    """Save a metadata dataframe as CSV under the standard file name.

    Creates `metadata_config.directory` when it does not exist. A file with the
    same name (same stage, same day) is overwritten.

    Args:
        metadata_df (pd.DataFrame): Metadata to save.
        metadata_config (DictConfig): The `<stage>.metadata` section. Reads
            `directory` and `saving`; `saving.save_csv_index` decides whether
            the dataframe index is written.
        project_name (str): Project name inserted into the file name.
        timestamp (datetime | None): Time used for the date prefix; `None`
            means now.

    Returns:
        Path: Full path of the written CSV file.
    """
    metadata_path = build_metadata_dataframe_path(
        metadata_config=metadata_config,
        project_name=project_name,
        timestamp=timestamp,
    )

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(metadata_path, index=metadata_config.saving.save_csv_index)
    logger.info("Saved metadata: %s", metadata_path)

    return metadata_path
