import logging
from datetime import datetime
from pathlib import Path

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def build_metadata_dataframe_filename(
    metadata_saving_config, project_name: str, timestamp: datetime | None = None
) -> str:
    """Build the standard ACID metadata dataframe filename.

    Args:
        metadata_saving_config: The ``metadata.saving`` section of a stage.
        project_name: Project name inserted into the filename.
        timestamp: Time used for the date prefix; defaults to now.

    Returns:
        A filename such as ``20261006_proj_metadata_part_4.csv``.
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
    metadata_config, project_name: str, timestamp: datetime | None = None
) -> Path:
    """Build the full save path for a metadata dataframe."""
    metadata_filename = build_metadata_dataframe_filename(
        metadata_saving_config=metadata_config.saving,
        project_name=project_name,
        timestamp=timestamp,
    )

    return Path(metadata_config.directory) / metadata_filename
