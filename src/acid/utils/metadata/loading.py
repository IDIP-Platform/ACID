import logging
from pathlib import Path

import pandas as pd

from acid.utils.filesystem.filesystem import list_directory_entries
from acid.utils.get_defaults import default_file_name

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def load_metadata(metadata_config: dict) -> tuple[pd.DataFrame, str]:
    """Load a metadata CSV using explicit or default file selection.

    Args:
        metadata_config: Metadata configuration with "directory" and
            "file_selection" entries.

    Returns:
        A tuple containing the loaded dataframe and the resolved metadata file
        name.

    Raises:
        ValueError: If automatic file selection finds no candidate files.
        ValueError: If "date_source" or "select" has an unsupported value.
        FileNotFoundError: If the resolved metadata file does not exist.
    """
    directory = Path(metadata_config["directory"])

    file_selection = metadata_config.get("file_selection", {})

    filename = file_selection.get("filename", "default")

    logger.debug(f"Filename: {filename}")

    if _is_default_filename(filename):
        logger.debug(f"Default option enabled or empty string: {filename}")
        resolved_filename = _resolve_default_metadata_filename(
            directory=directory,
            file_selection=file_selection,
        )
    else:
        resolved_filename = filename

    logger.info(f"Metadata file to load: {resolved_filename}")

    metadata_path = directory / resolved_filename

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    return pd.read_csv(metadata_path), resolved_filename


# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------


def _is_default_filename(filename: str | None) -> bool:
    """Return True if filename requests automatic metadata selection.

    Args:
        filename: Metadata file name or default-selection marker.

    Returns:
        True if filename is None, an empty string, or "default".

    Raises:
        TypeError: If filename is not None or a string.
    """
    if filename is None:
        return True

    if not isinstance(filename, str):
        raise TypeError("filename must be None or a string.")

    return filename == "" or filename.lower() == "default"


def _resolve_default_metadata_filename(
    directory: Path,
    file_selection: dict,
) -> str:
    """Resolve the default metadata filename from selection settings.

    Args:
        directory: Directory containing metadata files.
        file_selection: Configuration dictionary for default file selection.

    Returns:
        Resolved metadata filename.

    Raises:
        ValueError: If no candidate files are found.
        ValueError: If "date_source" or "select" has an unsupported value.
    """
    metadata_files = list_directory_entries(
        directory=directory,
        include=file_selection.get("include", None),
        exclude=file_selection.get("exclude", None),
        files_only=True,
        return_paths=False,
    )

    if not metadata_files:
        raise ValueError(f"No metadata files found in {directory}")

    return default_file_name(
        file_list=metadata_files,
        from_file_name=_use_filename_date(file_selection),
        directory_path=directory,
        separator=file_selection.get("filename_date_separator", "_"),
        date_position=file_selection.get("filename_date_position", 0),
        date_format=file_selection.get("filename_date_format", "%Y%m%d"),
        reverse=_select_newest(file_selection),
    )


def _use_filename_date(file_selection: dict) -> bool:
    """Return whether default selection should parse dates from filenames.

    Args:
        file_selection: Configuration dictionary for default file selection.

    Returns:
        True if dates should be parsed from filenames. False if filesystem
        modification time should be used.

    Raises:
        ValueError: If "date_source" has an unsupported value.
    """
    date_source = file_selection.get("date_source", "modified_time")

    if date_source == "filename":
        return True

    if date_source == "modified_time":
        return False

    raise ValueError(
        f'Invalid date_source {date_source!r}. Expected "filename" or "modified_time".'
    )


def _select_newest(file_selection: dict) -> bool:
    """Return whether default selection should choose the newest file.

    Args:
        file_selection: Configuration dictionary for default file selection.

    Returns:
        True if the `newest` file should be selected. False if the `oldest` file
        should be selected.

    Raises:
        ValueError: If "select" has an unsupported value.
    """
    select = file_selection.get("select", "newest")

    if select == "newest":
        return True

    if select == "oldest":
        return False

    raise ValueError('"select" must be either "newest" or "oldest".')
