"""Load TIFF images used by the pipeline stages."""

import logging
from pathlib import Path

import tifffile

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def load_tiff(file_path, **kwargs):
    """Load a TIFF image from disk.

    Args:
        file_path: Path to the TIFF file.
        **kwargs: Passed to ``tifffile.imread``.

    Returns:
        The image as a NumPy array.
    """
    return tifffile.imread(file_path, **kwargs)


def load_field_of_view(filename, fov_directory, **kwargs):
    """Load one field-of-view TIFF from a directory.

    Raises:
        OSError: If the file cannot be read.
    """
    return _load_from_directory(filename, fov_directory, "field of view", **kwargs)


# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------


def _load_from_directory(filename, directory, description, **kwargs):
    file_path = Path(directory) / str(filename)
    logger.debug("Load %s from: %s", description, file_path)

    try:
        return load_tiff(file_path, **kwargs)
    except Exception as error:
        raise OSError(f"Could not load {description} TIFF: {file_path}") from error
