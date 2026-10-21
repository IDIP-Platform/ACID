"""Load TIFF images used by the pipeline stages.

The stages store fields of view and segmentation masks as OME-TIFF files in
configured directories and refer to them by file name in the metadata
dataframe. These helpers join the two and turn read failures into `OSError`
with the full path, so per-row processing can report which file failed.
"""

import logging
import os
from pathlib import Path

import numpy as np
import tifffile

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def load_tiff(file_path: os.PathLike | str, **kwargs) -> np.ndarray:
    """Load a TIFF image from disk.

    Args:
        file_path (os.PathLike | str): Full path to the TIFF or OME-TIFF file.
        **kwargs: Keyword arguments passed to `tifffile.imread`, e.g. `key`
            to read selected pages only.

    Returns:
        np.ndarray: Image data with the axes and dtype stored in the file.

    Raises:
        FileNotFoundError: If `file_path` does not exist.
    """
    return tifffile.imread(file_path, **kwargs)


def load_field_of_view(
    filename: str, fov_directory: os.PathLike | str, **kwargs
) -> np.ndarray:
    """Load one field-of-view image by file name from a directory.

    Args:
        filename (str): File name of the field of view, as stored in the
            metadata dataframe (e.g. `ome_tif_file_name`).
        fov_directory (os.PathLike | str): Directory containing the fields of
            view, e.g. `shared.paths.extracted_fov_dir` or
            `shared.paths.corrected_fov_dir`.
        **kwargs: Keyword arguments passed to `tifffile.imread`.

    Returns:
        np.ndarray: Field-of-view image, usually `(channels, y, x)` with the
        channel axis given by `processing.channel_axis`.

    Raises:
        OSError: If the file is missing or cannot be read. The message contains
            the full path.
    """
    return _load_from_directory(filename, fov_directory, "field of view", **kwargs)


def load_segmentation_mask(
    filename: str, segmentation_directory: os.PathLike | str, **kwargs
) -> np.ndarray:
    """Load one segmentation mask by file name from a directory.

    Args:
        filename (str): File name of the mask, as stored in the metadata
            dataframe (e.g. `segmentation_file_name`).
        segmentation_directory (os.PathLike | str): Directory containing the
            masks, usually `shared.paths.segmentation_masks_dir`.
        **kwargs: Keyword arguments passed to `tifffile.imread`.

    Returns:
        np.ndarray: Label image of shape `(y, x)`; `0` is background and each
        object has its own positive integer label.

    Raises:
        OSError: If the file is missing or cannot be read. The message contains
            the full path.
    """
    return _load_from_directory(
        filename, segmentation_directory, "segmentation mask", **kwargs
    )


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
