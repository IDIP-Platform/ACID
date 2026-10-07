"""Load TIFF images used by the pipeline stages."""

import logging

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
