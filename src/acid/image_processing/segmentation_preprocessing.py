"""Prepare background-corrected fields of view for object segmentation.

The segmentation stage (`part5` notebook) feeds the model a two-channel image:
the nucleus channel and the mean of the concanavalin and actin channels, both
median-filtered and downsampled. Channel positions and filter sizes come from
`object_segmentation.processing`.
"""

import logging

import numpy as np
from omegaconf import DictConfig

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def select_segmentation_channels(
    image: np.ndarray, config: DictConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nucleus, concanavalin and actin channels of a field of view.

    Args:
        image (np.ndarray): Field of view with a channel axis, e.g.
            `(channels, y, x)`.
        config (DictConfig): The `object_segmentation.processing` section.
            Reads `channel_axis`, `nucleus_position`, `concanavalin_position`
            and `actin_position` (channel indices along `channel_axis`).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The three channels, each with
        the image shape without the channel axis.

    Raises:
        IndexError: If a configured position is not smaller than the number of
            channels.
    """
    unstacked_image = np.moveaxis(image, config.channel_axis, 0)

    return (
        unstacked_image[config.nucleus_position],
        unstacked_image[config.concanavalin_position],
        unstacked_image[config.actin_position],
    )


def merge_concanavalin_actin_channels(
    concanavalin_channel: np.ndarray, actin_channel: np.ndarray
) -> np.ndarray:
    """Average the concanavalin and actin channels pixel by pixel.

    The merged channel outlines the cell body for the segmentation model.

    Args:
        concanavalin_channel (np.ndarray): Concanavalin channel, `(y, x)`.
        actin_channel (np.ndarray): Actin channel with the same shape.

    Returns:
        np.ndarray: Pixel-wise mean as `float64`, same shape as the inputs.
    """
    return np.mean(np.stack([concanavalin_channel, actin_channel], axis=0), axis=0)
