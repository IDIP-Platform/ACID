"""Prepare background-corrected fields of view for object segmentation.

The segmentation stage (`part5` notebook) feeds the model a two-channel image:
the nucleus channel and the mean of the concanavalin and actin channels, both
median-filtered and downsampled. Channel positions and filter sizes come from
`object_segmentation.processing`.
"""

import logging

import numpy as np
from omegaconf import DictConfig

from acid.image_processing.filter_image import median_filter_image
from acid.image_processing.resize_image import downsample_local_mean

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


def preprocess_image_for_segmentation(
    image: np.ndarray, config: DictConfig
) -> np.ndarray:
    """Build the two-channel, downsampled input image for the segmentation model.

    Steps: select the nucleus, concanavalin and actin channels; average
    concanavalin and actin; median-filter the nucleus and the merged channel;
    stack them along `channel_axis`; downsample by local mean.

    Args:
        image (np.ndarray): Background-corrected field of view, e.g.
            `(channels, y, x)`.
        config (DictConfig): The `object_segmentation.processing` section.
            Reads `channel_axis`, `nucleus_position`, `concanavalin_position`,
            `actin_position`, `med_filter_nucleus`,
            `med_filter_concactin_merge` (median filter sizes) and
            `downsampling_factor`.

    Returns:
        np.ndarray: Image with two channels (nucleus, merged concanavalin and
        actin) on `channel_axis` and spatial axes divided by
        `downsampling_factor`, e.g. `(2, y / f, x / f)`.
    """
    nucleus_channel, concanavalin_channel, actin_channel = select_segmentation_channels(
        image=image,
        config=config,
    )

    concactin_merge = merge_concanavalin_actin_channels(
        concanavalin_channel=concanavalin_channel,
        actin_channel=actin_channel,
    )

    med_nucleus = median_filter_image(
        nucleus_channel,
        size=config.med_filter_nucleus,
    )

    med_concactin = median_filter_image(
        concactin_merge,
        size=config.med_filter_concactin_merge,
    )

    restacked_image = np.stack(
        [med_nucleus, med_concactin],
        axis=config.channel_axis,
    )

    return downsample_local_mean(
        restacked_image,
        factor=config.downsampling_factor,
        channel_axis=config.channel_axis,
    )
