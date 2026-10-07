import numpy as np
import pytest

from acid.image_processing.segmentation_preprocessing import (
    select_segmentation_channels,
)


@pytest.fixture
def processing_config(stage_config):
    return stage_config("object_segmentation").processing


def channel_image(shape=(5, 16, 16)):
    """Image whose channel `c` is filled with the value `c`."""
    return np.stack([np.full(shape[1:], channel) for channel in range(shape[0])])


def test_select_segmentation_channels_returns_configured_positions(processing_config):
    nucleus, concanavalin, actin = select_segmentation_channels(
        channel_image(), processing_config
    )

    assert nucleus.shape == (16, 16)
    assert nucleus[0, 0] == processing_config.nucleus_position
    assert concanavalin[0, 0] == processing_config.concanavalin_position
    assert actin[0, 0] == processing_config.actin_position
