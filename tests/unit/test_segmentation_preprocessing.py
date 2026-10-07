import numpy as np
import pytest

from acid.image_processing.segmentation_preprocessing import (
    merge_concanavalin_actin_channels,
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


def test_merge_concanavalin_actin_channels_averages_pixelwise():
    concanavalin = np.array([[0.0, 2.0], [4.0, 6.0]])
    actin = np.array([[2.0, 2.0], [0.0, 10.0]])

    merged = merge_concanavalin_actin_channels(concanavalin, actin)

    np.testing.assert_array_equal(merged, [[1.0, 2.0], [2.0, 8.0]])
