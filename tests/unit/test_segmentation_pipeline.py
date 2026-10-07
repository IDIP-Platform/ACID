import numpy as np
import pytest

from acid.segmentation.pipeline import (
    get_channel_shape,
    get_segmentation_metadata_columns,
)


@pytest.fixture
def segmentation_config(stage_config, tmp_path):
    config = stage_config("object_segmentation")
    config.image_saving.directory = str(tmp_path / "masks")
    return config


def test_get_segmentation_metadata_columns_has_twelve_names(segmentation_config):
    columns = get_segmentation_metadata_columns(
        segmentation_config.metadata.dataframe_columns
    )

    assert len(columns) == 12
    assert len(set(columns)) == 12


def test_get_channel_shape_drops_channel_axis():
    assert get_channel_shape(np.zeros((5, 16, 12)), channel_axis=0) == (16, 12)
    assert get_channel_shape(np.zeros((16, 12, 5)), channel_axis=2) == (16, 12)
