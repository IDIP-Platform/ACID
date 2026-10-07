import numpy as np
import pytest

from acid.feature_extraction.pipeline import (
    get_feature_metadata_columns,
    preprocess_field_of_view,
    preprocess_segmentation_mask,
)


@pytest.fixture
def feature_config(stage_config):
    return stage_config("feature_extraction")


def test_get_feature_metadata_columns(feature_config):
    columns = get_feature_metadata_columns(feature_config.metadata.dataframe_columns)

    assert columns == [
        "features_extraction_date",
        "features_data_frame",
        "features_extraction_preprocessing",
    ]


def test_preprocess_field_of_view_moves_channels_last(feature_config):
    image = np.random.default_rng(0).random((5, 8, 6))

    result = preprocess_field_of_view(image, feature_config)

    assert result.shape == (8, 6, 5)


def test_preprocess_segmentation_mask_removes_edge_objects():
    mask = np.zeros((6, 6), dtype=np.uint16)
    mask[0:2, 0:2] = 1
    mask[2:4, 2:4] = 2

    result = preprocess_segmentation_mask(mask)

    assert set(np.unique(result)) == {0, 2}
    assert mask[0, 0] == 1
