import numpy as np
import pytest

from acid.feature_extraction.pipeline import (
    extract_hessian_features,
    extract_regionprops_features,
    extract_structure_tensor_features,
    get_feature_metadata_columns,
    preprocess_field_of_view,
    preprocess_segmentation_mask,
)


def two_object_label_image():
    label_image = np.zeros((10, 10), dtype=np.uint16)
    label_image[2:4, 2:4] = 1
    label_image[5:8, 5:8] = 2
    return label_image


def large_object_label_image():
    """Objects large enough to survive the radius-9 erosion of the eigenvalue features."""
    label_image = np.zeros((100, 100), dtype=np.uint16)
    label_image[5:35, 5:35] = 1
    label_image[50:90, 50:90] = 2
    return label_image


def random_intensity(shape=(100, 100, 3)):
    return np.random.default_rng(0).random(shape)


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


def test_extract_regionprops_features_returns_one_row_per_object():
    intensity = np.ones((10, 10, 3))

    features = extract_regionprops_features(
        two_object_label_image(), intensity, ["label", "area"], []
    )

    assert features["label"].tolist() == [1, 2]
    assert features["area"].tolist() == [4, 9]


def test_extract_hessian_features_returns_one_row_per_object():
    features = extract_hessian_features(large_object_label_image(), random_intensity())

    assert features["label"].tolist() == [1, 2]
    assert "hessian_eigv_1_intensity_mean-0" in features.columns


def test_extract_structure_tensor_features_returns_one_row_per_object():
    features = extract_structure_tensor_features(
        large_object_label_image(), random_intensity()
    )

    assert features["label"].tolist() == [1, 2]
    assert "str_tens_eigv_1_intensity_mean-0" in features.columns
