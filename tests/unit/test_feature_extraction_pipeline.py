import pytest

from acid.feature_extraction.pipeline import get_feature_metadata_columns


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
