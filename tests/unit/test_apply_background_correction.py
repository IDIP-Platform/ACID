import pytest

from acid.image_processing.background.apply_background_correction import (
    get_correction_metadata_columns,
)


@pytest.fixture
def correction_config(stage_config, tmp_path):
    config = stage_config("background_correction")
    config.image_saving.directory = str(tmp_path / "corrected")
    config.background_function_selection.background_function_strategy = 1
    return config


def test_get_correction_metadata_columns_reads_names_from_config(correction_config):
    columns = get_correction_metadata_columns(correction_config.metadata.dataframe_columns)

    assert columns[:2] == [
        "illumination_correction_date",
        "illumination_correction_file_name",
    ]
    assert len(columns) == 9
