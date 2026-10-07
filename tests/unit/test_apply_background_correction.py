import numpy as np
import pandas as pd
import pytest

from acid.image_processing.background.apply_background_correction import (
    correct_background_image,
    get_background_for_fov,
    get_correction_metadata_columns,
    make_output_filename,
)


def background():
    return np.full((5, 16, 16), 1000.0)


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


def test_get_background_for_fov_selects_by_strategy(correction_config):
    selection = correction_config.background_function_selection
    row = pd.Series({selection.well_column_name: "A1", selection.gridpos_column_name: 2})

    selection.background_function_strategy = 1
    assert get_background_for_fov(row, "dataset", selection) == "dataset"
    selection.background_function_strategy = 2
    assert get_background_for_fov(row, {"A1": "well"}, selection) == "well"
    selection.background_function_strategy = 3
    assert get_background_for_fov(row, {2: "grid"}, selection) == "grid"
    selection.background_function_strategy = 4
    with pytest.raises(ValueError, match="Invalid background_function_strategy"):
        get_background_for_fov(row, None, selection)


def test_make_output_filename_inserts_savingword(correction_config):
    assert make_output_filename("a.ome.tif", correction_config.image_saving) == "a_bg.ome.tif"


def test_correct_background_image_divides_by_background(correction_config, write_fov):
    _, image = write_fov()

    corrected = correct_background_image(
        image, background(), correction_config.processing
    )

    assert corrected.shape == image.shape
    assert corrected.dtype == np.dtype(correction_config.processing.output_dtype)
