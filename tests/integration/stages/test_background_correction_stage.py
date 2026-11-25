import types

import numpy as np
import pandas as pd

from acid.image_processing.background.apply_background_correction import (
    apply_background_correction_batch,
    apply_background_correction_for_fov,
    update_metadata_with_correction_results,
)


def background():
    return np.full((5, 16, 16), 1000.0)


def test_apply_for_fov_writes_corrected_image(correction_config, write_fov):
    directory, _ = write_fov()
    columns = correction_config.metadata.dataframe_columns
    row = pd.Series({columns.fov_column_name: "a.ome.tif"})

    result = apply_background_correction_for_fov(
        0,
        row,
        background(),
        correction_config,
        types.SimpleNamespace(extracted_fov_dir=str(directory)),
    )

    assert result["success"] is True, result["error_message"]
    assert result["output_file"] == "a_bg.ome.tif"
    assert result[columns.illum_correct_df_method_clm_name] == "division"
    assert (directory.parent / "corrected" / "a_bg.ome.tif").is_file()


def test_apply_for_fov_missing_file_reports_stage(correction_config, tmp_path):
    columns = correction_config.metadata.dataframe_columns
    row = pd.Series({columns.fov_column_name: "missing.ome.tif"})

    result = apply_background_correction_for_fov(
        0,
        row,
        background(),
        correction_config,
        types.SimpleNamespace(extracted_fov_dir=str(tmp_path)),
    )

    assert result["success"] is False
    assert result["stage"] == "load_field_of_view"
    assert result["input_file"] == "missing.ome.tif"
    assert pd.isna(result[columns.illum_correct_df_file_name_clm_name])


def test_apply_for_fov_empty_filename_reports_stage(correction_config, tmp_path):
    columns = correction_config.metadata.dataframe_columns
    row = pd.Series({columns.fov_column_name: np.nan})

    result = apply_background_correction_for_fov(
        0,
        row,
        background(),
        correction_config,
        types.SimpleNamespace(extracted_fov_dir=str(tmp_path)),
    )

    assert result["stage"] == "get_field_of_view_file"
    assert result["input_file"] is None


def test_batch_returns_one_result_per_row(correction_config, write_fov):
    directory, _ = write_fov()
    columns = correction_config.metadata.dataframe_columns
    metadata_df = pd.DataFrame(
        {columns.fov_column_name: ["a.ome.tif", "missing.ome.tif"]}, index=[7, 8]
    )

    results = apply_background_correction_batch(
        metadata_df,
        background(),
        correction_config,
        types.SimpleNamespace(extracted_fov_dir=str(directory)),
    )

    assert [result["row_index"] for result in results] == [7, 8]
    assert [result["success"] for result in results] == [True, False]


def test_update_metadata_with_correction_results_fills_columns_per_row(
    correction_config, write_fov
):
    directory, _ = write_fov()
    columns = correction_config.metadata.dataframe_columns
    metadata_df = pd.DataFrame(
        {columns.fov_column_name: ["a.ome.tif", "missing.ome.tif"]}
    )
    results = apply_background_correction_batch(
        metadata_df,
        background(),
        correction_config,
        types.SimpleNamespace(extracted_fov_dir=str(directory)),
    )

    updated = update_metadata_with_correction_results(metadata_df, results, columns)

    file_column = columns.illum_correct_df_file_name_clm_name
    assert updated.loc[0, file_column] == "a_bg.ome.tif"
    assert pd.isna(updated.loc[1, file_column])
