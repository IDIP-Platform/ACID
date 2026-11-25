import types

import numpy as np
import pandas as pd
import tifffile

from acid.segmentation.pipeline import (
    apply_segmentation_batch,
    apply_segmentation_for_fov,
    update_metadata_with_segmentation_results,
)


class FakeModel:
    """Stand-in for Cellpose: one square object on the preprocessed grid."""

    name = "fake"
    version = "0.0"

    def __init__(self):
        self.calls = []

    def eval(self, image, **kwargs):
        self.calls.append(kwargs)
        shape = tuple(
            size
            for axis, size in enumerate(image.shape)
            if axis != kwargs["channel_axis"]
        )
        masks = np.zeros(shape, dtype=np.int32)
        masks[1:3, 1:3] = 1
        return masks, None, None


def paths_for(directory, tmp_path):
    return types.SimpleNamespace(
        corrected_fov_dir=str(directory),
        segmentation_masks_dir=str(tmp_path / "masks"),
    )


def test_apply_segmentation_for_fov_saves_full_size_mask(
    segmentation_config, write_fov, tmp_path
):
    directory, _ = write_fov()
    columns = segmentation_config.metadata.dataframe_columns
    row = pd.Series({columns.illum_correct_df_file_name_clm_name: "a.ome.tif"})

    result = apply_segmentation_for_fov(
        0, row, FakeModel(), segmentation_config, paths_for(directory, tmp_path)
    )

    assert result["success"] is True, result["error_message"]
    assert result["output_file"] == "a_segmentation.ome.tif"
    mask = tifffile.imread(tmp_path / "masks" / "a_segmentation.ome.tif")
    assert mask.shape == (16, 16)
    assert mask.dtype == np.uint16
    assert mask.max() == 1


def test_apply_segmentation_for_fov_missing_file_reports_stage(
    segmentation_config, tmp_path
):
    columns = segmentation_config.metadata.dataframe_columns
    row = pd.Series({columns.illum_correct_df_file_name_clm_name: "missing.ome.tif"})

    result = apply_segmentation_for_fov(
        0, row, FakeModel(), segmentation_config, paths_for(tmp_path, tmp_path)
    )

    assert result["success"] is False
    assert result["stage"] == "load_field_of_view"
    assert pd.isna(result[columns.metadata_df_file_name_clm_name])


def test_apply_segmentation_for_fov_model_error_reports_stage(
    segmentation_config, write_fov, tmp_path
):
    directory, _ = write_fov()
    columns = segmentation_config.metadata.dataframe_columns
    row = pd.Series({columns.illum_correct_df_file_name_clm_name: "a.ome.tif"})

    class BrokenModel(FakeModel):
        def eval(self, image, **kwargs):
            raise RuntimeError("CUDA out of memory")

    result = apply_segmentation_for_fov(
        0, row, BrokenModel(), segmentation_config, paths_for(directory, tmp_path)
    )

    assert result["stage"] == "segment_preprocessed_image"
    assert result["error_message"] == "CUDA out of memory"


def test_apply_segmentation_batch_returns_one_result_per_row(
    segmentation_config, write_fov, tmp_path
):
    directory, _ = write_fov()
    file_column = segmentation_config.metadata.dataframe_columns.illum_correct_df_file_name_clm_name
    metadata_df = pd.DataFrame(
        {file_column: ["a.ome.tif", "missing.ome.tif"]}, index=[3, 4]
    )

    results = apply_segmentation_batch(
        metadata_df, FakeModel(), segmentation_config, paths_for(directory, tmp_path)
    )

    assert [result["row_index"] for result in results] == [3, 4]
    assert [result["success"] for result in results] == [True, False]


def test_update_metadata_with_segmentation_results_fills_columns_per_row(
    segmentation_config, write_fov, tmp_path
):
    directory, _ = write_fov()
    columns = segmentation_config.metadata.dataframe_columns
    metadata_df = pd.DataFrame(
        {columns.illum_correct_df_file_name_clm_name: ["a.ome.tif", "missing.ome.tif"]}
    )
    results = apply_segmentation_batch(
        metadata_df, FakeModel(), segmentation_config, paths_for(directory, tmp_path)
    )

    updated = update_metadata_with_segmentation_results(metadata_df, results, columns)

    file_column = columns.metadata_df_file_name_clm_name
    assert updated.loc[0, file_column] == "a_segmentation.ome.tif"
    assert pd.isna(updated.loc[1, file_column])
