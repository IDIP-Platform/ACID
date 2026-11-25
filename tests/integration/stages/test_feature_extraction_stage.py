import types

import numpy as np
import pandas as pd
import pytest
import tifffile

from acid.feature_extraction.pipeline import (
    extract_features_batch,
    extract_features_for_fov,
    update_metadata_with_feature_results,
)


@pytest.fixture
def segmented_fov(write_fov, tmp_path):
    """A 5-channel FOV and its mask with two objects away from the edge."""
    fov_directory, _ = write_fov(filename="a_bg.ome.tif", shape=(5, 100, 100))
    mask_directory = tmp_path / "masks"
    mask_directory.mkdir()
    label_image = np.zeros((100, 100), dtype=np.uint16)
    label_image[10:40, 10:40] = 1
    label_image[50:90, 50:90] = 2
    tifffile.imwrite(mask_directory / "a_bg_segmentation.ome.tif", label_image)
    return types.SimpleNamespace(
        corrected_fov_dir=str(fov_directory),
        segmentation_masks_dir=str(mask_directory),
        feature_tables_dir=str(tmp_path / "features"),
    )


def metadata_row(config, fov="a_bg.ome.tif", mask="a_bg_segmentation.ome.tif"):
    columns = config.metadata.dataframe_columns
    return pd.Series(
        {columns.fov_column_name: fov, columns.segmentation_column_name: mask}
    )


def test_extract_features_for_fov_saves_one_row_per_object(
    feature_config, segmented_fov
):
    result = extract_features_for_fov(
        row_index=0,
        metadata_row=metadata_row(feature_config),
        config=feature_config,
        properties=["label", "area", "intensity_mean"],
        extra_properties=[],
        paths=segmented_fov,
    )

    assert result["success"] is True
    assert result["output_file"] == "a_bg.csv"
    features = pd.read_csv(f"{segmented_fov.feature_tables_dir}/a_bg.csv")
    assert features["label"].tolist() == [1, 2]
    assert {"area", "hessian_eigv_1_intensity_mean-0"} <= set(features.columns)


def test_extract_features_for_fov_raises_on_missing_mask(feature_config, segmented_fov):
    with pytest.raises(OSError, match="segmentation mask"):
        extract_features_for_fov(
            row_index=0,
            metadata_row=metadata_row(feature_config, mask="missing.ome.tif"),
            config=feature_config,
            properties=["label"],
            extra_properties=[],
            paths=segmented_fov,
        )


def test_extract_features_batch_turns_errors_into_failure_results(
    feature_config, segmented_fov
):
    columns = feature_config.metadata.dataframe_columns
    metadata_df = pd.DataFrame(
        {
            columns.fov_column_name: ["a_bg.ome.tif", "x.ome.tif"],
            columns.segmentation_column_name: [
                "a_bg_segmentation.ome.tif",
                "y.ome.tif",
            ],
        }
    )

    results = extract_features_batch(metadata_df, feature_config, segmented_fov)

    assert results[0]["success"] is True
    assert results[1]["success"] is False
    assert results[1]["stage"] == "extract_features_for_fov"
    assert results[1]["error_type"] == "OSError"
    assert results[1]["segmentation_file"] == "y.ome.tif"
    assert pd.isna(results[1][columns.metadata_df_file_name_clm_name])


def test_update_metadata_with_feature_results_fills_columns_per_row(
    feature_config, segmented_fov
):
    columns = feature_config.metadata.dataframe_columns
    metadata_df = pd.DataFrame(
        {
            columns.fov_column_name: ["a_bg.ome.tif", "x.ome.tif"],
            columns.segmentation_column_name: [
                "a_bg_segmentation.ome.tif",
                "y.ome.tif",
            ],
        }
    )
    results = extract_features_batch(metadata_df, feature_config, segmented_fov)

    updated = update_metadata_with_feature_results(metadata_df, results, columns)

    assert updated.loc[0, columns.metadata_df_file_name_clm_name] == "a_bg.csv"
    assert pd.isna(updated.loc[1, columns.metadata_df_file_name_clm_name])
