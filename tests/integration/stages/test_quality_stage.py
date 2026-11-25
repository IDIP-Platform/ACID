import pandas as pd

from acid.image_quality_control.fov_quality import (
    get_quality_column_names,
    measure_quality,
    measure_quality_for_fov,
)

MEAN_COLUMNS = [f"mean_over_std-{channel}" for channel in range(5)]


def test_measure_quality_adds_per_channel_columns_and_nan_for_unreadable(
    qc_config, write_fov
):
    directory, _ = write_fov()
    qc_config.processing.fov_directory = str(directory)
    fov_column = qc_config.metadata.dataframe_columns.fov_column_name
    metadata_df = pd.DataFrame({fov_column: ["a.ome.tif", "missing.ome.tif"]})

    measured, num_channels, mean_columns, skew_columns = measure_quality(
        metadata_df, qc_config
    )

    assert num_channels == 5
    assert mean_columns == MEAN_COLUMNS
    assert skew_columns[0] == "normalized_intensity_skewness-0"
    assert measured.loc[0, ["plls-0", "mean_over_std-0", skew_columns[0]]].notna().all()
    assert measured.loc[1, ["plls-0", "mean_over_std-0", skew_columns[0]]].isna().all()
    assert "plls-0" not in metadata_df.columns


def test_measure_quality_for_fov_matches_measure_quality(qc_config, write_fov):
    directory, _ = write_fov()
    qc_config.processing.fov_directory = str(directory)
    fov_column = qc_config.metadata.dataframe_columns.fov_column_name
    measured, _, _, _ = measure_quality(
        pd.DataFrame({fov_column: ["a.ome.tif"]}), qc_config
    )

    single = measure_quality_for_fov("a.ome.tif", 5, qc_config)

    for statistic, names in get_quality_column_names(5, qc_config).items():
        assert list(single[statistic]) == measured.loc[0, names].tolist()


def test_measure_quality_for_fov_returns_null_values_for_unreadable_file(
    qc_config, tmp_path
):
    qc_config.processing.fov_directory = str(tmp_path)

    single = measure_quality_for_fov("missing.ome.tif", 3, qc_config)

    assert set(single) == {"plls", "skewness", "mean_over_std"}
    assert all(pd.isna(value) for values in single.values() for value in values)
    assert all(len(values) == 3 for values in single.values())
