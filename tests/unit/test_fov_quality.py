import pandas as pd
import pytest

from acid.image_quality_control.fov_quality import flag_quality, measure_quality

MEAN_COLUMNS = [f"mean_over_std-{channel}" for channel in range(5)]


@pytest.fixture
def qc_config(stage_config):
    return stage_config("quality_control")


def test_flag_quality_flags_rows_outside_thresholds(qc_config):
    metadata_df = pd.DataFrame(
        [[5, 10, 1.5, 10, 10], [10, 10, 1.5, 10, 10], [5, 10, 1.5, 10, 2]],
        columns=MEAN_COLUMNS,
    )

    flagged = flag_quality(metadata_df, MEAN_COLUMNS, qc_config)

    flag_column = qc_config.metadata.dataframe_columns.flag_column_name
    assert flagged[flag_column].tolist() == [
        qc_config.processing.ok_value,
        qc_config.processing.flag_value,
        qc_config.processing.flag_value,
    ]


def test_flag_quality_requires_one_threshold_per_channel(qc_config):
    with pytest.raises(ValueError, match="threshold counts"):
        flag_quality(pd.DataFrame({"m-0": [1.0]}), ["m-0"], qc_config)


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
