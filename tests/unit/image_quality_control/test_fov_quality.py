import pandas as pd
import pytest

from acid.image_quality_control.fov_quality import (
    add_quality_columns,
    flag_quality,
    get_quality_column_names,
)

MEAN_COLUMNS = [f"mean_over_std-{channel}" for channel in range(5)]


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


def test_get_quality_column_names_uses_separator_and_channel(qc_config):
    names = get_quality_column_names(2, qc_config)

    assert names == {
        "plls": ["plls-0", "plls-1"],
        "skewness": [
            "normalized_intensity_skewness-0",
            "normalized_intensity_skewness-1",
        ],
        "mean_over_std": ["mean_over_std-0", "mean_over_std-1"],
    }


def test_add_quality_columns_writes_one_column_per_statistic_and_channel(qc_config):
    metadata_df = pd.DataFrame({"x": [1, 2]}, index=[5, 9])
    measurements = [
        {"plls": [1.0, 2.0], "skewness": [3.0, 4.0], "mean_over_std": [5.0, 6.0]},
        {"plls": [7.0, 8.0], "skewness": [9.0, 10.0], "mean_over_std": [11.0, 12.0]},
    ]

    names = add_quality_columns(metadata_df, measurements, 2, qc_config)

    assert names == get_quality_column_names(2, qc_config)
    assert metadata_df.loc[9, names["mean_over_std"]].tolist() == [11.0, 12.0]
    assert metadata_df.loc[5, names["plls"]].tolist() == [1.0, 2.0]
    assert (metadata_df[names["skewness"]].dtypes == "float64").all()
