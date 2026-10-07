import pandas as pd
import pytest

from acid.image_quality_control.fov_quality import flag_quality

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
