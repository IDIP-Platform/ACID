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


def test_plot_quality_saves_four_graphs(qc_config, tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from acid.image_quality_control.display_qc import plot_quality

    columns = qc_config.metadata.dataframe_columns
    qc_config.graphs_saving.graph_saving_directory = str(tmp_path / "graphs")
    skew_columns = [f"normalized_intensity_skewness-{channel}" for channel in range(5)]
    values = [[float(row + channel) for channel in range(5)] for row in range(4)]
    metadata_df = pd.concat(
        [
            pd.DataFrame(values, columns=MEAN_COLUMNS),
            pd.DataFrame(values, columns=skew_columns),
            pd.DataFrame(
                {
                    columns.experiment_column_name: ["A07.2", "A07.2", "A07.3", "A07.3"],
                    columns.well_column_name: ["well1", "well2", "well1", "well2"],
                }
            ),
        ],
        axis=1,
    )

    plot_quality(metadata_df, MEAN_COLUMNS, skew_columns, 5, qc_config, "proj")
    plt.close("all")

    saved = sorted(path.name for path in (tmp_path / "graphs").iterdir())
    graphs = qc_config.graphs_saving
    assert len(saved) == 4
    for word in (
        graphs.mean_over_std_per_channel_savingword,
        graphs.skewness_per_channel_savingword,
        graphs.mean_over_std_per_ch_exp_savingword,
        graphs.mean_over_std_per_ch_well_savingword,
    ):
        assert any(name.endswith(f"_proj_{word}{graphs.graph_suffix}") for name in saved)
