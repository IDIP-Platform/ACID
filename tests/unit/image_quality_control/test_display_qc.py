import pandas as pd
import pytest

MEAN_COLUMNS = [f"mean_over_std-{channel}" for channel in range(5)]


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
                    columns.experiment_column_name: [
                        "A07.2",
                        "A07.2",
                        "A07.3",
                        "A07.3",
                    ],
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
        assert any(
            name.endswith(f"_proj_{word}{graphs.graph_suffix}") for name in saved
        )
