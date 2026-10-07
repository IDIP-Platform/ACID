import pandas as pd
import pytest

from acid.image_processing.background.compute_background_function import (
    update_background_metadata,
)


@pytest.fixture
def background_config(stage_config, tmp_path):
    config = stage_config("background_function_calculation")
    config.image_saving.directory = str(tmp_path / "backgrounds")
    return config


def test_update_background_metadata_simple_fit_writes_per_channel_columns(
    background_config,
):
    background_config.processing.background_fit_method = "simple"

    updated = update_background_metadata(pd.DataFrame({"a": [1, 2]}), background_config)

    assert updated.columns.tolist() == [
        "a",
        "background_funct_date",
        "background_funct_avg_method",
        "background_funct_ball_radius",
        *[f"background_funct_white_bg-{channel}" for channel in range(5)],
        "background_funct_gau_smooth",
    ]
    assert updated["background_funct_white_bg-4"].tolist() == [True, True]


def test_update_background_metadata_polyfit_records_polynomial_orders(
    background_config,
):
    background_config.processing.background_fit_method = "polyfit"

    updated = update_background_metadata(pd.DataFrame({"a": [1]}), background_config)

    columns = background_config.metadata.dataframe_columns
    separator = columns.channel_name_separator
    assert f"{columns.background_df_poly_order_x_clm_name}{separator}0" in updated
    assert "background_funct_ball_radius" not in updated
