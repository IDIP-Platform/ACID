import numpy as np
import pandas as pd
import pytest
import tifffile

from acid.image_processing.background.compute_background_function import (
    calculate_backgrounds,
    fit_background,
    save_backgrounds,
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


def average_background():
    return np.random.default_rng(0).random((5, 32, 32)) * 100 + 50


@pytest.mark.parametrize("method", ["simple", "polyfit"])
def test_fit_background_keeps_shape(background_config, method):
    background_config.processing.background_fit_method = method

    fitted = fit_background(average_background(), background_config)

    assert fitted.shape == (5, 32, 32)


def test_fit_background_rejects_unknown_method(background_config):
    background_config.processing.background_fit_method = "bogus"

    with pytest.raises(ValueError, match="polyfit or simple"):
        fit_background(average_background(), background_config)


@pytest.fixture
def fov_metadata(background_config, tmp_path):
    """Three small fields of view, two in well1 and one in well2."""
    directory = tmp_path / "fov"
    directory.mkdir()
    rng = np.random.default_rng(0)
    for index in range(3):
        image = (rng.random((5, 32, 32)) * 100 + 50).astype(np.uint16)
        tifffile.imwrite(directory / f"f{index}.tif", image)
    background_config.processing.fov_directory = str(directory)
    columns = background_config.metadata.dataframe_columns
    return pd.DataFrame(
        {
            columns.fov_column_name: ["f0.tif", "f1.tif", "f2.tif"],
            columns.well_column_name: ["well1", "well1", "well2"],
        }
    )


def test_calculate_backgrounds_dataset_strategy_gives_one_background(
    background_config, fov_metadata
):
    background_config.processing.background_function_strategy = 1

    backgrounds = calculate_backgrounds(fov_metadata, (5, 32, 32), background_config)

    assert list(backgrounds) == [None]
    assert backgrounds[None].shape == (5, 32, 32)


def test_calculate_backgrounds_well_strategy_gives_one_background_per_well(
    background_config, fov_metadata
):
    background_config.processing.background_function_strategy = 2

    backgrounds = calculate_backgrounds(fov_metadata, (5, 32, 32), background_config)

    assert list(backgrounds) == ["well1", "well2"]


def test_calculate_backgrounds_rejects_unknown_strategy(background_config):
    background_config.processing.background_function_strategy = 4

    with pytest.raises(ValueError, match="must be 1, 2 or 3"):
        calculate_backgrounds(pd.DataFrame(), (5, 32, 32), background_config)


def test_calculate_backgrounds_rejects_missing_well_for_well_strategy(
    background_config,
):
    background_config.processing.background_function_strategy = 2
    well = background_config.metadata.dataframe_columns.well_column_name

    with pytest.raises(ValueError, match="Missing background grouping values"):
        calculate_backgrounds(
            pd.DataFrame({well: ["A1", None]}), (5, 32, 32), background_config
        )


def test_save_backgrounds_writes_fitted_background_per_group(background_config, tmp_path):
    background_config.processing.background_function_strategy = 2
    backgrounds = {"well1": average_background(), "well2": average_background()}

    saved = save_backgrounds(backgrounds, background_config, "proj")

    names = sorted(path.name for path in (tmp_path / "backgrounds").iterdir())
    assert [entry["group"] for entry in saved] == ["well1", "well2"]
    assert len(names) == 2
    assert names[0].endswith("_proj_background_well1.ome.tif")
    image = tifffile.imread(saved[0]["output_file"])
    assert image.shape == (5, 32, 32)
    assert image.dtype == np.dtype(background_config.image_saving.background_img_dtype)
