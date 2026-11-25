import numpy as np
import pandas as pd
import pytest
import tifffile

from acid.image_processing.background.compute_background_function import (
    calculate_backgrounds,
    save_backgrounds,
)


def average_background():
    return np.random.default_rng(0).random((5, 32, 32)) * 100 + 50


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


@pytest.mark.parametrize("n_threads, expected", [(3, 3), (None, "available")])
def test_calculate_backgrounds_reads_and_averages_with_given_threads(
    background_config, fov_metadata, monkeypatch, n_threads, expected
):
    from acid.image_processing.background import compute_background_function

    calls = []
    for name in ("import_fov", "calculate_background_function"):
        original = getattr(compute_background_function, name)

        def spy(*args, _original=original, **kwargs):
            calls.append(kwargs["n_workers"])
            return _original(*args, **kwargs)

        monkeypatch.setattr(compute_background_function, name, spy)
    monkeypatch.setattr(compute_background_function, "_available_cpus", lambda: 7)
    background_config.processing.background_function_strategy = 1

    calculate_backgrounds(
        fov_metadata, (5, 32, 32), background_config, n_threads=n_threads
    )

    assert calls == [7 if expected == "available" else expected] * 2


def test_save_backgrounds_writes_fitted_background_per_group(
    background_config, tmp_path
):
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
