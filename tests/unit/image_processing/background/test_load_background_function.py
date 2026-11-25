import os

import numpy as np
import pytest
import tifffile
from omegaconf import OmegaConf

from acid.image_processing.background.load_background_function import (
    load_background_function,
)


def test_background_selection_uses_nested_file_selection(tmp_path):
    older = tmp_path / "20260901_ACID_background.ome.tif"
    newer = tmp_path / "20260902_ACID_background.ome.tif"
    tifffile.imwrite(older, np.full((2, 2), 1, dtype=np.uint16))
    tifffile.imwrite(newer, np.full((2, 2), 2, dtype=np.uint16))
    (tmp_path / "unrelated.csv").write_text("not an image")
    os.utime(older, (100, 100))
    os.utime(newer, (200, 200))

    config = OmegaConf.create(
        {
            "directory": str(tmp_path),
            "background_function_strategy": 1,
            "file_selection": {
                "filename": "default",
                "include": ".ome.tif",
                "exclude": None,
                "date_source": "modified_time",
                "select": "newest",
            },
        }
    )

    background, filename = load_background_function(config)

    assert filename == newer.name
    assert np.all(background == 2)


def test_background_selection_reports_no_matching_files(tmp_path):
    config = OmegaConf.create(
        {
            "directory": str(tmp_path),
            "background_function_strategy": 1,
            "file_selection": {"include": ".ome.tif"},
        }
    )

    with pytest.raises(ValueError, match="No background function files match"):
        load_background_function(config)


def test_background_selection_keeps_legacy_flat_settings(tmp_path):
    older = tmp_path / "20260901_ACID_background.ome.tif"
    newer = tmp_path / "20260902_ACID_background.ome.tif"
    tifffile.imwrite(older, np.full((2, 2), 1, dtype=np.uint16))
    tifffile.imwrite(newer, np.full((2, 2), 2, dtype=np.uint16))

    config = {
        "directory": str(tmp_path),
        "background_function_strategy": 1,
        "default_bg_funct_file_target": ".ome.tif",
        "background_from_file_name": True,
        "background_default_date_format": "%Y%m%d",
    }

    background, filename = load_background_function(config)

    assert filename == newer.name
    assert np.all(background == 2)
