import os

import numpy as np
import pandas as pd
import pytest
import tifffile
from omegaconf import OmegaConf

from acid.image_processing.background.load_background_function import (
    load_background_function,
)
from acid.utils.fov_axis_utils import get_fov_ch_shape
from acid.utils.metadata.loading import load_metadata


def test_metadata_selection_uses_stage_filter(tmp_path):
    upstream = tmp_path / "20260901_ACID_metadata_part_5.csv"
    later = tmp_path / "20260902_ACID_metadata_part_6.csv"
    pd.DataFrame({"stage": [5]}).to_csv(upstream, index=False)
    pd.DataFrame({"stage": [6]}).to_csv(later, index=False)
    os.utime(upstream, (100, 100))
    os.utime(later, (200, 200))

    config = OmegaConf.create(
        {
            "directory": str(tmp_path),
            "file_selection": {
                "filename": "default",
                "include": "part_5.csv",
                "exclude": None,
                "date_source": "modified_time",
                "select": "newest",
            },
        }
    )

    metadata, filename = load_metadata(config)

    assert filename == upstream.name
    assert metadata["stage"].tolist() == [5]


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


def test_fov_shape_returns_null_values_when_no_image_can_be_opened(tmp_path):
    metadata = pd.DataFrame({"ome_tif_file_name": ["missing.ome.tif"]})

    result = get_fov_ch_shape(
        metadata,
        tmp_path,
        fov_clm="ome_tif_file_name",
        null_value=None,
    )

    assert result == (None, None, None)
