from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from acid.config import load_config
from acid.utils.save_image import tifffile_save_ometiff

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def default_config():
    """Return the packaged default configuration (read-only)."""
    return load_config(project_root=ROOT, preview=False)


@pytest.fixture
def stage_config(default_config):
    """Return an editable copy of one stage section of the default configuration."""

    def _stage(name):
        return OmegaConf.create(
            OmegaConf.to_container(default_config[name], resolve=True)
        )

    return _stage


@pytest.fixture
def write_fov(tmp_path):
    """Write a small ImageJ-compatible OME-TIFF and return its directory and data.

    Pixel values start at ``offset`` because the default background-correction
    offset is 400 and the correction rejects images whose minimum is below it.
    """

    def _write(filename="a.ome.tif", shape=(5, 16, 16), offset=500):
        directory = tmp_path / "fov"
        directory.mkdir(exist_ok=True)
        image = (
            (np.arange(np.prod(shape)) % 200 + offset).astype(np.uint16).reshape(shape)
        )
        tifffile_save_ometiff(
            directory / filename,
            data=image,
            imagej=True,
            photometric="minisblack",
            metadata={"custom_raw_file_name": "a.lif"},
        )
        return directory, image

    return _write


@pytest.fixture
def correction_config(stage_config, tmp_path):
    config = stage_config("background_correction")
    config.image_saving.directory = str(tmp_path / "corrected")
    config.background_function_selection.background_function_strategy = 1
    return config


@pytest.fixture
def background_config(stage_config, tmp_path):
    config = stage_config("background_function_calculation")
    config.image_saving.directory = str(tmp_path / "backgrounds")
    return config


@pytest.fixture
def feature_config(stage_config):
    return stage_config("feature_extraction")


@pytest.fixture
def segmentation_config(stage_config, tmp_path):
    config = stage_config("object_segmentation")
    config.image_saving.directory = str(tmp_path / "masks")
    return config


@pytest.fixture
def extraction_config(stage_config, tmp_path):
    config = stage_config("field_of_view_extraction")
    config.acquisitions.directory = str(tmp_path / "raw")
    config.image_saving.directory = str(tmp_path / "extracted")
    return config


@pytest.fixture
def qc_config(stage_config):
    return stage_config("quality_control")
