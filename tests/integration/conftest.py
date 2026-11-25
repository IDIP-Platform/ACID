from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acid.config import load_config
from acid.utils.save_image import tifffile_save_ometiff

# QC thresholds that flag no FOV of `write_pipeline_dataset`.
PERMISSIVE_QC = (
    "quality_control:\n  processing:\n"
    "    mean_over_std_lowpass_thresholds: [100.0, 100.0, 100.0, 100.0, 100.0]\n"
    "    mean_over_std_highpass_thresholds: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
)


@pytest.fixture
def make_pipeline_config(tmp_path):
    """Return a factory of read-only configurations rooted at ``tmp_path``.

    ``make_pipeline_config(experiment_id, extra_yaml="")`` writes
    ``tmp_path/configs/<experiment_id>.yaml`` and loads it; outputs go to
    ``tmp_path/data/processed/<experiment_id>``, raw acquisitions are read
    from ``tmp_path/raw``.
    """

    def _make(experiment_id="pipeline_test", extra_yaml=""):
        directory = tmp_path / "configs"
        directory.mkdir(exist_ok=True)
        (directory / f"{experiment_id}.yaml").write_text(
            f'experiment:\n  id: "{experiment_id}"\n'
            'shared:\n  paths:\n    raw_acquisitions_dir: "raw"\n' + extra_yaml
        )
        return load_config(
            f"configs/{experiment_id}.yaml", project_root=tmp_path, preview=False
        )

    return _make


@pytest.fixture
def pipeline_config(make_pipeline_config):
    """Return the default ``make_pipeline_config`` configuration."""
    return make_pipeline_config()


class SquareModel:
    """Stand-in for Cellpose: one 20x20 object on the preprocessed grid."""

    name = "square"
    version = "0.0"

    def eval(self, image, **kwargs):
        shape = tuple(
            size
            for axis, size in enumerate(image.shape)
            if axis != kwargs["channel_axis"]
        )
        masks = np.zeros(shape, dtype=np.int32)
        masks[10:30, 10:30] = 1
        return masks, None, None


@pytest.fixture
def square_model():
    return SquareModel()


def write_pipeline_dataset(config):
    """Write three extracted FOVs and a dataset-splitting metadata CSV.

    Rows 0 and 2 are in the training split. Returns the path of the CSV,
    which stands in for ``part_2.csv``.
    """
    paths = config.shared.paths
    directory = Path(paths.extracted_fov_dir)
    directory.mkdir(parents=True)
    names = ["p_well1_A07p2_A1", "p_well1_A07p2_A2", "p_well2_A07p2_A1"]
    rng = np.random.default_rng(0)
    for name in names:
        image = rng.integers(500, 3000, (5, 64, 64)).astype(np.uint16)
        tifffile_save_ometiff(
            directory / f"{name}.ome.tif",
            data=image,
            imagej=True,
            photometric="minisblack",
            metadata={"custom_raw_file_name": "p.nd2"},
        )
    metadata = pd.DataFrame(
        {
            "raw_file_name": ["p_well1.nd2", "p_well1.nd2", "p_well2.nd2"],
            "scene_name": ["A1", "A2", "A1"],
            "ome_tif_file_name": [f"{name}.ome.tif" for name in names],
            "experiment": ["A07.2"] * 3,
            "well": ["well1", "well1", "well2"],
            "is_train": [1, 0, 1],
        }
    )
    path = Path(paths.metadata_dir) / "input_part_2.csv"
    path.parent.mkdir(parents=True)
    metadata.to_csv(path, index=False)
    return path


@pytest.fixture
def pipeline_dataset(pipeline_config):
    """Return ``write_pipeline_dataset(pipeline_config)``."""
    return write_pipeline_dataset(pipeline_config)


@pytest.fixture
def permissive_qc():
    return PERMISSIVE_QC


@pytest.fixture
def make_pipeline_dataset():
    return write_pipeline_dataset
