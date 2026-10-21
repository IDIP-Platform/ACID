import numpy as np
import pandas as pd
import pytest
import tifffile

from acid.image_processing import fov_extraction
from acid.image_processing.fov_extraction import (
    discover_acquisitions,
    extract_acquisition,
    extract_fields_of_view,
)


@pytest.fixture
def extraction_config(stage_config, tmp_path):
    config = stage_config("field_of_view_extraction")
    config.acquisitions.directory = str(tmp_path / "raw")
    config.image_saving.directory = str(tmp_path / "extracted")
    return config


def test_discover_acquisitions_finds_matching_files_in_experiment_dirs(
    extraction_config, tmp_path
):
    raw = tmp_path / "raw"
    (raw / "experiment_A07.2").mkdir(parents=True)
    (raw / "experiment_A07.2" / "plate_well2.nd2").write_text("")
    (raw / "experiment_A07.2" / "plate_well1.nd2").write_text("")
    (raw / "experiment_A07.2" / "notes.txt").write_text("")
    (raw / "20251127_plate_layout.csv").write_text("")

    result = discover_acquisitions(extraction_config)

    assert result == [
        raw / "experiment_A07.2" / "plate_well1.nd2",
        raw / "experiment_A07.2" / "plate_well2.nd2",
    ]


def test_discover_acquisitions_without_matches_raises(extraction_config, tmp_path):
    (tmp_path / "raw" / "experiment_A07.2").mkdir(parents=True)

    with pytest.raises(ValueError, match="No matching acquisition files"):
        discover_acquisitions(extraction_config)


class FakeBioImage:
    """Stand-in for bioio.BioImage of an ND2 file with two scenes."""

    scenes = ("A1", "A2")

    def __init__(self):
        self.current_scene = None

    def set_scene(self, scene):
        self.current_scene = scene

    @property
    def data(self):
        value = self.scenes.index(self.current_scene) + 1
        return np.full((1, 5, 1, 8, 8), value, dtype=np.uint16)


@pytest.fixture
def fake_bioio(monkeypatch):
    """Replace the ND2 reader and the bioio-specific metadata helpers.

    No ND2 writer exists, so the reader is faked; file naming, name parsing
    and OME-TIFF/XML writing run for real.
    """
    scene_metadata_calls = []

    def fake_extract_bioio_scene_metadata(**kwargs):
        scene_metadata_calls.append(kwargs)
        metadata = {
            "ome_tif_file_name": kwargs["ome_tif_file_name"],
            "scene": kwargs["scene_name"],
            "well": kwargs["well"],
        }
        return pd.Series(metadata), metadata

    monkeypatch.setattr(
        fov_extraction,
        "bioio_open_image",
        lambda path, return_metadata: (FakeBioImage(), "ome-metadata"),
    )
    monkeypatch.setattr(fov_extraction, "to_xml", lambda metadata: "<OME/>")
    monkeypatch.setattr(
        fov_extraction, "extract_bioio_scene_metadata", fake_extract_bioio_scene_metadata
    )
    return scene_metadata_calls


def test_extract_acquisition_saves_one_ome_tiff_per_scene(
    extraction_config, tmp_path, fake_bioio
):
    (tmp_path / "extracted").mkdir()
    acquisition = (
        tmp_path / "raw" / "experiment_A07.4" / "H7_DENV2_MOI1_40h_fixed_stained_well6.nd2"
    )

    scenes = extract_acquisition(acquisition, extraction_config)

    stem = "H7_DENV2_MOI1_40h_fixed_stained_well6"
    names = [f"{stem}_A07p4_A1.ome.tif", f"{stem}_A07p4_A2.ome.tif"]
    assert [scene["ome_tif_file_name"] for scene in scenes] == names
    assert tifffile.imread(tmp_path / "extracted" / names[1]).shape == (5, 8, 8)
    assert tifffile.imread(tmp_path / "extracted" / names[1])[0, 0, 0] == 2
    assert "<OME/>" in (tmp_path / "extracted" / f"{stem}_str.xml").read_text()
    assert fake_bioio[0]["experiment"] == "A07.4"
    assert fake_bioio[0]["raw_file_name"] == f"{stem}.nd2"


def test_extract_fields_of_view_builds_one_row_per_scene(
    extraction_config, tmp_path, fake_bioio
):
    (tmp_path / "extracted").mkdir()
    experiment = tmp_path / "raw" / "experiment_A07.2"
    acquisitions = [experiment / "H7_DENV2_MOI1_30h_fixed_stained_well1.nd2",
                    experiment / "H7_DENV2_MOI1_30h_fixed_stained_well2.nd2"]

    metadata_df = extract_fields_of_view(acquisitions, extraction_config)

    assert metadata_df.shape[0] == 4
    assert metadata_df["scene"].tolist() == ["A1", "A2", "A1", "A2"]


def test_extract_fields_of_view_requires_five_channels(extraction_config):
    extraction_config.image_metadata.channels = ["a", "b", "c", "d"]

    with pytest.raises(ValueError, match="five configured channels"):
        extract_fields_of_view([], extraction_config)


def test_extract_fields_of_view_without_scenes_raises(extraction_config):
    with pytest.raises(ValueError, match="No fields of view"):
        extract_fields_of_view([], extraction_config)
