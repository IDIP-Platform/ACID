import numpy as np
import pandas as pd
import pytest
import tifffile

from acid.image_processing import fov_extraction
from acid.image_processing.fov_extraction import (
    extract_acquisition,
    extract_fields_of_view,
)


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
        fov_extraction,
        "extract_bioio_scene_metadata",
        fake_extract_bioio_scene_metadata,
    )
    return scene_metadata_calls


def test_extract_acquisition_saves_one_ome_tiff_per_scene(
    extraction_config, tmp_path, fake_bioio
):
    (tmp_path / "extracted").mkdir()
    acquisition = (
        tmp_path
        / "raw"
        / "experiment_A07.4"
        / "H7_DENV2_MOI1_40h_fixed_stained_well6.nd2"
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
    acquisitions = [
        experiment / "H7_DENV2_MOI1_30h_fixed_stained_well1.nd2",
        experiment / "H7_DENV2_MOI1_30h_fixed_stained_well2.nd2",
    ]

    metadata_df = extract_fields_of_view(acquisitions, extraction_config)

    assert metadata_df.shape[0] == 4
    assert metadata_df["scene"].tolist() == ["A1", "A2", "A1", "A2"]
