import pytest

from acid.image_processing.fov_extraction import discover_acquisitions


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
