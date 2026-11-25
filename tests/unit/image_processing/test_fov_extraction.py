import pytest

from acid.image_processing.fov_extraction import (
    discover_acquisitions,
    extract_fields_of_view,
)


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


def test_extract_fields_of_view_requires_five_channels(extraction_config):
    extraction_config.image_metadata.channels = ["a", "b", "c", "d"]

    with pytest.raises(ValueError, match="five configured channels"):
        extract_fields_of_view([], extraction_config)


def test_extract_fields_of_view_without_scenes_raises(extraction_config):
    with pytest.raises(ValueError, match="No fields of view"):
        extract_fields_of_view([], extraction_config)
