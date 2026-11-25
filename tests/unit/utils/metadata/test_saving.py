from datetime import UTC, datetime

import pandas as pd
from omegaconf import OmegaConf

from acid.utils.metadata.saving import (
    build_metadata_dataframe_filename,
    build_metadata_dataframe_path,
    build_undated_metadata_dataframe_filename,
    save_metadata_dataframe,
)


def metadata_config(directory):
    return OmegaConf.create(
        {
            "directory": str(directory),
            "saving": {
                "save_file_name_separator": "_",
                "metadata_file_suffix": "part{save_file_name_separator}4.csv",
                "metadata_date_format": "%Y%m%d",
                "metadata_savingword": "metadata",
                "save_csv_index": False,
            },
        }
    )


def test_build_metadata_dataframe_filename_uses_timestamp_and_suffix(tmp_path):
    config = metadata_config(tmp_path)

    result = build_metadata_dataframe_filename(
        config.saving, "proj", timestamp=datetime(2026, 10, 6, tzinfo=UTC)
    )

    assert result == "20261006_proj_metadata_part_4.csv"


def test_build_metadata_dataframe_path_joins_directory(tmp_path):
    result = build_metadata_dataframe_path(
        metadata_config(tmp_path), "proj", timestamp=datetime(2026, 10, 6, tzinfo=UTC)
    )

    assert result == tmp_path / "20261006_proj_metadata_part_4.csv"


def test_save_metadata_dataframe_creates_directory_and_writes_csv(tmp_path):
    config = metadata_config(tmp_path / "nested")
    frame = pd.DataFrame({"a": [1, 2]})

    path = save_metadata_dataframe(
        frame, config, "proj", timestamp=datetime(2026, 10, 6, tzinfo=UTC)
    )

    assert path.is_file()
    pd.testing.assert_frame_equal(pd.read_csv(path), frame)


def test_save_metadata_dataframe_works_with_every_stage_config(
    default_config, tmp_path
):
    for stage in (
        "field_of_view_extraction",
        "dataset_splitting",
        "quality_control",
        "background_correction",
        "object_segmentation",
        "feature_extraction",
    ):
        stage_metadata = OmegaConf.to_container(
            default_config[stage].metadata, resolve=True
        )
        stage_metadata["directory"] = str(tmp_path / stage)
        path = save_metadata_dataframe(
            pd.DataFrame({"a": [1]}), OmegaConf.create(stage_metadata), "proj"
        )
        assert path.is_file(), stage


def test_build_undated_metadata_dataframe_filename_has_no_date(tmp_path):
    result = build_undated_metadata_dataframe_filename(
        metadata_config(tmp_path).saving, "proj"
    )

    assert result == "proj_metadata_part_4.csv"


def test_save_metadata_dataframe_writes_to_explicit_path(tmp_path):
    config = metadata_config(tmp_path / "unused")
    target = tmp_path / "out" / "x.csv"
    frame = pd.DataFrame({"a": [1, 2]})

    path = save_metadata_dataframe(frame, config, "proj", path=target)

    assert path == target
    assert not (tmp_path / "unused").exists()
    pd.testing.assert_frame_equal(pd.read_csv(target), frame)
