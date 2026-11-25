import os

import pandas as pd
import pytest
from omegaconf import OmegaConf

from acid.utils.metadata.loading import load_metadata, load_metadata_file


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


def test_load_metadata_file_reads_explicit_path(tmp_path):
    path = tmp_path / "ACID_metadata_part_5.csv"
    pd.DataFrame({"stage": [5]}).to_csv(path, index=False)

    assert load_metadata_file(path)["stage"].tolist() == [5]


def test_load_metadata_file_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        load_metadata_file(tmp_path / "missing.csv")


def filename_date_config(directory):
    return OmegaConf.create(
        {
            "directory": str(directory),
            "file_selection": {
                "filename": "default",
                "include": "part_3.csv",
                "exclude": None,
                "date_source": "filename",
                "select": "newest",
            },
        }
    )


def test_metadata_selection_by_filename_date_skips_undated_files(tmp_path):
    pd.DataFrame({"stage": [1]}).to_csv(
        tmp_path / "20261007_ACID_metadata_part_3.csv", index=False
    )
    pd.DataFrame({"stage": [2]}).to_csv(
        tmp_path / "ACID_metadata_part_3.csv", index=False
    )

    metadata, filename = load_metadata(filename_date_config(tmp_path))

    assert filename == "20261007_ACID_metadata_part_3.csv"
    assert metadata["stage"].tolist() == [1]


def test_metadata_selection_by_filename_date_without_dated_files_raises(tmp_path):
    pd.DataFrame({"stage": [2]}).to_csv(
        tmp_path / "ACID_metadata_part_3.csv", index=False
    )

    with pytest.raises(ValueError, match="with a date in their name"):
        load_metadata(filename_date_config(tmp_path))
