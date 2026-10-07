from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from acid.utils.metadata.rows import get_required_filename
from acid.utils.metadata.saving import build_metadata_dataframe_filename


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


def test_get_required_filename_strips_whitespace():
    row = pd.Series({"file": "  a.ome.tif "})

    assert get_required_filename(row, "file") == "a.ome.tif"


@pytest.mark.parametrize("value", [np.nan, None, "", "   "])
def test_get_required_filename_rejects_empty_cells(value):
    row = pd.Series({"file": value})

    with pytest.raises(ValueError, match="Missing filename in column 'file'"):
        get_required_filename(row, "file")


def test_get_required_filename_rejects_missing_column():
    with pytest.raises(ValueError, match="'other'"):
        get_required_filename(pd.Series({"file": "a"}), "other")


def test_build_metadata_dataframe_filename_uses_timestamp_and_suffix(tmp_path):
    config = metadata_config(tmp_path)

    result = build_metadata_dataframe_filename(
        config.saving, "proj", timestamp=datetime(2026, 10, 6)
    )

    assert result == "20261006_proj_metadata_part_4.csv"
