import numpy as np
import pandas as pd
import pytest

from acid.utils.metadata.rows import get_required_filename


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
