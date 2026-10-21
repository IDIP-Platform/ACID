import math

import numpy as np
import pandas as pd
import pytest

from acid.pipeline.records import (
    fov_name,
    list_fov_names,
    make_batches,
    read_records,
    select_fov_rows,
    select_stage_rows,
    write_records,
)


@pytest.fixture
def metadata_df():
    return pd.DataFrame(
        {
            "ome_tif_file_name": ["a_A1.ome.tif", "a_A2.ome.tif", "b_A1.ome.tif"],
            "is_train": [1, 0, 1],
        }
    )


def test_fov_name_strips_ome_suffix(default_config):
    assert fov_name("a_A1.ome.tif", default_config) == "a_A1"


def test_fov_name_rejects_other_suffix(default_config):
    with pytest.raises(ValueError, match="does not end with"):
        fov_name("a_A1.tif", default_config)


def test_select_stage_rows_keeps_training_rows_and_index(default_config, metadata_df):
    selected = select_stage_rows(metadata_df, "background_correction", default_config)

    assert selected.index.tolist() == [0, 2]


def test_select_stage_rows_keeps_all_rows_for_feature_extraction(
    default_config, metadata_df
):
    selected = select_stage_rows(metadata_df, "feature_extraction", default_config)

    pd.testing.assert_frame_equal(selected, metadata_df)


def test_list_fov_names_rejects_duplicates(default_config, metadata_df):
    metadata_df.loc[1, "ome_tif_file_name"] = "a_A1.ome.tif"

    with pytest.raises(ValueError, match="not unique"):
        list_fov_names(metadata_df, default_config)


def test_select_fov_rows_keeps_requested_order_and_original_index(
    default_config, metadata_df
):
    selected = select_stage_rows(metadata_df, "object_segmentation", default_config)

    rows = select_fov_rows(selected, ["b_A1", "a_A1"], default_config)

    assert rows.index.tolist() == [2, 0]


def test_select_fov_rows_rejects_unknown_fov(default_config, metadata_df):
    with pytest.raises(KeyError, match="a_A9"):
        select_fov_rows(metadata_df, ["a_A1", "a_A9"], default_config)


def test_make_batches_splits_in_order_with_short_last_batch():
    assert make_batches(["a", "b", "c", "d", "e"], 2) == [["a", "b"], ["c", "d"], ["e"]]
    assert make_batches(["a"], 50) == [["a"]]
    assert make_batches([], 3) == []


def test_make_batches_rejects_size_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        make_batches(["a"], 0)


def test_write_records_round_trips_numpy_and_nan(tmp_path):
    record = {
        "row_index": np.int64(3),
        "value": np.float32(0.5),
        "values": np.array([1.25, np.nan]),
        "missing": np.nan,
        "na": pd.NA,
        "text": "x",
    }

    (result,) = read_records([write_records([record], tmp_path / "nested" / "r.json")])

    assert result["row_index"] == 3
    assert result["value"] == 0.5
    assert result["values"][0] == 1.25 and math.isnan(result["values"][1])
    assert math.isnan(result["missing"])
    assert result["na"] is None
    assert result["text"] == "x"


def test_read_records_concatenates_files_in_order(tmp_path):
    first = write_records([{"row_index": 0}, {"row_index": 1}], tmp_path / "a.json")
    second = write_records([{"row_index": 2}], tmp_path / "b.json")

    records = read_records([first, second])

    assert [record["row_index"] for record in records] == [0, 1, 2]
