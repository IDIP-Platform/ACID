import numpy as np
import pandas as pd
import pytest

from acid.utils.row_processing import (
    make_failure_result,
    make_success_result,
    process_rows,
    update_metadata_with_results,
)

COLUMNS = ["out_date", "out_file"]


def test_make_success_result_has_common_and_metadata_keys():
    result = make_success_result(
        row_index=3,
        input_file="a.tif",
        output_file="b.tif",
        metadata_values={"out_date": "261006", "out_file": "b.tif"},
        segmentation_file="m.tif",
    )

    assert list(result) == [
        "row_index",
        "input_file",
        "segmentation_file",
        "output_file",
        "success",
        "stage",
        "error_type",
        "error_message",
        "out_date",
        "out_file",
    ]
    assert result["success"] is True
    assert result["stage"] is None


def test_make_failure_result_fills_metadata_with_null_value():
    result = make_failure_result(
        row_index=1,
        input_file="a.tif",
        error=OSError("boom"),
        metadata_columns=COLUMNS,
        null_value=np.nan,
        stage="load_field_of_view",
    )

    assert result["success"] is False
    assert result["output_file"] is None
    assert result["stage"] == "load_field_of_view"
    assert result["error_type"] == "OSError"
    assert result["error_message"] == "boom"
    assert all(np.isnan(result[column]) for column in COLUMNS)


def test_update_metadata_with_results_adds_missing_columns_and_values():
    metadata_df = pd.DataFrame({"fov": ["a", "b"]}, index=[10, 11])
    results = [
        make_success_result(10, "a", "a_out", {"out_date": "d", "out_file": "a_out"}),
        make_failure_result(11, "b", OSError("x"), COLUMNS, np.nan),
    ]

    updated = update_metadata_with_results(metadata_df, results, COLUMNS)

    assert updated.loc[10, "out_file"] == "a_out"
    assert pd.isna(updated.loc[11, "out_file"])
    assert "out_file" not in metadata_df.columns


def test_update_metadata_with_results_empty_results_returns_copy():
    metadata_df = pd.DataFrame({"fov": ["a"]})

    updated = update_metadata_with_results(metadata_df, [], COLUMNS)

    pd.testing.assert_frame_equal(updated, metadata_df)
    assert updated is not metadata_df


def test_update_metadata_with_results_rejects_results_without_columns():
    metadata_df = pd.DataFrame({"fov": ["a"]})
    results = [{"row_index": 0, "out_date": "d"}]

    with pytest.raises(KeyError, match="out_file"):
        update_metadata_with_results(metadata_df, results, COLUMNS)


def test_process_rows_passes_index_and_row_and_respects_max_rows():
    metadata_df = pd.DataFrame({"fov": ["a", "b", "c"]}, index=[5, 6, 7])

    results = process_rows(
        metadata_df,
        lambda row_index, row: {"row_index": row_index, "fov": row["fov"]},
        description="test",
        max_rows=2,
    )

    assert results == [{"row_index": 5, "fov": "a"}, {"row_index": 6, "fov": "b"}]
