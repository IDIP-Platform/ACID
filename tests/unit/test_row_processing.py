import numpy as np

from acid.utils.row_processing import make_failure_result, make_success_result

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
