from acid.utils.row_processing import make_success_result


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
