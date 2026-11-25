from pathlib import Path

import pandas as pd
import pytest

from acid.pipeline.merge_jobs import (
    merge_fov_extraction,
    merge_results,
    split_dataset_job,
)
from acid.pipeline.records import write_records


def test_merge_fov_extraction_keeps_text_and_skips_empty_files(
    pipeline_config, tmp_path
):
    first, empty, second = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    first.write_text("ome_tif_file_name,date\na_A1.ome.tif,010203\n")
    empty.write_text("")
    second.write_text("ome_tif_file_name,date\nc_A1.ome.tif,\n")

    output = merge_fov_extraction(
        pipeline_config, [first, empty, second], tmp_path / "part_1.csv"
    )

    assert output.read_text() == (
        "ome_tif_file_name,date\na_A1.ome.tif,010203\nc_A1.ome.tif,\n"
    )


def test_merge_fov_extraction_without_scenes_raises(pipeline_config, tmp_path):
    (tmp_path / "a.csv").write_text("")

    with pytest.raises(ValueError, match="No fields of view"):
        merge_fov_extraction(pipeline_config, [tmp_path / "a.csv"], tmp_path / "o.csv")


def test_split_dataset_job_adds_treatment_and_split(pipeline_config, tmp_path):
    (tmp_path / "raw").mkdir()
    pd.DataFrame(
        {"experiment": ["A07.2", "A07.2"], "well": [1, 2], "treatment": ["a", "b"]}
    ).to_csv(tmp_path / "raw" / "plate_layout.csv", index=False)
    part_1 = tmp_path / "part_1.csv"
    pd.DataFrame({"experiment": ["A07.2"] * 10, "well": ["well1", "well2"] * 5}).to_csv(
        part_1, index=False
    )

    output = split_dataset_job(pipeline_config, part_1, tmp_path / "part_2.csv")

    result = pd.read_csv(output)
    assert result["treatment"].tolist()[:2] == ["a", "b"]
    assert result["is_train"].value_counts().to_dict() == {1: 7, 0: 3}


def test_merge_results_rejects_missing_records(
    pipeline_config, pipeline_dataset, tmp_path
):
    record = write_records([{"row_index": 0, "success": True}], tmp_path / "0.json")

    with pytest.raises(ValueError, match="1 result records do not match the 2"):
        merge_results(
            pipeline_config,
            "background_correction",
            pipeline_dataset,
            [record],
            tmp_path / "out.csv",
        )


def failure_record(row_index, directory):
    return write_records(
        [
            {
                "row_index": row_index,
                "input_file": f"{row_index}.ome.tif",
                "output_file": None,
                "success": False,
                "stage": "load_field_of_view",
                "error_type": "OSError",
                "error_message": "unreadable",
            }
        ],
        directory / f"{row_index}.json",
    )


def test_merge_results_raises_when_every_fov_failed_and_reports_them(
    pipeline_config, pipeline_dataset, tmp_path
):
    records = [failure_record(0, tmp_path), failure_record(2, tmp_path)]

    with pytest.raises(ValueError, match="every field of view failed"):
        merge_results(
            pipeline_config,
            "background_correction",
            pipeline_dataset,
            records,
            tmp_path / "out.csv",
        )

    report = pd.read_csv(
        Path(pipeline_config.shared.output_root)
        / "reports"
        / "background_correction_failures.csv"
    )
    assert report["row_index"].tolist() == [0, 2]
    assert report["stage"].tolist() == ["load_field_of_view"] * 2
