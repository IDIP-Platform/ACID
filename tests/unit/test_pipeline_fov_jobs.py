from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acid.feature_extraction.pipeline import extract_features_batch
from acid.image_processing.background.apply_background_correction import (
    apply_background_correction_batch,
    update_metadata_with_correction_results,
)
from acid.image_quality_control.fov_quality import (
    get_quality_column_names,
    measure_quality,
)
from acid.pipeline import fov_jobs
from acid.pipeline.fov_jobs import (
    correct_background_job,
    extract_acquisition_job,
    extract_features_job,
    measure_quality_job,
    segment_job,
)
from acid.pipeline.records import read_records, select_stage_rows
from acid.segmentation.pipeline import (
    apply_segmentation_batch,
    update_metadata_with_segmentation_results,
)
from acid.utils.save_image import tifffile_save_ometiff


def without_dates(record):
    """Drop processing dates, which differ when a test runs across midnight."""
    return {key: value for key, value in record.items() if "date" not in key}


def write_background(config):
    directory = Path(config.shared.paths.background_functions_dir)
    directory.mkdir(parents=True)
    tifffile_save_ometiff(
        directory / "20260901_ACID_background.ome.tif",
        data=np.full((5, 64, 64), 1000.0, dtype=np.float32),
        imagej=True,
        photometric="minisblack",
        metadata={},
    )


def corrected_metadata(config, metadata_path, tmp_path):
    """Run the batch background correction and save its metadata (part_4b)."""
    from acid.image_processing.background.load_background_function import (
        load_background_function,
    )

    write_background(config)
    metadata_df = select_stage_rows(
        pd.read_csv(metadata_path), "background_correction", config
    )
    backgrounds, _ = load_background_function(
        config.background_correction.background_function_selection, metadata_df
    )
    results = apply_background_correction_batch(
        metadata_df, backgrounds, config.background_correction, config.shared.paths
    )
    updated = update_metadata_with_correction_results(
        metadata_df, results, config.background_correction.metadata.dataframe_columns
    )
    path = tmp_path / "part_4b.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(path, index=False)
    return path


def test_extract_acquisition_job_writes_scene_rows(
    pipeline_config, tmp_path, monkeypatch
):
    scenes = [
        pd.Series({"ome_tif_file_name": "a_A1.ome.tif", "scene_name": "A1"}),
        pd.Series({"ome_tif_file_name": "a_A2.ome.tif", "scene_name": "A2"}),
    ]
    monkeypatch.setattr(fov_jobs, "extract_acquisition", lambda path, cfg: scenes)

    output = extract_acquisition_job(
        pipeline_config, tmp_path / "raw" / "a.nd2", tmp_path / "work" / "a.csv"
    )

    assert pd.read_csv(output)["scene_name"].tolist() == ["A1", "A2"]
    assert Path(
        pipeline_config.field_of_view_extraction.image_saving.directory
    ).is_dir()


def test_extract_acquisition_job_without_scenes_writes_empty_file(
    pipeline_config, tmp_path, monkeypatch
):
    monkeypatch.setattr(fov_jobs, "extract_acquisition", lambda path, cfg: [])

    output = extract_acquisition_job(pipeline_config, "a.nd2", tmp_path / "a.csv")

    assert output.read_text() == ""


def test_measure_quality_job_matches_measure_quality(
    pipeline_config, pipeline_dataset, tmp_path
):
    metadata_df = select_stage_rows(
        pd.read_csv(pipeline_dataset), "quality_control", pipeline_config
    )
    measured, num_channels, _, _ = measure_quality(
        metadata_df, pipeline_config.quality_control
    )

    (record,) = read_records(
        [
            measure_quality_job(
                pipeline_config,
                pipeline_dataset,
                ["p_well2_A07p2_A1"],
                tmp_path / "r.json",
            )
        ]
    )

    assert record["row_index"] == 2
    assert record["num_channels"] == num_channels == 5
    names = get_quality_column_names(5, pipeline_config.quality_control)
    for statistic, columns in names.items():
        assert record[statistic] == measured.loc[2, columns].tolist()


def test_measure_quality_job_rejects_fov_outside_training_split(
    pipeline_config, pipeline_dataset, tmp_path
):
    with pytest.raises(KeyError, match="p_well1_A07p2_A2"):
        measure_quality_job(
            pipeline_config, pipeline_dataset, ["p_well1_A07p2_A2"], tmp_path / "r.json"
        )


def test_correct_background_job_matches_batch(
    pipeline_config, pipeline_dataset, tmp_path
):
    corrected_metadata(pipeline_config, pipeline_dataset, tmp_path / "batch")
    batch = pd.read_csv(tmp_path / "batch" / "part_4b.csv")

    (record,) = read_records(
        [
            correct_background_job(
                pipeline_config,
                pipeline_dataset,
                ["p_well2_A07p2_A1"],
                tmp_path / "r.json",
            )
        ]
    )

    assert record["success"] is True
    assert record["row_index"] == 2
    columns = pipeline_config.background_correction.metadata.dataframe_columns
    assert (
        record["output_file"]
        == batch.loc[1, columns.illum_correct_df_file_name_clm_name]
    )
    assert (
        Path(pipeline_config.shared.paths.corrected_fov_dir) / record["output_file"]
    ).is_file()


def test_segment_job_matches_batch(
    pipeline_config, pipeline_dataset, square_model, tmp_path
):
    part_4b = corrected_metadata(pipeline_config, pipeline_dataset, tmp_path)
    metadata_df = select_stage_rows(
        pd.read_csv(part_4b), "object_segmentation", pipeline_config
    )
    (expected, _) = apply_segmentation_batch(
        metadata_df,
        square_model,
        pipeline_config.object_segmentation,
        pipeline_config.shared.paths,
    )

    (record,) = read_records(
        [
            segment_job(
                pipeline_config,
                part_4b,
                ["p_well1_A07p2_A1"],
                tmp_path / "r.json",
                model=square_model,
            )
        ]
    )

    assert record["success"] is True
    assert without_dates(record) == without_dates(expected)


def test_extract_features_job_matches_batch(
    pipeline_config, pipeline_dataset, square_model, tmp_path
):
    part_4b = corrected_metadata(pipeline_config, pipeline_dataset, tmp_path)
    metadata_df = select_stage_rows(
        pd.read_csv(part_4b), "object_segmentation", pipeline_config
    )
    results = apply_segmentation_batch(
        metadata_df,
        square_model,
        pipeline_config.object_segmentation,
        pipeline_config.shared.paths,
    )
    part_5 = tmp_path / "part_5.csv"
    update_metadata_with_segmentation_results(
        metadata_df,
        results,
        pipeline_config.object_segmentation.metadata.dataframe_columns,
    ).to_csv(part_5, index=False)
    (_, expected) = extract_features_batch(
        pd.read_csv(part_5),
        pipeline_config.feature_extraction,
        pipeline_config.shared.paths,
    )

    (record,) = read_records(
        [
            extract_features_job(
                pipeline_config, part_5, ["p_well2_A07p2_A1"], tmp_path / "r.json"
            )
        ]
    )

    assert record["success"] is True
    assert record["row_index"] == 1
    assert without_dates(record) == without_dates(expected)


def test_segment_job_processes_a_batch_in_the_given_order(
    pipeline_config, pipeline_dataset, square_model, tmp_path
):
    part_4b = corrected_metadata(pipeline_config, pipeline_dataset, tmp_path)
    metadata_df = select_stage_rows(
        pd.read_csv(part_4b), "object_segmentation", pipeline_config
    )
    expected = apply_segmentation_batch(
        metadata_df,
        square_model,
        pipeline_config.object_segmentation,
        pipeline_config.shared.paths,
    )

    records = read_records(
        [
            segment_job(
                pipeline_config,
                part_4b,
                ["p_well2_A07p2_A1", "p_well1_A07p2_A1"],
                tmp_path / "r.json",
                model=square_model,
            )
        ]
    )

    assert [without_dates(r) for r in records] == [
        without_dates(r) for r in reversed(expected)
    ]
