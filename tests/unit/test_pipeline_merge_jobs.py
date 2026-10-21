"""Merge jobs, and the per-FOV workflow compared with the notebook workflow."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from conftest import PERMISSIVE_QC, write_pipeline_dataset

from acid.feature_extraction.pipeline import (
    extract_features_batch,
    update_metadata_with_feature_results,
)
from acid.image_processing.background.apply_background_correction import (
    apply_background_correction_batch,
    update_metadata_with_correction_results,
)
from acid.image_processing.background.compute_background_function import (
    calculate_backgrounds,
    save_backgrounds,
    update_background_metadata,
)
from acid.image_processing.background.load_background_function import (
    load_background_function,
)
from acid.image_quality_control.fov_quality import flag_quality, measure_quality
from acid.pipeline.fov_jobs import (
    correct_background_job,
    extract_features_job,
    measure_quality_job,
    segment_job,
)
from acid.pipeline.merge_jobs import (
    background_function_job,
    flag_quality_job,
    merge_fov_extraction,
    merge_results,
    split_dataset_job,
)
from acid.pipeline.records import (
    list_fov_names,
    read_record,
    select_stage_rows,
    write_record,
)
from acid.segmentation.pipeline import (
    apply_segmentation_batch,
    update_metadata_with_segmentation_results,
)
from acid.utils.fov_axis_utils import get_fov_ch_shape
from acid.utils.metadata.filtering import filter_metadata_by_splits


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
    record = write_record({"row_index": 0, "success": True}, tmp_path / "0.json")

    with pytest.raises(ValueError, match="1 result records do not match the 2"):
        merge_results(
            pipeline_config,
            "background_correction",
            pipeline_dataset,
            [record],
            tmp_path / "out.csv",
        )


def failure_record(row_index, directory):
    return write_record(
        {
            "row_index": row_index,
            "input_file": f"{row_index}.ome.tif",
            "output_file": None,
            "success": False,
            "stage": "load_field_of_view",
            "error_type": "OSError",
            "error_message": "unreadable",
        },
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


# ----------------------------------------------------------
# Per-FOV workflow == notebook workflow
# ----------------------------------------------------------


def run_notebook_workflow(config, part_2, model, directory):
    """Run parts 2-6 as the notebooks do; return the metadata of each part."""
    paths = config.shared.paths
    outputs = {}

    def round_trip(name):
        # Every notebook reads the previous stage's metadata back from CSV.
        path = directory / f"{name}.csv"
        outputs[name].to_csv(path, index=False)
        return pd.read_csv(path)

    qc = config.quality_control
    metadata_df = filter_metadata_by_splits(
        pd.read_csv(part_2), "train", qc.dataset_split
    )
    metadata_df, _, mean_columns, _ = measure_quality(metadata_df, qc)
    outputs["part_3"] = flag_quality(metadata_df, mean_columns, qc)

    bg = config.background_function_calculation
    metadata_df = round_trip("part_3")
    metadata_df_ok = metadata_df.loc[
        metadata_df[bg.metadata.dataframe_columns.flag_column_name]
        != qc.processing.flag_value
    ].copy()
    fov_shape, _, _ = get_fov_ch_shape(
        metadata_df_ok,
        bg.processing.fov_directory,
        fov_clm=bg.metadata.dataframe_columns.fov_column_name,
        channel_axis=bg.processing.channel_axis,
        null_value=bg.metadata.dataframe_columns.null_value,
    )
    Path(bg.image_saving.directory).mkdir(parents=True)
    backgrounds = calculate_backgrounds(metadata_df_ok, fov_shape, bg)
    save_backgrounds(backgrounds, bg, config.project_identity.project_name)
    outputs["part_4a"] = update_background_metadata(metadata_df, bg)

    bc = config.background_correction
    metadata_df = round_trip("part_4a")
    backgrounds, _ = load_background_function(
        bc.background_function_selection, metadata_df
    )
    Path(paths.corrected_fov_dir).mkdir(parents=True)
    results = apply_background_correction_batch(metadata_df, backgrounds, bc, paths)
    outputs["part_4b"] = update_metadata_with_correction_results(
        metadata_df, results, bc.metadata.dataframe_columns
    )

    seg = config.object_segmentation
    metadata_df = round_trip("part_4b")
    Path(paths.segmentation_masks_dir).mkdir(parents=True)
    results = apply_segmentation_batch(metadata_df, model, seg, paths)
    outputs["part_5"] = update_metadata_with_segmentation_results(
        metadata_df, results, seg.metadata.dataframe_columns
    )

    fe = config.feature_extraction
    metadata_df = round_trip("part_5")
    Path(paths.feature_tables_dir).mkdir(parents=True)
    results = extract_features_batch(metadata_df, fe, paths)
    outputs["part_6"] = update_metadata_with_feature_results(
        metadata_df, results, fe.metadata.dataframe_columns
    )
    round_trip("part_6")
    return {name: pd.read_csv(directory / f"{name}.csv") for name in outputs}


def run_per_fov_workflow(config, part_2, model, directory):
    """Run parts 2-6 as the Snakemake workflow does, one job per FOV."""
    fovs = list_fov_names(
        select_stage_rows(pd.read_csv(part_2), "quality_control", config), config
    )

    def per_fov(job, stage, input_metadata, **kwargs):
        return [
            job(
                config, input_metadata, fov, directory / stage / f"{fov}.json", **kwargs
            )
            for fov in fovs
        ]

    part_3 = flag_quality_job(
        config,
        part_2,
        per_fov(measure_quality_job, "qc", part_2),
        directory / "part_3.csv",
    )
    part_4a = background_function_job(config, part_3, directory / "part_4a.csv")
    part_4b = merge_results(
        config,
        "background_correction",
        part_4a,
        per_fov(correct_background_job, "correction", part_4a),
        directory / "part_4b.csv",
    )
    part_5 = merge_results(
        config,
        "object_segmentation",
        part_4b,
        per_fov(segment_job, "segmentation", part_4b, model=model),
        directory / "part_5.csv",
    )
    merge_results(
        config,
        "feature_extraction",
        part_5,
        per_fov(extract_features_job, "features", part_5),
        directory / "part_6.csv",
    )
    names = ("part_3", "part_4a", "part_4b", "part_5", "part_6")
    return {name: pd.read_csv(directory / f"{name}.csv") for name in names}


def without_date_columns(metadata_df):
    return metadata_df.drop(columns=[c for c in metadata_df if "date" in c])


def test_per_fov_workflow_matches_notebook_workflow(
    make_pipeline_config, square_model, tmp_path
):
    config = make_pipeline_config("per_fov", PERMISSIVE_QC)
    reference_config = make_pipeline_config("notebooks", PERMISSIVE_QC)
    part_2 = write_pipeline_dataset(config)
    shutil.copytree(
        config.shared.paths.extracted_fov_dir,
        reference_config.shared.paths.extracted_fov_dir,
    )
    (tmp_path / "reference").mkdir()
    (tmp_path / "per_fov").mkdir()

    expected = run_notebook_workflow(
        reference_config, part_2, square_model, tmp_path / "reference"
    )
    result = run_per_fov_workflow(config, part_2, square_model, tmp_path / "per_fov")

    assert expected["part_3"]["flag"].tolist() == [0, 0]
    assert expected["part_6"]["features_data_frame"].notna().all()
    for name in expected:
        pd.testing.assert_frame_equal(
            without_date_columns(result[name]),
            without_date_columns(expected[name]),
            obj=name,
        )
    for directory in ("corrected_fov_dir", "segmentation_masks_dir"):
        for file in Path(reference_config.shared.paths[directory]).iterdir():
            np.testing.assert_array_equal(
                tifffile.imread(Path(config.shared.paths[directory]) / file.name),
                tifffile.imread(file),
            )
    for file in Path(reference_config.shared.paths.feature_tables_dir).iterdir():
        pd.testing.assert_frame_equal(
            pd.read_csv(Path(config.shared.paths.feature_tables_dir) / file.name),
            pd.read_csv(file),
        )
    assert (
        read_record(tmp_path / "per_fov" / "qc" / "p_well1_A07p2_A1.json")["row_index"]
        == 0
    )
