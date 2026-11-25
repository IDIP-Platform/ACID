import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

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
    merge_results,
)
from acid.pipeline.records import (
    list_fov_names,
    make_batches,
    read_records,
    select_stage_rows,
)
from acid.segmentation.pipeline import (
    apply_segmentation_batch,
    update_metadata_with_segmentation_results,
)
from acid.utils.fov_axis_utils import get_fov_ch_shape
from acid.utils.metadata.filtering import filter_metadata_by_splits


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


def run_batched_workflow(config, part_2, model, directory, batch_size):
    """Run parts 2-6 as the Snakemake workflow does, one job per FOV batch."""
    fovs = list_fov_names(
        select_stage_rows(pd.read_csv(part_2), "quality_control", config), config
    )
    batches = make_batches(fovs, batch_size)

    def per_fov(job, stage, input_metadata, **kwargs):
        return [
            job(
                config,
                input_metadata,
                batch,
                directory / stage / f"batch_{i:04d}.json",
                **kwargs,
            )
            for i, batch in enumerate(batches)
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


@pytest.mark.parametrize("batch_size", [1, 2])
def test_batched_workflow_matches_notebook_workflow(
    make_pipeline_config,
    make_pipeline_dataset,
    permissive_qc,
    square_model,
    tmp_path,
    batch_size,
):
    config = make_pipeline_config("per_fov", permissive_qc)
    reference_config = make_pipeline_config("notebooks", permissive_qc)
    part_2 = make_pipeline_dataset(config)
    shutil.copytree(
        config.shared.paths.extracted_fov_dir,
        reference_config.shared.paths.extracted_fov_dir,
    )
    (tmp_path / "reference").mkdir()
    (tmp_path / "per_fov").mkdir()

    expected = run_notebook_workflow(
        reference_config, part_2, square_model, tmp_path / "reference"
    )
    result = run_batched_workflow(
        config, part_2, square_model, tmp_path / "per_fov", batch_size
    )

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
    qc_files = sorted((tmp_path / "per_fov" / "qc").iterdir())
    assert len(qc_files) == 2 // batch_size
    assert [r["row_index"] for r in read_records(qc_files)] == [0, 2]
