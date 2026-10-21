"""Whole-stage and merge jobs of the Snakemake workflow.

Stages that need all fields of view at once (dataset splitting, QC
flagging, background-function calculation) run as one job, and the per-FOV
jobs of `acid.pipeline.fov_jobs` are combined here into the stage's metadata
CSV. Each function does what the stage notebook does after its processing
loop, reads its inputs from explicit paths and writes the metadata to an
explicit path (`save_metadata_dataframe(..., path=...)`).

Merging checks that there is exactly one result record per selected row, so
a missing or extra record is an error rather than a silent gap.
"""

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from acid.config import prepare_output
from acid.data_preparation.map_category import map_treatments
from acid.data_preparation.train_test_split import split_dataset_train_test
from acid.feature_extraction.pipeline import update_metadata_with_feature_results
from acid.image_processing.background.apply_background_correction import (
    update_metadata_with_correction_results,
)
from acid.image_processing.background.compute_background_function import (
    calculate_backgrounds,
    save_backgrounds,
    update_background_metadata,
)
from acid.image_quality_control.fov_quality import (
    add_quality_columns,
    flag_quality,
)
from acid.pipeline.records import read_records, select_stage_rows
from acid.segmentation.pipeline import update_metadata_with_segmentation_results
from acid.utils.fov_axis_utils import get_fov_ch_shape
from acid.utils.metadata.loading import load_metadata, load_metadata_file
from acid.utils.metadata.saving import save_metadata_dataframe

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

# Per-FOV stages whose records are merged by `merge_results`.
RESULT_UPDATERS = {
    "background_correction": update_metadata_with_correction_results,
    "object_segmentation": update_metadata_with_segmentation_results,
    "feature_extraction": update_metadata_with_feature_results,
}

FAILURE_REPORT_COLUMNS = [
    "row_index",
    "input_file",
    "stage",
    "error_type",
    "error_message",
]


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def merge_fov_extraction(
    config: DictConfig, acquisition_csvs: list[str | Path], output: str | Path
) -> Path:
    """Combine the per-acquisition metadata into the part0 metadata.

    The files are read as text and concatenated in the given order, so the
    result is the same as extracting all acquisitions in one loop
    (`extract_fields_of_view`) when the files are given in
    `discover_acquisitions` order. Empty files (acquisitions without scenes)
    are skipped.

    Args:
        config (DictConfig): The whole configuration. Reads
            `field_of_view_extraction.metadata`.
        acquisition_csvs (list[str | Path]): Outputs of
            `extract_acquisition_job`, in acquisition order.
        output (str | Path): Metadata CSV to write (`part_1.csv`).

    Returns:
        Path: The written file.

    Raises:
        ValueError: If no field of view was extracted.
    """
    prepare_output(config)
    frames = [
        pd.read_csv(path, dtype=str, keep_default_na=False)
        for path in acquisition_csvs
        if Path(path).stat().st_size > 0
    ]
    if not frames:
        raise ValueError("No fields of view were extracted")
    return _save(
        pd.concat(frames, ignore_index=True),
        config.field_of_view_extraction.metadata,
        config,
        output,
    )


def split_dataset_job(
    config: DictConfig, input_metadata: str | Path, output: str | Path
) -> Path:
    """Map treatments and split the dataset into train and test (part1).

    The plate layout is found with `dataset_splitting.plate_layout` file
    selection, as in the notebook.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `dataset_splitting` section (see `map_treatments` and
            `split_dataset_train_test`).
        input_metadata (str | Path): Extraction metadata (`part_1.csv`).
        output (str | Path): Metadata CSV to write (`part_2.csv`).

    Returns:
        Path: The written file.
    """
    prepare_output(config)
    cfg = config.dataset_splitting
    metadata_df = load_metadata_file(input_metadata)
    plate_layout_df, _ = load_metadata(cfg.plate_layout)

    metadata_df = map_treatments(metadata_df, plate_layout_df, cfg)
    metadata_df = split_dataset_train_test(metadata_df, cfg)
    logger.info(
        "Split sizes:\n%s", metadata_df[cfg.dataset_split.column].value_counts()
    )

    return _save(metadata_df, cfg.metadata, config, output)


def flag_quality_job(
    config: DictConfig,
    input_metadata: str | Path,
    records: list[str | Path],
    output: str | Path,
) -> Path:
    """Combine the per-FOV QC measurements, flag and plot them (part2).

    Writes the QC columns with `add_quality_columns`, as `measure_quality`
    does, flags the rows with `flag_quality` and, if
    `quality_control.graphs_saving.save_graphs` is true, saves the QC graphs
    with `plot_quality`.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `quality_control` section.
        input_metadata (str | Path): Dataset-splitting metadata (`part_2.csv`).
        records (list[str | Path]): Outputs of `measure_quality_job`, one per
            training row.
        output (str | Path): Metadata CSV to write (`part_3.csv`).

    Returns:
        Path: The written file.

    Raises:
        ValueError: If no training row is selected, the records do not match
            the selected rows one to one, or they disagree on the number of
            channels.
    """
    prepare_output(config)
    cfg = config.quality_control
    metadata_df = select_stage_rows(
        load_metadata_file(input_metadata), "quality_control", config
    )
    if metadata_df.empty:
        raise ValueError("No training images selected")
    results = _records_by_row(read_records(records), metadata_df)
    channel_counts = {result["num_channels"] for result in results}
    if len(channel_counts) != 1:
        raise ValueError(f"QC records disagree on the channel count: {channel_counts}")
    (num_channels,) = channel_counts

    column_names = add_quality_columns(metadata_df, results, num_channels, cfg)
    flagged_df = flag_quality(metadata_df, column_names["mean_over_std"], cfg)

    if cfg.graphs_saving.save_graphs:
        _plot_quality(flagged_df, column_names, num_channels, config)

    return _save(flagged_df, cfg.metadata, config, output)


def background_function_job(
    config: DictConfig,
    input_metadata: str | Path,
    output: str | Path,
    threads: int | None = None,
) -> Path:
    """Calculate and save the background functions (part3).

    Uses the unflagged training rows, as the notebook does, and records the
    settings in the metadata of all training rows.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `background_function_calculation` section and
            `quality_control.processing.flag_value`.
        input_metadata (str | Path): QC metadata (`part_3.csv`).
        output (str | Path): Metadata CSV to write (`part_4a.csv`).
        threads (int | None): Threads for reading and averaging the FOVs (see
            `calculate_backgrounds`); `None` uses the available CPUs.

    Returns:
        Path: The written file. The background images are saved into
        `background_function_calculation.image_saving.directory`.

    Raises:
        ValueError: If no unflagged training row is left.
    """
    prepare_output(config)
    cfg = config.background_function_calculation
    columns = cfg.metadata.dataframe_columns
    metadata_df = select_stage_rows(
        load_metadata_file(input_metadata), "background_function_calculation", config
    )
    metadata_df_ok = metadata_df.loc[
        metadata_df[columns.flag_column_name]
        != config.quality_control.processing.flag_value
    ].copy()
    if metadata_df_ok.empty:
        raise ValueError(
            "No unflagged training images available for background estimation"
        )
    fov_shape, _num_channels, _shape_of_channels = get_fov_ch_shape(
        metadata_df_ok,
        cfg.processing.fov_directory,
        fov_clm=columns.fov_column_name,
        channel_axis=cfg.processing.channel_axis,
        null_value=columns.null_value,
    )
    Path(cfg.image_saving.directory).mkdir(parents=True, exist_ok=True)

    backgrounds = calculate_backgrounds(metadata_df_ok, fov_shape, cfg, threads)
    for result in save_backgrounds(
        backgrounds, cfg, config.project_identity.project_name
    ):
        logger.info("Saved background %s", result)
    metadata_df_updated = update_background_metadata(metadata_df, cfg)

    return _save(metadata_df_updated, cfg.metadata, config, output)


def merge_results(
    config: DictConfig,
    stage: str,
    input_metadata: str | Path,
    records: list[str | Path],
    output: str | Path,
) -> Path:
    """Write the per-FOV results of a stage into its metadata (parts 4–6).

    Failed rows keep `null_value` in the stage's columns, as in the
    notebooks, and are also listed in
    `<shared.output_root>/reports/<stage>_failures.csv` (written even when
    empty).

    Args:
        config (DictConfig): The whole configuration. Reads
            `<stage>.metadata`.
        stage (str): `"background_correction"`, `"object_segmentation"` or
            `"feature_extraction"`.
        input_metadata (str | Path): Input metadata of the stage
            (`part_4a.csv`, `part_4b.csv` or `part_5.csv`).
        records (list[str | Path]): Result records of the stage's per-FOV
            jobs, one per selected row.
        output (str | Path): Metadata CSV to write (`part_4b.csv`,
            `part_5.csv` or `part_6.csv`).

    Returns:
        Path: The written file.

    Raises:
        KeyError: If `stage` is not a per-FOV stage.
        ValueError: If the records do not match the selected rows one to one,
            or if every row failed.
    """
    prepare_output(config)
    update = RESULT_UPDATERS[stage]
    metadata_cfg = config[stage].metadata
    metadata_df = select_stage_rows(load_metadata_file(input_metadata), stage, config)
    results = _records_by_row(read_records(records), metadata_df)

    failures = [result for result in results if not result["success"]]
    _write_failure_report(failures, stage, config)
    if failures and len(failures) == len(results):
        raise ValueError(f"{stage}: every field of view failed")

    metadata_df_updated = update(
        metadata_df=metadata_df,
        results=results,
        dataframe_columns=metadata_cfg.dataframe_columns,
    )

    return _save(metadata_df_updated, metadata_cfg, config, output)


# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------


def _save(metadata_df, metadata_cfg, config, output):
    return save_metadata_dataframe(
        metadata_df,
        metadata_cfg,
        config.project_identity.project_name,
        path=output,
    )


def _records_by_row(results, metadata_df):
    """Return the records in row order, checking one record per row."""
    by_row = {result["row_index"]: result for result in results}
    if len(by_row) != len(results) or set(by_row) != set(metadata_df.index):
        raise ValueError(
            f"{len(results)} result records do not match the "
            f"{len(metadata_df)} selected metadata rows"
        )
    return [by_row[row_index] for row_index in metadata_df.index]


def _write_failure_report(failures, stage, config):
    path = Path(config.shared.output_root) / "reports" / f"{stage}_failures.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=FAILURE_REPORT_COLUMNS).to_csv(path, index=False)
    if failures:
        logger.warning(
            "%s: %d field(s) of view failed, see %s", stage, len(failures), path
        )


def _plot_quality(flagged_df, column_names, num_channels, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from acid.image_quality_control.display_qc import plot_quality

    plot_quality(
        flagged_df,
        column_names["mean_over_std"],
        column_names["skewness"],
        num_channels,
        config.quality_control,
        config.project_identity.project_name,
    )
    plt.close("all")
