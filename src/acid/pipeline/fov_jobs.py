"""Per-acquisition and per-field-of-view jobs of the Snakemake workflow.

Each function is one Snakemake job: it processes a single raw acquisition or
a single field of view (FOV) exactly as the stage notebook processes each of
its rows, and writes its result to its own file. The merge jobs in
`acid.pipeline.merge_jobs` then combine these files into the stage's
metadata CSV.

Every job takes the whole configuration (`load_config`), calls
`prepare_output(config)` like the notebooks, reads its input metadata from an
explicit path and selects its rows with `select_stage_rows`, so that the
`row_index` of its result record matches the row the merge job updates.
A FOV that cannot be processed gives a failure record, not an exception.
"""

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from acid.config import prepare_output
from acid.feature_extraction.pipeline import extract_features_batch
from acid.image_processing.background.apply_background_correction import (
    apply_background_correction_for_fov,
)
from acid.image_processing.background.load_background_function import (
    load_background_function,
)
from acid.image_processing.fov_extraction import extract_acquisition
from acid.image_quality_control.fov_quality import measure_quality_for_fov
from acid.pipeline.records import select_fov_row, select_stage_rows, write_record
from acid.segmentation.pipeline import apply_segmentation_for_fov
from acid.utils.fov_axis_utils import get_fov_ch_shape
from acid.utils.metadata.loading import load_metadata_file

# ---- Setting built-in logging
logger = logging.getLogger(__name__)


# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


def extract_acquisition_job(
    config: DictConfig, acquisition: str | Path, output_csv: str | Path
) -> Path:
    """Extract the fields of view of one raw acquisition (part0, one `.nd2`).

    Saves one OME-TIFF per scene into
    `field_of_view_extraction.image_saving.directory` (see
    `extract_acquisition`) and writes the scenes' metadata rows to
    `output_csv`. An acquisition without scenes gives an empty file.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `field_of_view_extraction` section; `image_metadata.channels`
            must list exactly five channel names.
        acquisition (str | Path): Raw acquisition file inside its experiment
            directory.
        output_csv (str | Path): Metadata CSV of this acquisition.

    Returns:
        Path: `output_csv`.

    Raises:
        ValueError: If not exactly five channels are configured.
    """
    prepare_output(config)
    cfg = config.field_of_view_extraction
    if len(cfg.image_metadata.channels) != 5:
        raise ValueError("This extraction workflow requires five configured channels")
    Path(cfg.image_saving.directory).mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    scenes = extract_acquisition(Path(acquisition), cfg)
    if not scenes:
        logger.warning("No scenes in %s", acquisition)
        output_csv.write_text("")
        return output_csv

    pd.concat(scenes, axis=1).T.to_csv(output_csv, index=False)
    return output_csv


def measure_quality_job(
    config: DictConfig, input_metadata: str | Path, fov: str, output_json: str | Path
) -> Path:
    """Measure the QC statistics of one field of view (part2).

    The number of channels is taken from the first readable FOV of the
    selected rows, as in `measure_quality`.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `quality_control` section (see `measure_quality_for_fov`).
        input_metadata (str | Path): Dataset-splitting metadata (`part_2.csv`).
        fov (str): FOV name (see `acid.pipeline.records.fov_name`).
        output_json (str | Path): Result record with keys `row_index`,
            `input_file`, `num_channels`, `plls`, `skewness` and
            `mean_over_std` (one value per channel).

    Returns:
        Path: `output_json`.
    """
    prepare_output(config)
    cfg = config.quality_control
    columns = cfg.metadata.dataframe_columns
    metadata_df = _load_stage_rows(input_metadata, "quality_control", config)
    _fov_shape, num_channels, _shape_of_channels = get_fov_ch_shape(
        metadata_df,
        cfg.processing.fov_directory,
        fov_clm=columns.fov_column_name,
        channel_axis=cfg.processing.channel_axis,
        null_value=columns.null_value,
    )
    row = select_fov_row(metadata_df, fov, config)
    row_index = row.index[0]
    field_of_view_file = row.loc[row_index, columns.fov_column_name]

    measurement = measure_quality_for_fov(field_of_view_file, num_channels, cfg)

    return write_record(
        {
            "row_index": row_index,
            "input_file": field_of_view_file,
            "num_channels": num_channels,
            **measurement,
        },
        output_json,
    )


def correct_background_job(
    config: DictConfig, input_metadata: str | Path, fov: str, output_json: str | Path
) -> Path:
    """Correct the background of one field of view (part4).

    Loads the background functions with `load_background_function` (using
    all selected rows, as the notebook does) and runs
    `apply_background_correction_for_fov`.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `background_correction` section and `shared.paths`.
        input_metadata (str | Path): Background-function metadata
            (`part_4a.csv`).
        fov (str): FOV name.
        output_json (str | Path): Result record of
            `apply_background_correction_for_fov`.

    Returns:
        Path: `output_json`.
    """
    prepare_output(config)
    cfg = config.background_correction
    metadata_df = _load_stage_rows(input_metadata, "background_correction", config)
    backgrounds, _ = load_background_function(
        background_config=cfg.background_function_selection, metadata_df=metadata_df
    )
    Path(config.shared.paths.corrected_fov_dir).mkdir(parents=True, exist_ok=True)
    row = select_fov_row(metadata_df, fov, config)
    row_index = row.index[0]

    record = apply_background_correction_for_fov(
        row_index, row.loc[row_index], backgrounds, cfg, config.shared.paths
    )

    return write_record(record, output_json)


def segment_job(
    config: DictConfig,
    input_metadata: str | Path,
    fov: str,
    output_json: str | Path,
    model=None,
) -> Path:
    """Segment one field of view (part5).

    Args:
        config (DictConfig): The whole configuration. Reads the
            `object_segmentation` section and `shared.paths`.
        input_metadata (str | Path): Background-correction metadata
            (`part_4b.csv`).
        fov (str): FOV name.
        output_json (str | Path): Result record of
            `apply_segmentation_for_fov`.
        model: Segmentation model (see `segment_objects`). If `None`, it is
            created with `create_segmentation_model` from
            `object_segmentation.processing.model_backend`.

    Returns:
        Path: `output_json`.
    """
    prepare_output(config)
    cfg = config.object_segmentation
    metadata_df = _load_stage_rows(input_metadata, "object_segmentation", config)
    row = select_fov_row(metadata_df, fov, config)
    row_index = row.index[0]
    if model is None:
        # Imported here: torch/cellpose are slow to import and not needed elsewhere.
        from acid.segmentation.registry import create_segmentation_model

        model = create_segmentation_model(name=cfg.processing.model_backend)
    Path(config.shared.paths.segmentation_masks_dir).mkdir(parents=True, exist_ok=True)

    record = apply_segmentation_for_fov(
        row_index, row.loc[row_index], model, cfg, config.shared.paths
    )

    return write_record(record, output_json)


def extract_features_job(
    config: DictConfig, input_metadata: str | Path, fov: str, output_json: str | Path
) -> Path:
    """Extract the per-object features of one field of view (part6).

    Runs `extract_features_batch` on the single row, which turns an exception
    into a failure record.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `feature_extraction` section and `shared.paths`.
        input_metadata (str | Path): Segmentation metadata (`part_5.csv`).
        fov (str): FOV name.
        output_json (str | Path): Result record of the row.

    Returns:
        Path: `output_json`.
    """
    prepare_output(config)
    metadata_df = _load_stage_rows(input_metadata, "feature_extraction", config)
    row = select_fov_row(metadata_df, fov, config)
    Path(config.shared.paths.feature_tables_dir).mkdir(parents=True, exist_ok=True)

    (record,) = extract_features_batch(
        row, config=config.feature_extraction, paths=config.shared.paths
    )

    return write_record(record, output_json)


# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------


def _load_stage_rows(input_metadata, stage, config):
    return select_stage_rows(load_metadata_file(input_metadata), stage, config)
