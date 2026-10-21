"""Per-acquisition and per-FOV-batch jobs of the Snakemake workflow.

Each function is one Snakemake job: it processes a single raw acquisition or
a batch of fields of view (FOVs) exactly as the stage notebook processes each
of its rows, and writes the results to its own file. Expensive setup (the
segmentation model, the background functions, the metadata) is done once per
batch. The merge jobs in `acid.pipeline.merge_jobs` then combine these files
into the stage's metadata CSV.

Every job takes the whole configuration (`load_config`), calls
`prepare_output(config)` like the notebooks, reads its input metadata from an
explicit path and selects its rows with `select_stage_rows`, so that the
`row_index` of each result record matches the row the merge job updates.
A FOV that cannot be processed gives a failure record, not an exception.
"""

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from acid.config import prepare_output
from acid.feature_extraction.pipeline import extract_features_batch
from acid.image_processing.background.apply_background_correction import (
    apply_background_correction_batch,
)
from acid.image_processing.background.load_background_function import (
    load_background_function,
)
from acid.image_processing.fov_extraction import extract_acquisition
from acid.image_quality_control.fov_quality import measure_quality_for_fov
from acid.pipeline.records import select_fov_rows, select_stage_rows, write_records
from acid.segmentation.pipeline import apply_segmentation_batch
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
    config: DictConfig,
    input_metadata: str | Path,
    fovs: list[str],
    output_json: str | Path,
) -> Path:
    """Measure the QC statistics of a batch of fields of view (part2).

    The number of channels is taken from the first readable FOV of all
    selected rows, as in `measure_quality`.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `quality_control` section (see `measure_quality_for_fov`).
        input_metadata (str | Path): Dataset-splitting metadata (`part_2.csv`).
        fovs (list[str]): FOV names (see `acid.pipeline.records.fov_name`).
        output_json (str | Path): Result records, one per FOV, with keys
            `row_index`, `input_file`, `num_channels`, `plls`, `skewness` and
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
    rows = select_fov_rows(metadata_df, fovs, config)

    records = []
    for row_index, field_of_view_file in rows[columns.fov_column_name].items():
        measurement = measure_quality_for_fov(field_of_view_file, num_channels, cfg)
        records.append(
            {
                "row_index": row_index,
                "input_file": field_of_view_file,
                "num_channels": num_channels,
                **measurement,
            }
        )

    return write_records(records, output_json)


def correct_background_job(
    config: DictConfig,
    input_metadata: str | Path,
    fovs: list[str],
    output_json: str | Path,
) -> Path:
    """Correct the background of a batch of fields of view (part4).

    Loads the background functions once with `load_background_function`
    (using all selected rows, as the notebook does) and runs
    `apply_background_correction_batch` on the batch.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `background_correction` section and `shared.paths`.
        input_metadata (str | Path): Background-function metadata
            (`part_4a.csv`).
        fovs (list[str]): FOV names.
        output_json (str | Path): Result records of
            `apply_background_correction_for_fov`, one per FOV.

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
    rows = select_fov_rows(metadata_df, fovs, config)

    records = apply_background_correction_batch(
        rows, backgrounds, cfg, config.shared.paths
    )

    return write_records(records, output_json)


def segment_job(
    config: DictConfig,
    input_metadata: str | Path,
    fovs: list[str],
    output_json: str | Path,
    model=None,
) -> Path:
    """Segment a batch of fields of view with one model (part5).

    Args:
        config (DictConfig): The whole configuration. Reads the
            `object_segmentation` section and `shared.paths`.
        input_metadata (str | Path): Background-correction metadata
            (`part_4b.csv`).
        fovs (list[str]): FOV names.
        output_json (str | Path): Result records of
            `apply_segmentation_for_fov`, one per FOV.
        model: Segmentation model (see `segment_objects`). If `None`, it is
            created once with `create_segmentation_model` from
            `object_segmentation.processing.model_backend`.

    Returns:
        Path: `output_json`.
    """
    prepare_output(config)
    cfg = config.object_segmentation
    metadata_df = _load_stage_rows(input_metadata, "object_segmentation", config)
    rows = select_fov_rows(metadata_df, fovs, config)
    if model is None:
        # Imported here: torch/cellpose are slow to import and not needed elsewhere.
        from acid.segmentation.registry import create_segmentation_model

        model = create_segmentation_model(name=cfg.processing.model_backend)
    Path(config.shared.paths.segmentation_masks_dir).mkdir(parents=True, exist_ok=True)

    records = apply_segmentation_batch(rows, model, cfg, config.shared.paths)

    return write_records(records, output_json)


def extract_features_job(
    config: DictConfig,
    input_metadata: str | Path,
    fovs: list[str],
    output_json: str | Path,
) -> Path:
    """Extract the per-object features of a batch of fields of view (part6).

    Runs `extract_features_batch`, which turns an exception of one FOV into a
    failure record.

    Args:
        config (DictConfig): The whole configuration. Reads the
            `feature_extraction` section and `shared.paths`.
        input_metadata (str | Path): Segmentation metadata (`part_5.csv`).
        fovs (list[str]): FOV names.
        output_json (str | Path): Result records, one per FOV.

    Returns:
        Path: `output_json`.
    """
    prepare_output(config)
    metadata_df = _load_stage_rows(input_metadata, "feature_extraction", config)
    rows = select_fov_rows(metadata_df, fovs, config)
    Path(config.shared.paths.feature_tables_dir).mkdir(parents=True, exist_ok=True)

    records = extract_features_batch(
        rows, config=config.feature_extraction, paths=config.shared.paths
    )

    return write_records(records, output_json)


# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------


def _load_stage_rows(input_metadata, stage, config):
    return select_stage_rows(load_metadata_file(input_metadata), stage, config)
