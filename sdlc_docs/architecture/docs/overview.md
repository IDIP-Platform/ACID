# ACID architecture: current state

ACID processes fixed-sample microscopy acquisitions. These notes describe the checked-in implementation, not a target design or a claim that its runners produce equivalent scientific results. There is no web app or HTTP API. Cellpose runs in-process; the SLURM profile is an unconfigured template.

## Views

| View | Purpose |
| --- | --- |
| `ExecutionPaths` | C4 container view: people, notebook kernels, Snakemake, batch workers, and file boundaries. It includes the context needed here, so a separate context diagram adds little. |
| `BatchArtifacts` | Snakemake rule groups and their CSV/OME-TIFF handoffs. Arrows represent files passed through the filesystem, not function calls. |

The source building blocks are mapped to code in the table below. A Python package is imported by several processes; it is not a separately running C4 container.

In `ExecutionPaths`, store names state their roles: experiment inputs, configuration, and pipeline outputs. Baseline settings are inline notebook variables, so no separate baseline configuration file is shown. The diagram key explains the colors and shapes.

## Execution paths

| Runner | Current entry and configuration | File contract |
| --- | --- | --- |
| `notebooks/` | Ten notebooks with mostly inline logic and notebook-specific paths; QC, segmentation, and feature notebooks import selected `acid` helpers. | Notebook-specific filenames and outputs, including checked-in Windows paths, `data/proc`, and `secondary_output`. |
| `notebooks_refactored/` | Seven notebooks call `acid` stages. All checked-in `CONFIG_FILE` values are `None`, selecting packaged defaults. | Select newest matching metadata CSV; save dated CSVs under `data/processed/default` by default. |
| `workflow/Snakefile` | Requires `acid_config=configs/<id>.yaml`. Snakemake imports `acid` during DAG construction, then starts `python -m acid.pipeline` workers. | Explicit input/output paths; batch JSON records merge into undated stage CSVs under the configured output root. |

The baseline is not independent of the current branch because three notebooks import its `src/acid` helpers. The runners also have different checked-in settings and file-selection rules; compare results only after aligning those inputs.

## Stage migration map

| Stage | Baseline notebook | Refactored notebook | `src/acid` entry functions | Snakemake rules |
| --- | --- | --- | --- | --- |
| FOV extraction | `part1_extract_field_of_view` | `part0_extract_field_of_view` | `fov_extraction.discover_acquisitions`, `extract_fields_of_view` | `extract_acquisition`, `merge_extraction` |
| Dataset split | `part2_split_train_test` | `part1_split_train_test` | `map_category.map_treatments`, `train_test_split.split_dataset_train_test` | `split` checkpoint |
| Quality | `part3_control_image_quality` | `part2_control_image_quality` | `fov_quality.measure_quality`, `flag_quality` | `measure_quality`, `flag_quality` |
| Background estimate | `part4a_compute_background_function` | `part3_compute_background_function` | `background.compute_background_function.calculate_backgrounds`, `save_backgrounds` | `background_function` |
| Background correction | `part4b_correct_background` | `part4_background_correction_pipeline` | `background.load_background_function`, `apply_background_correction_batch` | `correct_background`, `merge_background_correction` |
| Segmentation | `part5_segment_object` | `part5_segment_object` | `segmentation.registry.create_segmentation_model`, `pipeline.apply_segmentation_batch` | `segment`, `merge_segmentation` |
| Features | `part6_extract_feature` | `part6_extract_feature` | `feature_extraction.pipeline.extract_features_batch` | `extract_features`, `merge_features` |

Three additional baseline notebooks (`part3extra_*`, `part4extra_*`) are diagnostics outside the main sequence. Snakemake has adapter and merge logic that the notebook cells do not have; stage names alone do not establish execution or numerical parity.

## Source building blocks

The paths below are relative to `src/acid/`. This table is the maintained mapping from code to responsibility; the diagrams do not duplicate it.

| Source | Responsibility |
| --- | --- |
| `config/{__init__,loader}.py`, `config/default.yaml` | Load settings and guard output configuration. |
| `pipeline/{__main__,cli,fov_jobs,merge_jobs,records}.py` | Batch commands, per-acquisition/FOV jobs, whole-stage jobs, row selection, and result records. |
| `image_processing/fov_extraction.py` | Discover acquisitions and extract scenes. |
| `data_preparation/{map_category,train_test_split,format_str}.py` | Map treatments and assign train/test rows. |
| `image_quality_control/{fov_quality,flag_column,measure_mean_over_std,measure_plls,measure_percentil_fraction_stat,display_qc}.py` | FOV QC measurement, flags, and plots. |
| `image_measurement/measure_glob_image_stat.py` | Global image statistics used by QC. |
| `image_processing/background/*.py` | Estimate, load, and apply background functions. |
| `image_processing/{extract_metadata,make_imagej_metadata,name_metadata,save_metadata,calculate_background_function,correct_background,segmentation_preprocessing,filter_image,resize_image}.py` | OME metadata and scientific image-processing helpers. |
| `segmentation/{pipeline,registry,base}.py`, `segmentation/models/cellpose.py` | Mask pipeline, model selection/interface, and Cellpose adapter. |
| `feature_extraction/{pipeline,default_regionprops,extra_regionprops,measure_hessian_matrix,measure_structure_tensor,measure_object_instensity}.py` | Per-object feature tables and their measures. |
| `io/image_loading.py` | Load FOV images and masks. |
| `utils/metadata/*.py` | Select, read, and save metadata CSVs; resolve row filenames. |
| Selected `utils/*` helpers | Axes, acquisition opening, image writing, rows, filesystem, and labels. |
| `image_visualization/display_illumination_correction.py` | Optional correction plots in the refactored notebook. |

These are traced runner paths, not an exhaustive reachability audit. Other files in these directories are not classified as removable. The shared `image_processing` helpers remain dependencies of refactored stages despite their older names.

## Handoffs and refactor constraints

`BatchArtifacts` shows the batch metadata chain from part 1 through part 6, including part 4a and 4b. After the split checkpoint, batch jobs select training FOVs; background estimation uses unflagged training rows. The Snakefile also uses `acid` configuration, discovery, record selection, and metadata naming while building the DAG, so changes there can fail before a worker starts. See `workflow/Snakefile`, `workflow/rules/*.smk`, and `src/acid/pipeline/records.py` for the current contract.

The declared `acid` console script targets a missing `acid.main`; `python -m acid.pipeline` is the working batch entry point. The `config.resolved.yaml` guard checks current settings but does not prove what settings produced existing files. Notebook tests check structure and compilation, workflow tests dry-run Snakemake, and synthetic equivalence tests cover parts 2–6 with a fake segmentation model. They do not prove real-data end-to-end equivalence or future multi-timepoint support.
