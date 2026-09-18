# Configuration

Use this guide to configure **background correction (4b), segmentation (5), and
feature extraction (6)** in `notebooks_refactored/config.yaml`. A field of view
(FOV) is one image processed by the pipeline.

**Current value** means the value shipped in the YAML, including resolved shared
references. It is not necessarily a function's fallback default or a suitable
scientific choice for every dataset. Accepted values below describe the current
consumers; recommendations are marked explicitly. There is no complete schema
validation when loading this file: some errors appear only during processing.

## Start here

1. **Set your directories** in `shared.paths`. Locate extracted images, background
   functions, and metadata from stage 4a. Choose where to write corrected images,
   masks, and feature tables.
2. **Choose metadata inputs** in each stage's `metadata.file_selection`. For a
   reproducible run, pin an exact filename separately for each stage.
3. **Check image channels and processing settings.** Verify `channel_axis`, the
   segmentation channel positions, and the background offset against your images.
   Review stage 6's smoothing axis before interpreting intensity features.
4. **Run 4b → 5 → 6**, executing each notebook from top to bottom. After changing
   the YAML, rerun configuration loading and all dependent processing cells.
5. **Check outputs before continuing.** Inspect the per-image `results` records
   for failures, verify that output filenames in metadata are populated, and
   inspect corrected images and masks before extracting features.

### Inputs and outputs

The names below are metadata suffixes used for automatic selection, not complete
filenames. With the shipped naming settings, a metadata filename resembles
`20260918_ACID_metadata_part_4b.csv`.

<div class="configuration-table" role="region" aria-label="Inputs and outputs parameters 1" tabindex="0" markdown="1">

| Stage | Input metadata and images | Outputs |
| --- | --- | --- |
| **4b · Background correction** | `part_4a.csv`, extracted FOV images, previously computed background functions | Corrected images in `corrected_fov_dir`; `part_4b.csv` in `metadata_dir`; optional correction plots |
| **5 · Segmentation** | `part_4b.csv` and corrected images | One object-label mask per successful FOV in `segmentation_masks_dir`; `part_5.csv` |
| **6 · Feature extraction** | `part_5.csv`, corrected images, and matching masks | One feature CSV per successful FOV in `feature_tables_dir`; `part_6.csv` |

</div>


Run these notebooks in order:

- **4b:** `notebooks_refactored/part4_background_correction_pipeline.ipynb`
- **5:** `notebooks_refactored/part5_segment_object.ipynb`
- **6:** `notebooks_refactored/part6_extract_feature copy.ipynb`

Stages 4b and 5 explicitly select `"train"` in notebook code. Stage 6 processes
all supplied metadata rows. Changing dataset label codes does **not** change which
split those earlier notebooks select. Failed rows can remain in saved metadata
with missing output references; a saved CSV alone does not prove every FOV succeeded.

### Loading, paths, and references

The notebooks first look for `config.yaml` in the working directory, then for
`notebooks_refactored/config.yaml`. Run from the repository root or the refactored
notebook directory, and ensure the selected file is the one you intended.

OmegaConf loads the file and resolves references such as
`${shared.paths.metadata_dir}`. Editing a shared value affects settings that still
reference it. Replacing a reference with a literal changes only that setting.

The loading cell makes **`shared.paths` relative to the configuration directory**.
For example, `data/processed/corrected_fov` resolves to
`notebooks_refactored/data/processed/corrected_fov`. Use
`../data/processed/corrected_fov` for a repository-level data directory, or an
absolute path. Arbitrary literal paths elsewhere in the YAML are not normalized
by this loop; keep their shared references or use absolute paths when overriding.

Use YAML `true`/`false` for booleans, `null` for no value, and `.nan` for the missing
numeric marker. Quoted `"null"` is a string, not an absent value. The sections below
show parent paths so you can locate each short parameter name precisely.

## Shared settings

On smaller screens, scroll tables sideways to see all columns. You can also focus
a table with Tab and use the arrow keys.
{ .table-scroll-hint }

### Directories

**YAML path:** `shared.paths` · All values are directory-path strings.

<div class="configuration-table" role="region" aria-label="Directories parameters 2" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `extracted_fov_dir` | Existing image directory | `data/processed/extracted_fov` | Stage 4b input images. |
| `background_functions_dir` | Existing background directory | `data/processed/background_functions` | Stage 4b background images; no background estimation is performed by 4b. |
| `corrected_fov_dir` | Writable output / subsequent input directory | `data/processed/corrected_fov` | Stage 4b output; stages 5 and 6 input. |
| `segmentation_masks_dir` | Writable output / subsequent input directory | `data/processed/segmentation_masks` | Stage 5 output; stage 6 input. |
| `feature_tables_dir` | Writable output directory | `data/processed/feature_tables` | Stage 6 per-FOV CSVs. |
| `metadata_dir` | Directory containing input metadata; writable for output | `data/metadata` | Metadata handoffs between stages. |
| `background_correction_reports_dir` | Writable directory | `reports/background_correction` | Stage 4b plots when saving is enabled. |
| `raw_acquisitions_dir` | Directory-path string | `data/raw` | Proposed stage 1–2 input. |
| `quality_control_reports_dir` | Directory-path string | `reports/quality_control` | Proposed stage 3 reports. |
| `background_function_reports_dir` | Directory-path string | `reports/background_functions` | Proposed stage 4a reports. |

</div>


Keep output directories separate from original images. Output names are
predictable and repeated runs can overwrite existing outputs; choose different
output locations when retaining multiple configurations' results.

### File selection

**YAML path:** `shared.file_selection`, referenced by each stage's
`metadata.file_selection` and stage 4b's `background_function_selection.file_selection`.

<div class="configuration-table" role="region" aria-label="File selection parameters 3" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `filename` | Filename string; `"default"` (case-insensitive), `""`, or `null` for automatic selection | `"default"` | Metadata: exact filename. Background selection: matching name fragment; see stage 4b. |
| `date_source` | `"modified_time"`, `"filename"` | `"modified_time"` | Choose using filesystem modification time or a date parsed from each filename. |
| `select` | **`"newest"`, `"oldest"`** | `"newest"` | Select the latest or earliest candidate according to `date_source`. `"latest"` is not accepted. |
| `filename_date_separator` | Nonempty string | `"_"` | Separator used to split filenames when `date_source` is `"filename"`. |
| `filename_date_position` | Integer indexing an existing filename component; `0` is first, `-1` last | `0` | Position of the date after splitting. |
| `filename_date_format` | Python date-format string matching the date component | `"%Y%m%d"` | Eight-digit date, e.g. `20260918`; `%y%m%d` instead matches `260918`. |

</div>


**YAML path:** each stage's `metadata.file_selection` (filters are stage-specific).

<div class="configuration-table" role="region" aria-label="File selection parameters 4" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `include` | String, list of strings, or `null` | 4b: `"part_4a.csv"`; 5: `"part_4b.csv"`; 6: `"part_5.csv"` | Keep names containing any supplied substring. This is not a glob or regular expression. |
| `exclude` | String, list of strings, or `null` | `null` | Remove names containing any supplied substring. Exclusion takes priority. |

</div>


Automatic metadata selection considers visible regular files matching these
filters. No candidates causes an error. An explicit metadata filename bypasses
the filters and date-selection settings; the file must exist. Date parsing must
succeed for candidates when using filename dates. Pin a filename to avoid
ambiguity, including when several candidates have the same date.

**Example 1 — pin an exact metadata file.** Replace only `filename` inside the
existing `background_correction.metadata.file_selection` block:

```yaml
filename: "20260918_ACID_metadata_part_4a.csv"
```

Do this independently for stages 5 and 6 with their respective input metadata.
Do not pin `shared.file_selection.filename` to one stage's CSV: that value is
also inherited by other stages and background image selection.

**Example 2 — select by filename date.** Replace these entries inside the same
existing block, preserving `include` and `exclude`:

```yaml
filename: "default"
date_source: "filename"
select: "newest"
filename_date_separator: "_"
filename_date_position: 0
filename_date_format: "%Y%m%d"
```

Between matching files `20260917_ACID_metadata_part_4a.csv` and
`20260918_ACID_metadata_part_4a.csv`, this chooses the second. Modification-time
selection can change after files are copied or edited; filename-date selection
avoids that particular dependency.

### Dataset and channel conventions

**YAML path:** `shared.dataset_split` (inherited by 4b and 5 as `dataset_split`).

<div class="configuration-table" role="region" aria-label="Dataset and channel conventions parameters 5" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `column` | Existing metadata column name | `"is_train"` | Column containing dataset membership. |
| `labels.train` | Value matching training rows in that column | `1` | Label selected by the current 4b and 5 notebook calls. |
| `labels.validation` | Value matching validation rows | `0` | Planned validation label; legacy stage 2 called this group test. |
| `labels.test` | Value matching test rows | `3` | Planned separate test label; legacy stage 2 does not create this three-way split. |

</div>


**YAML path:** `shared.processing`.

<div class="configuration-table" role="region" aria-label="Dataset and channel conventions parameters 6" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `channel_axis` | Valid integer axis; use a nonnegative axis for the segmentation downsampling helper | `0` | Channel dimension of the input image; `0` means channels-first. |

</div>


Channel **axis** describes array layout; channel **positions** select stains
within that dimension. Keep these distinct. Individual processing sections
inherit this axis; stage 6 subsequently moves it to the last position.

## Background correction · Stage 4b

### Select the background

**YAML path:** `background_correction.background_function_selection`.

<div class="configuration-table" role="region" aria-label="Select the background parameters 7" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `directory` | Existing directory | `${shared.paths.background_functions_dir}` | Previously computed background images. |
| `background_function_strategy` | `1`, `2`, `3` | `1` | `1`: one image for the dataset; `2`: one per well; `3`: one per grid position. Match the strategy used to prepare the backgrounds. |
| `well_column_name` | Existing metadata column for strategy 2 | `"well"` | Well used to choose a background. |
| `gridpos_column_name` | Existing metadata column for strategy 3 | `"scene_name"` | Grid position used to choose a background. |

</div>


Its `file_selection` uses the shared options, with `include: ".ome.tif"` and
`exclude: null`. Unlike metadata loading, an explicit `filename` is a **substring
filter applied after include/exclude**. Strategy 1 requires exactly one match;
a complete filename is the clearest way to identify it. Strategies 2 and 3
require a matching set with one file per condition, whose names can be mapped to
the well or grid-position values. Automatic selection for these strategies uses
the multi-file date-selection helper: it selects all files sharing the exact
newest or oldest parsed date/timestamp. With `"modified_time"`, different save
times can select an incomplete set. Use a common filename date or an explicit
name fragment for a background set.

### Correction parameters

**YAML path:** `background_correction.processing`.

<div class="configuration-table" role="region" aria-label="Correction parameters parameters 8" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `channel_axis` | Valid integer axis; correction helper also supports `null` | `0` (shared) | Integer: correct each channel separately. `null`: whole-array correction, but other notebook consumers still expect a channel axis. |
| `method` | `"division"`, `"subtraction"` | `"division"` | Divide by or subtract the prepared background. |
| `offset` | Number, subject to the image/background checks below | `400` | Constant subtracted from the image; not a universally suitable camera offset. |
| `offset_background` | `true`, `false` | `true` | Also subtract the offset from the background before rescaling. |
| `rescale_background` | `null`, `"max"`, `"minmax"` | `"max"` | No rescaling, maximum normalization, or minimum–maximum normalization. Constant backgrounds cannot use `"minmax"`. |
| `epsilon` | Number greater than `0` (enforced) | `1.0e-8` | Stabilizes division and rescaling denominators. |

</div>


Image and background shapes must match and contain finite values. With channelwise
correction, these operations and checks apply per channel. For division with a
positive offset, the offset must be **strictly below the minimum image intensity**.
When also offsetting the background, it must not exceed the minimum background
intensity. A failure here requires reviewing the offset and input images.

After optional background offsetting and rescaling, division computes
`(image - offset) / (prepared_background + epsilon)`; subtraction computes
`image - prepared_background - offset`. Clipping follows correction, then the
result is cast to the output dtype.

### Numeric output and diagnostics

**YAML path:** `background_correction.processing`.

<div class="configuration-table" role="region" aria-label="Numeric output and diagnostics parameters 9" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `working_dtype` | NumPy dtype string; floating-point recommended | `"float32"` | Intermediate calculations. `"float64"` offers more precision at higher memory cost. |
| `output_dtype` | NumPy dtype string or `null` | `"float32"` | Saved result type; `null` retains the working type. Integer conversion can lose intensity information. |
| `clip_corrected_image` | `true`, `false` | `false` | Enable the bounds below. |
| `min_clip_value` | Number or `null` | `0` | Lower clipping bound; `null` disables that bound. |
| `max_clip_value` | Number or `null` | `null` | Upper clipping bound; `null` disables that bound. |
| `zero_kwargs` | Mapping of `numpy.zeros_like` options, or `null`; no `dtype` key | `null` | Advanced allocation options for channelwise correction. |
| `verbose` | `true`, `false` | `false` | Extra correction diagnostics; independent of notebook logging configuration. |

</div>


Clipping bounds are ignored when clipping is disabled. When enabled, provide at
least one bound and keep the lower bound no greater than the upper bound.

**YAML path:** `background_correction.graphs_saving`.

<div class="configuration-table" role="region" aria-label="Numeric output and diagnostics parameters 10" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `save_graphs` | `true`, `false` | `true` | Save the randomly selected successful FOV's comparison plot when the display cells run. |
| `graph_saving_directory` | Writable directory | `${shared.paths.background_correction_reports_dir}` | Plot destination. |
| `graph_date_format` | Date-format string | `"%Y%m%d"` | Date prefix. |
| `graph_suffix` | Extension supported by Matplotlib | `".png"` | Plot file format. |
| `graph_savingword` | Filename component | `"illumination_correction"` | Identifies the plot in its filename. |

</div>


## Segmentation · Stage 5

### Channels and preprocessing

**YAML path:** `object_segmentation.processing`.

<div class="configuration-table" role="region" aria-label="Channels and preprocessing parameters 11" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `channel_axis` | Valid nonnegative channel axis for this pipeline | `0` (shared) | Input channel dimension. |
| `nucleus_position` | Integer indexing an existing channel; use zero-based positions | `0` | Nuclear stain. |
| `concanavalin_position` | Integer indexing an existing channel | `1` | Concanavalin stain. |
| `actin_position` | Integer indexing an existing channel | `2` | Actin stain. |
| `med_filter_nucleus` | Positive integer filter size, or per-dimension sizes | `3` | Median-filter neighborhood for the nuclear image; `1` leaves values unchanged. |
| `med_filter_concactin_merge` | Positive integer filter size, or per-dimension sizes | `3` | Median-filter neighborhood after averaging concanavalin and actin. |
| `downsampling_factor` | Positive integer yielding nonzero spatial dimensions | `2` | Divide spatial dimensions by this factor using local-mean resizing. `1` retains size. |

</div>


The notebook averages concanavalin and actin, filters that result and the nuclear
channel separately, then stacks nuclear first and merged channel second. It
segments this downsampled image once to produce an object mask. It does not run
separate cell and nucleus segmentation passes.

### Model and mask output

**YAML path:** `object_segmentation.processing`.

<div class="configuration-table" role="region" aria-label="Model and mask output parameters 12" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `model_backend` | `"cellpose"` (only registered backend) | `"cellpose"` | Segmentation implementation. |
| `flow_threshold` | Numeric threshold passed to Cellpose; not restricted to an enum by ACID | `0.4` | Flow-error threshold used to retain masks; higher positive values are less restrictive; `0` disables this flow-error check in the installed backend. |
| `cellprob_threshold` | Numeric threshold, including negative values | `0.0` | Pixel inclusion threshold; lowering it can produce more/larger masks. It is not a probability constrained to 0–1. |
| `diameter` | Number or `null`; positive number enables diameter-based rescaling in the installed backend | `75` | Expected diameter in pixels of the **downsampled image passed to Cellpose**. |
| `order` | Integer `0`–`5` accepted by resizing; **keep `0` for labels** | `0` | Nearest-neighbor resizing preserves label identities. |
| `preserve_range` | `true`, `false`; **keep `true` for labels** | `true` | Preserve numeric label values while resizing. |
| `anti_aliasing` | `true`, `false`; **keep `false` for labels** | `false` | Avoid smoothing categorical label values. |
| `output_dtype` | NumPy dtype string or `null`; integer type recommended for labels | `"uint16"` | Cast saved masks; `null` retains their type. Ensure the type can hold the largest label (`uint16`: 65535). |

</div>


The notebook forwards model settings to the installed Cellpose version; ACID does
not add numeric range validation. GPU selection is automatic in the model wrapper,
not configured by this YAML. Changing downsampling changes the scale seen by the
model, so review `diameter` together with `downsampling_factor`. The output mask is
resized back to the original spatial dimensions.

## Feature extraction · Stage 6

**YAML path:** `feature_extraction.processing`.

<div class="configuration-table" role="region" aria-label="Feature extraction · Stage 6 parameters 13" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `channel_axis` | Valid integer axis in the input image | `0` (shared) | Moved to the last dimension before measurement. |
| `sigma` | Nonnegative number or per-filtered-axis sequence (recommended domain) | `3` | Gaussian standard deviation in samples along the selected axes; `0` skips smoothing. |
| `axes` | Valid integer axis, sequence of axes, or `null` | `-1` | Axes **after** moving channels last; `null` filters all axes. |

</div>


!!! important "The current setting smooths across channels"
    With `axes: -1`, the Gaussian filter acts on the channel dimension, mixing
    intensity information between channels. For an ordinary 2D channels-last
    image `(Y, X, C)`, spatial-only smoothing would use `axes: [0, 1]`. Choose axes
    according to your actual image dimensions and scientific intent. This guide
    documents the current setting without changing it.

Edge-touching labels are removed before measurement. Feature selection remains
in code: standard/custom region properties, Hessian, and structure-tensor
measurements are joined by object label. There is no YAML list of enabled features.

**YAML path:** `feature_extraction.features_saving`.

<div class="configuration-table" role="region" aria-label="Feature extraction · Stage 6 parameters 14" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `ome_suffix` | String matching the input filename suffix | `".ome.tif"` | Removed before naming the per-FOV feature CSV. |
| `output_suffix` | Filename suffix; keep `".csv"` for CSV content | `".csv"` | Changing the extension does not change the serializer. |
| `save_csv_index` | `true`, `false` | `false` | Include the pandas row index as a separate CSV column. |

</div>


## Advanced naming and metadata

These settings control file names and recorded information. They do not tune the
analysis itself. Keep them unchanged unless integrating with an existing metadata
schema. Changing a producer's output column name requires changing the subsequent
stage's input-column setting as well; those names are not all linked by interpolation.

### Shared file conventions

**YAML path:** `project_identity`.

<div class="configuration-table" role="region" aria-label="Shared file conventions parameters 15" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `project_name` | Filename component | `"ACID"` | Project identifier included in metadata and plot filenames. |

</div>


**YAML path:** `shared.naming`.

<div class="configuration-table" role="region" aria-label="Shared file conventions parameters 16" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `separator` | String | `"_"` | Separator between generated filename components. |
| `metadata_savingword` | Filename component | `"metadata"` | Identifies metadata tables. |
| `metadata_date_format` | Date-format string | `"%Y%m%d"` | Date prefix for metadata files. |
| `processing_date_format` | Date-format string | `"%y%m%d"` | Processing dates recorded inside image/CSV metadata. |
| `ome_suffix` | Filename suffix; keep the TIFF suffix | `".ome.tif"` | Suffix removed or appended when naming images. Does not change the TIFF writer. |
| `save_csv_index` | `true`, `false` | `false` | Whether to include a separate pandas index column. |

</div>


**YAML path:** `shared.image_saving`.

<div class="configuration-table" role="region" aria-label="Shared file conventions parameters 17" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `photometric` | TIFF photometric value; `"minisblack"` is the established grayscale setting | `"minisblack"` | Photometric interpretation passed to the TIFF writer. Other image types require matching data and writer support. |
| `save_imagej_compatible` | `true`, `false` | `true` | Passed as the TIFF writer’s `imagej` option; constrains supported array types and metadata. |

</div>


Stages 4b and 5 inherit `photometric`, `save_imagej_compatible`,
`save_file_name_separator`, and `ome_suffix` in their `image_saving` blocks.
All three stages inherit shared naming settings in `metadata.saving`.
These are references, not an additional automatic merge of shared settings.

### Stage file naming

**YAML path:** `background_correction.image_saving`.

<div class="configuration-table" role="region" aria-label="Stage file naming parameters 18" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `directory` | Writable directory | `"data/processed/corrected_fov"` | Destination of corrected images; retain the shared reference so subsequent stages find them. |
| `fov_illumin_corrected_savingword` | Filename component | `"bg"` | Appended to the original image stem; e.g. `field.ome.tif` becomes `field_bg.ome.tif`. |

</div>


**YAML path:** `object_segmentation.image_saving`.

<div class="configuration-table" role="region" aria-label="Stage file naming parameters 19" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `segmentation_savingword` | Filename component | `"segmentation"` | Appended to the corrected image stem; e.g. `field_bg_segmentation.ome.tif`. The directory is `shared.paths.segmentation_masks_dir`. |

</div>


**YAML path:** each stage’s `metadata`.

<div class="configuration-table" role="region" aria-label="Stage file naming parameters 20" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `directory` | Existing input / writable output directory | `${shared.paths.metadata_dir}` | Read the preceding metadata table and write this stage’s updated table. |

</div>


**YAML path:** each stage’s `metadata.saving`.

<div class="configuration-table" role="region" aria-label="Stage file naming parameters 21" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `metadata_file_suffix` | Filename suffix; retain `.csv` | 4b: `"part_4b.csv"`; 5: `"part_5.csv"`; 6: `"part_6.csv"` | Last component of the output metadata filename. Update the next stage’s selection filter if changing this. |
| `metadata_savingword` | Filename component | `"metadata"` (shared) | Identifies the table. |
| `metadata_date_format` | Date-format string | `"%Y%m%d"` (shared) | Date prefix. |
| `save_file_name_separator` | String | `"_"` (shared) | Joins date, project name, saving word, and suffix. |
| `save_csv_index` | `true`, `false` | `false` (shared) | Include an extra pandas index column. |

</div>


### CSV column mappings

These values name CSV columns; they are not the processing values stored in those
columns. Input mappings must match the selected CSV. Output mappings should be
nonempty, distinct names. `null_value` is the marker recorded for failed results.

**YAML path:** `background_correction.metadata.dataframe_columns`.

<div class="configuration-table" role="region" aria-label="CSV column mappings parameters 22" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `fov_column_name` | Column-name string | `"ome_tif_file_name"` | Input image filename column: extracted images in 4b, corrected images in 6. |
| `well_column_name` | Column-name string | `"well"` | Input well column, also referenced by background selection strategy 2. |
| `gridpos_column_name` | Column-name string | `"scene_name"` | Input grid-position column, also referenced by background selection strategy 3. |
| `null_value` | Missing-value marker | `.nan` | Recorded when processing fails; keep `.nan` for missing numeric data. |
| `column_name_separator` | String | `"_"` | Retained convention; the current 4b metadata updater uses explicit names rather than assembling column names with this setting. |
| `illum_correct_df_date_clm_name` | Column-name string | `"illumination_correction_date"` | Output column recording the correction date. |
| `illum_correct_df_meta_date_format` | Date-format string | `"%y%m%d"` | Format of the processing date written into the CSV. |

</div>


**YAML path:** `background_correction.metadata.dataframe_columns`.

<div class="configuration-table" role="region" aria-label="CSV column mappings parameters 23" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `illum_correct_df_file_name_clm_name` | Column-name string | `"illumination_correction_file_name"` | Output column recording the corrected-image filename. |
| `illum_correct_df_method_clm_name` | Column-name string | `"illumination_correction_method"` | Output column recording the correction method. |
| `illum_correct_df_offset_clm_name` | Column-name string | `"illumination_correction_offset"` | Output column recording the image offset. |
| `illum_correct_df_rescale_clm_name` | Column-name string | `"illumination_correction_rescale_background"` | Output column recording the background rescaling method. |
| `illum_correct_df_clipping_clm_name` | Column-name string | `"illumination_correction_clip_output"` | Output column recording the clipping switch. |
| `illum_correct_df_clip_min_value_clm_name` | Column-name string | `"illumination_correction_min_clip_value"` | Output column recording the lower clipping bound. |
| `illum_correct_df_clip_max_value_clm_name` | Column-name string | `"illumination_correction_max_clip_value"` | Output column recording the upper clipping bound. |

</div>


**YAML path:** `background_correction.metadata.dataframe_columns`.

<div class="configuration-table" role="region" aria-label="CSV column mappings parameters 24" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `illum_correct_df_offset_background_clm_name` | Column-name string | `"illumination_correction_offset_background"` | Output column recording the background-offset switch. |

</div>


**YAML path:** `object_segmentation.metadata.dataframe_columns`.

<div class="configuration-table" role="region" aria-label="CSV column mappings parameters 25" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `illum_correct_df_file_name_clm_name` | Column-name string | `"illumination_correction_file_name"` | Input corrected-image filename column produced by stage 4b. |
| `null_value` | Missing-value marker | `.nan` | Recorded when processing fails; keep `.nan` for missing numeric data. |
| `metadata_df_date_clm_name` | Column-name string | `"segmentation_date"` | Output column recording the processing date. |
| `metadata_df_file_name_clm_name` | Column-name string | `"segmentation_file_name"` | Output column recording the output mask filename. |
| `metadata_df_method_clm_name` | Column-name string | `"segmentation_method"` | Output column recording the segmentation backend. |
| `metadata_df_method_version_clm_name` | Column-name string | `"segmentation_method_version"` | Output column recording the model version. |
| `metadata_df_diameter_clm_name` | Column-name string | `"cellpose_diameter"` | Output column recording the model diameter. |

</div>


**YAML path:** `object_segmentation.metadata.dataframe_columns`.

<div class="configuration-table" role="region" aria-label="CSV column mappings parameters 26" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `metadata_df_flow_threshold_clm_name` | Column-name string | `"cellpose_flow_threshold"` | Output column recording the flow threshold. |
| `metadata_df_cellprob_threshold_clm_name` | Column-name string | `"cellpose_cellprob_threshold"` | Output column recording the pixel inclusion threshold. |
| `metadata_df_downsampling_factor_clm_name` | Column-name string | `"downsampling_factor"` | Output column recording the downsampling factor. |
| `metadata_df_nucleus_med_filter_size_name` | Column-name string | `"nucleus_median_filter_size"` | Output column recording the nuclear median-filter size. |
| `metadata_df_concactin_merge_med_filter_size_name` | Column-name string | `"concactin_merge_median_filter_size"` | Output column recording the merged-channel median-filter size. |
| `metadata_df_resize_order_name` | Column-name string | `"resize_order"` | Output column recording the mask interpolation order. |
| `metadata_df_output_dtype_name` | Column-name string | `"output_dtype"` | Output column recording the actual saved mask dtype. |

</div>


**YAML path:** `object_segmentation.metadata.dataframe_columns`.

<div class="configuration-table" role="region" aria-label="CSV column mappings parameters 27" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `metadata_df_meta_date_format` | Date-format string | `"%y%m%d"` | Format of the processing date written into the CSV. |

</div>


**YAML path:** `feature_extraction.metadata.dataframe_columns`.

<div class="configuration-table" role="region" aria-label="CSV column mappings parameters 28" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `null_value` | Missing-value marker | `.nan` | Recorded when processing fails; keep `.nan` for missing numeric data. |
| `fov_column_name` | Column-name string | `"illumination_correction_file_name"` | Input image filename column: extracted images in 4b, corrected images in 6. |
| `segmentation_column_name` | Column-name string | `"segmentation_file_name"` | Input mask filename column produced by stage 5. |
| `metadata_df_meta_date_format` | Date-format string | `"%y%m%d"` | Format of the processing date written into the CSV. |
| `metadata_df_date_clm_name` | Column-name string | `"features_extraction_date"` | Output column recording the processing date. |
| `metadata_df_file_name_clm_name` | Column-name string | `"features_data_frame"` | Output column recording the per-FOV feature CSV filename. |
| `metadata_df_method_clm_name` | Column-name string | `"features_extraction_preprocessing"` | Output column recording the preprocessing description. |

</div>


**YAML path:** `feature_extraction.metadata.dataframe_columns`.

<div class="configuration-table" role="region" aria-label="CSV column mappings parameters 29" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `preprocessing_steps` | Descriptive string with optional interpolation | `"move channel axis to last position; gaussian smoothing with sigma 3 along last axis; exclude labels touching image edge"` | Recorded description, not executable instructions. Currently describes channel-last smoothing using `${feature_extraction.processing.sigma}` and edge-label exclusion. |

</div>


### Embedded image metadata

These names identify entries inside saved TIFF metadata, separate from CSV column
names. They record how an image was processed. Changing an entry name does not
change the processing algorithm.

**YAML path:** `object_segmentation.metadata`.

<div class="configuration-table" role="region" aria-label="Embedded image metadata parameters 30" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `segmentation_method_name` | Descriptive string | `"cellpose"` | Recorded backend name; references `object_segmentation.processing.model_backend`. |

</div>


**YAML path:** `background_correction.metadata.image_metadata`.

<div class="configuration-table" role="region" aria-label="Embedded image metadata parameters 31" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `proc_img_meta_date_name` | Metadata-key string | `"processing_date_yymmdd"` | Entry recording processing date yymmdd. |
| `processing_date_format` | Date-format string | `"%y%m%d"` | Format for the embedded processing date. |
| `proc_img_meta_dtype_name` | Metadata-key string | `"dtype"` | Entry recording dtype. |
| `illum_corr_method_metadata_entry` | Metadata-key string | `"illumination_correction_method"` | Entry recording illumination correction method. |
| `illum_corr_offset_metadata_entry` | Metadata-key string | `"illumination_correction_offset"` | Entry recording illumination correction offset. |
| `illum_corr_rescale_metadata_entry` | Metadata-key string | `"illumination_correction_rescale_background"` | Entry recording illumination correction rescale background. |

</div>


**YAML path:** `background_correction.metadata.image_metadata`.

<div class="configuration-table" role="region" aria-label="Embedded image metadata parameters 32" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `illum_corr_clipping_metadata_entry` | Metadata-key string | `"illumination_correction_clipping"` | Entry recording illumination correction clipping. |
| `illum_corr_clip_min_value_metadata_entry` | Metadata-key string | `"illumination_correction_min_clip_value"` | Entry recording illumination correction min clip value. |
| `illum_corr_clip_max_value_metadata_entry` | Metadata-key string | `"illumination_correction_max_clip_value"` | Entry recording illumination correction max clip value. |
| `illum_corr_offset_background_metadata_entry` | Metadata-key string | `"illumination_correction_offset_background"` | Entry recording illumination correction offset background. |

</div>


**YAML path:** `object_segmentation.metadata.image_metadata`.

<div class="configuration-table" role="region" aria-label="Embedded image metadata parameters 33" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `processing_date_format` | Date-format string | `"%y%m%d"` | Format for the embedded processing date. |
| `preproc_img_meta_raw_file_name_entry` | Metadata-key string | `"raw_file_name"` | Substring used to copy matching source-image metadata entries for raw file name. |
| `preproc_img_meta_scene_file_name_entry` | Metadata-key string | `"scene_file_name"` | Substring used to copy matching source-image metadata entries for scene file name. |
| `preproc_img_meta_x_physic_px_size_entry` | Metadata-key string | `"physical_size_x"` | Substring used to copy matching source-image metadata entries for physical size x. |
| `preproc_img_meta_y_physic_px_size_entry` | Metadata-key string | `"physical_size_y"` | Substring used to copy matching source-image metadata entries for physical size y. |
| `preproc_img_meta_x_physic_px_size_unit_entry` | Metadata-key string | `"physical_size_unit_x"` | Substring used to copy matching source-image metadata entries for physical size unit x. |

</div>


**YAML path:** `object_segmentation.metadata.image_metadata`.

<div class="configuration-table" role="region" aria-label="Embedded image metadata parameters 34" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `preproc_img_meta_y_physic_px_size_unit_entry` | Metadata-key string | `"physical_size_unit_y"` | Substring used to copy matching source-image metadata entries for physical size unit y. |
| `segmented_img_meta_date_name` | Metadata-key string | `"processing_date_yymmdd"` | Entry recording processing date yymmdd. |
| `segmented_img_meta_method_name` | Metadata-key string | `"method"` | Entry recording method. |
| `segmentation_method_version_name` | Metadata-key string | `"version"` | Entry recording version. |
| `segmented_img_meta_diameter_name` | Metadata-key string | `"cellpose_diameter"` | Entry recording cellpose diameter. |
| `segmented_img_meta_flow_threshold_name` | Metadata-key string | `"cellpose_flow_threshold"` | Entry recording cellpose flow threshold. |

</div>


**YAML path:** `object_segmentation.metadata.image_metadata`.

<div class="configuration-table" role="region" aria-label="Embedded image metadata parameters 35" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `segmented_img_meta_cellprob_threshold_name` | Metadata-key string | `"cellpose_cellprob_threshold"` | Entry recording cellpose cellprob threshold. |
| `segmented_img_meta_downsampling_factor_name` | Metadata-key string | `"downsampling_factor"` | Entry recording downsampling factor. |
| `segmented_img_meta_nucleus_med_filter_size_name` | Metadata-key string | `"nucleus_median_filter_size"` | Entry recording nucleus median filter size. |
| `segmented_img_meta_concactin_merge_med_filter_size_name` | Metadata-key string | `"concanavalin_actin_merge_median_filter_size"` | Entry recording concanavalin actin merge median filter size. |
| `segmented_img_meta_resize_order_name` | Metadata-key string | `"upsampling_resize_order"` | Entry recording upsampling resize order. |
| `segmented_img_meta_processing_name` | Metadata-key string | `"processing_steps"` | Entry recording processing steps. |

</div>


**YAML path:** `object_segmentation.metadata.image_metadata`.

<div class="configuration-table" role="region" aria-label="Embedded image metadata parameters 36" tabindex="0" markdown="1">

| Parameter | Accepted values / constraints | Current value | Meaning |
| --- | --- | --- | --- |
| `segmented_img_meta_dtype_name` | Metadata-key string | `"dtype"` | Entry recording dtype. |

</div>


Under `object_segmentation.metadata.image_metadata`,
`segmented_img_meta_processing_steps` accepts a descriptive string. Its current
text describes median filtering, averaging concanavalin and actin, downsampling,
stacking, segmentation, and upsampling. It is recorded under the key specified by
`segmented_img_meta_processing_name`; it does not execute those operations. The
notebook appends a dtype-conversion note when `output_dtype` is set.

## Earlier stages and implementation notes

`field_of_view_extraction`, `dataset_splitting`, `quality_control`, and
`background_function_calculation` describe proposed settings for stages 1–4a.
Their legacy notebooks do not load this YAML. Editing those sections does not
reconfigure the legacy pipeline. In particular, the planned three-way split is
not implemented by the legacy two-way splitting notebook.

This reference describes the refactored notebooks and their current helpers.
The legacy notebooks and `hyperparameters.md` provide historical context; notes
proposing different formats, removed settings, or new responsibilities are not
implemented configuration features. Functions may still live inside notebooks.
For callable APIs, see [Image processing](image-processing.md),
[Feature extraction](feature-extraction.md), and [Utilities](utilities.md).
