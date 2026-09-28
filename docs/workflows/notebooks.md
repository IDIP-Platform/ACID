# Notebooks

The README contains historical notebook names, but the current repository uses a
more granular notebook workflow.

## Current Notebook Map

| Notebook | Workflow stage |
| --- | --- |
| `part1_extract_field_of_view.ipynb` | Extract fields of view from raw ND2 files and prepare metadata. |
| `part2_split_train_test.ipynb` | Split data into train/test sets. |
| `part3_control_image_quality.ipynb` | Run image quality-control checks. |
| `part3extra_check_ch_treatment.ipynb` | Extra channel/treatment checks. |
| `part3extra_test_image_qc.ipynb` | Extra image quality-control testing. |
| `part4a_compute_background_function.ipynb` | Estimate background-correction functions. |
| `part4b_correct_background.ipynb` | Apply background correction. |
| `part4extra_test_correct_background.ipynb` | Extra background-correction testing. |
| `part5_segment_object.ipynb` | Segment nuclei/cells or other objects. |
| `part6_extract_feature.ipynb` | Extract features and measurements. |

## Historical README Names

The README mentions:

- `part1_raw_to_segmentation.ipynb`
- `part2_segmentation_to_measurement.ipynb`

These notebooks are not present in the current repository. Treat those names as
historical references to earlier workflow organization.

## Running Notebooks

Install the package and create the `ACID` Jupyter kernel as described in
[Installation](../installation.md). In VS Code or JupyterLab, select that kernel
before running notebook cells.

!!! note "Data access"
    The notebooks require local microscopy data paths. When sharing validation
    steps in issues or pull requests, describe the dataset used without exposing
    private or sensitive data.
