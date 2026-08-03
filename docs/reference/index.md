# Reference

The reference documentation is generated from Python docstrings in `src/acid`
with `mkdocstrings`.

Use this section when you need details about functions, classes, and modules.
For workflow-level explanations, start with the user guide or notebooks.

## Package Areas

| Area | Description |
| --- | --- |
| [Data preparation](data-preparation.md) | Metadata formatting, category mapping, and train/test splitting. |
| [Image processing](image-processing.md) | Metadata extraction, filtering, resizing, background correction, and image output. |
| [Image quality control](image-quality-control.md) | QC metrics, flags, and display helpers. |
| [Image measurement](image-measurement.md) | Global and local image statistics. |
| [Feature extraction](feature-extraction.md) | Haralick, Hessian, structure tensor, intensity, and region-based features. |
| [Image visualization](image-visualization.md) | Visualization helpers for image correction workflows. |
| [Utilities](utilities.md) | File handling, image I/O, labels, strings, and general helpers. |

## Not Yet Included

Some experimental, script-like, or import-problematic modules are intentionally
excluded from this first reference pass:

- `acid.image_quality_control.measure_laplacian_var`
- `acid.image_processing.segmentation_preprocessing`
- `acid.main_df`
- `acid.main_fm`
- `acid.main_fm_develop`
- `acid.main_mbfm_develop`
- source test modules under `acid.image_quality_control`
