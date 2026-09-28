# Workflow Overview

ACID organizes microscopy analysis as a sequence of notebook-driven processing
steps supported by reusable package modules in `src/acid`.

The README describes three broad workflow stages:

1. Extract selected images and metadata.
2. Segment nuclei and cells.
3. Extract intensity measurements.

The current repository splits the workflow across more focused notebooks:

1. Extract fields of view.
2. Split train/test data.
3. Control image quality.
4. Compute and apply background correction.
5. Segment objects.
6. Extract features.

## Pipeline Flow

```text
raw ND2 files
    ↓
field-of-view extraction and metadata preparation
    ↓
train/test split
    ↓
image quality control
    ↓
background estimation and correction
    ↓
nucleus/cell segmentation
    ↓
label filtering and cytosol mask generation
    ↓
feature extraction and global measurements
```

## Working Metadata Table

After field-of-view extraction, the README describes a metadata table named
`metadata_df`. Later workflow stages update this table with segmentation,
cytosol mask, and processing information.

The metadata table is the main link between raw files, extracted fields of view,
segmentation masks, and final measurements.

!!! warning "Workflow description under development"
    The README states that some analysis-step descriptions were refactored from
    a different project. The workflow pages keep those details visible but mark
    inherited or stale names where needed.
