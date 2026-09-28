# Field-of-View Extraction

The first workflow stage extracts individual fields of view, also called
scenes, from raw ND2 files and prepares metadata for later processing.

## Raw Metadata

The README describes extraction of original XML metadata from the raw files.
Those metadata are saved separately because a raw ND2 file may contain metadata
for multiple positions or fields of view.

## Processing Metadata

The workflow extracts raw-file metadata and scene names, then combines them with
information provided by the user. The resulting table is referred to in the
README as `metadata_df`.

Later workflow stages use `metadata_df` to link fields of view to the relevant
raw file, acquisition metadata, segmentation file, and output measurements.

## OME-TIFF Export

Each selected field of view is saved as an OME-TIFF file in the output
`fov` directory.

The OME-TIFF file receives metadata relevant to that specific field of view.
The global raw XML metadata are not blindly copied into every field-of-view file
because that would attach metadata for other fields to the wrong image.

## Current Notebook

Use:

```text
notebooks/part1_extract_field_of_view.ipynb
```

The historical README phrase `part1_raw_to_segmentation.ipynb` should be treated
as stale because that notebook is not present in the current repository.
