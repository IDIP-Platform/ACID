# Output Data

ACID writes extracted fields of view, segmentation masks, metadata tables,
background-related outputs, and global measurements.

!!! warning "Structure under development"
    The README states that the output structure is still under development and
    that some descriptions were refactored from a different project. Treat this
    page as a guide to the current intended structure, not a finalized public
    data contract.

## Expected Layout

```text
output_data_directory/
├── [date]_ACID_glob_measurements.csv
├── fov/
│   ├── file1_field_of_view1.ome.tif
│   ├── file1_field_of_view2.ome.tif
│   ├── file1_string.xml
│   └── fileN_field_of_viewM.ome.tif
├── seg/
│   ├── file1_field_of_view1.ome.tif
│   ├── file1_field_of_view2.ome.tif
│   └── fileN_field_of_viewM.ome.tif
├── metadata/
│   └── [date]_ACID_metadata_part[notebook_progressive_number].csv
└── background/
    └── [...]
```

The README also describes a `secondary_output` directory created inside the
working directory. It stores run hyperparameters and other secondary outputs.

## Snakemake Pipeline Outputs

The [Snakemake pipeline](../workflows/snakemake.md) writes everything under
`shared.output_root` (`data/processed/<experiment id>` by default), in the
folders named by `shared.paths` in the configuration:

```text
data/processed/<experiment id>/
├── config.resolved.yaml          # resolved configuration of the run
├── extracted_fov/                # part0: one OME-TIFF per field of view
├── background_functions/         # part3
├── corrected_fov/                # part4
├── segmentation_masks/           # part5
├── feature_tables/               # part6
├── metadata/
│   └── ACID_metadata_part_<N>.csv   # no date prefix (part_1 ... part_6)
├── reports/
│   ├── quality_control/          # QC graphs
│   └── <stage>_failures.csv      # FOVs that a stage could not process
├── work/<stage>/batch_<i>.json   # result records of each batch job
└── logs/<rule>/                  # one log file per job
```

The metadata files have no date prefix so that Snakemake output names stay
the same from day to day; they still match the notebooks' `file_selection`
patterns. `work/` and `logs/` are intermediate files of the workflow.

## `fov`

The `fov` directory contains individual fields of view extracted from ND2 raw
files and saved as OME-TIFF files.

Fields from multiple raw files are pooled into the same `fov` directory. Global
raw metadata can be saved separately as XML rather than copied into every
field-of-view image.

## `seg`

The `seg` directory contains segmentation masks for individual fields of view.
The README describes this location as the destination for cell, nucleus, and
later cytosol segmentation masks.

## `metadata`

The `metadata` directory contains CSV files describing raw files, extracted
fields of view, processing dates, segmentation files, and other processing
metadata.

The README gives the metadata filename pattern:

```text
[date]_ACID_metadata_part[notebook_progressive_number].csv
```

## `background`

The `background` directory is reserved for background-related outputs. The
current README does not fully specify the files in this directory.

## Global Measurements

The main measurement output is described as:

```text
[date]_ACID_glob_measurements.csv
```

Rows correspond to individual cells. Columns include raw-file metadata,
field-of-view metadata, segmentation identifiers, geometry measurements,
intensity measurements, and background offsets.
