# Input Data

ACID expects raw microscopy data from a Nikon Ti2 microscope saved as ND2 files.
ND2 is a proprietary Nikon format, so the pipeline uses package dependencies
such as `bioio` and `bioio-nd2` to read the image data.

## Expected Layout

The README describes the input data layout as:

```text
input_data_directory/
├── experiment_directory/
│   ├── file1.nd2
│   ├── file2.nd2
│   ├── file3.nd2
│   └── fileN.nd2
└── plate_layout.csv

output_data_directory/
```

The input directory may contain one or more experiment directories. The README
states that the latest saved `.csv` file in the input directory is used by
default as the plate layout file.

## Experiment Directories

Each experiment directory is expected to contain ND2 files. The README notes
that experiment directories may also contain other files or subdirectories, as
long as their names do not contain the `.nd2` string.

The described project experiments are:

- `Experiment A07.2`
- `Experiment A07.3`
- `Experiment A07.4`

## Image Assumptions

The workflow was designed around ND2 raw files containing one or more fields of
view. The README describes each field of view as:

- single plane
- 1024 x 1024 pixels
- 0.325 x 0.325 micron pixel size
- five channels

The expected channel order is:

| Channel | Signal |
| --- | --- |
| channel 1 | Hoechst, nucleus |
| channel 2 | Concanavalin A 488, endoplasmic reticulum |
| channel 3 | Phalloidin 568, actin |
| channel 4 | Anti-NS3 647, infection marker |
| channel 5 | Transmitted light |

!!! warning "Unevaluated edge cases"
    The README states that cases outside the original assumptions had not been
    evaluated as of 2026-02-02. This includes empty segmentation masks, fields
    of view without segmentable cells, images with severe artifacts, unexpected
    channel counts, and other non-standard inputs.
