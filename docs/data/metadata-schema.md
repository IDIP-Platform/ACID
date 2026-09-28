# Metadata and Measurement Schema

This page summarizes columns described in the README for metadata and global
measurement tables.

!!! warning "Schema still being refined"
    Some field descriptions in the README are inherited from a previous project
    or explicitly marked as needing verification. Use this page as a working
    reference and confirm column semantics against the current pipeline code
    before relying on them as a formal schema.

## File and Scene Metadata

| Column | Meaning |
| --- | --- |
| `raw_file_name` | Name of the input ND2 raw file. |
| `scene_name` | Name of the field of view, also called scene, inside the raw file. |
| `acquisition_date_yyyymmdd` | Acquisition date at the microscope. |
| `processing_date_yymmdd` | Date when the field of view was extracted and saved as OME-TIFF. |
| `ome_tif_file_name` | OME-TIFF filename used for the extracted field of view. |
| `location` | Acquisition location. |
| `microscope` | Microscope used for acquisition. |
| `objective` | Objective used for acquisition. |

## Experimental Metadata

The README includes several columns that appear inherited from a previous
project context:

| Column | Meaning in README |
| --- | --- |
| `donor` | Donor patient code. |
| `transfection` | Transfected gene or control. |
| `stiffness` | Material onto which cells were seeded. |
| `stimulation` | Stimulation condition. |
| `time_of_stimulation` | Stimulation duration. |

!!! note "Project-specific review needed"
    These columns should be reviewed against the current Dengue/Cell Painting
    ACID metadata. They may need to be renamed or replaced by infection,
    treatment, well, experiment, and timepoint fields.

## Channel Metadata

| Column | Meaning |
| --- | --- |
| `channel_0` | Reporter acquired at channel-axis position 0. |
| `channel_1` | Reporter acquired at channel-axis position 1. |
| `channel_2` | Reporter acquired at channel-axis position 2. |
| `channel_3` | Reporter acquired at channel-axis position 3. |
| `channel_4` | Reporter acquired at channel-axis position 4. |

The README uses Python indexing for these columns, so numbering starts at 0.

## Image Metadata

| Column | Meaning |
| --- | --- |
| `physical_size_unit_x` | Unit for `physical_size_x`. |
| `physical_size_unit_y` | Unit for `physical_size_y`. |
| `dtype` | Raw ND2 data type. |
| `size_t` | Number of timepoints. |
| `size_c` | Number of channels. |
| `size_z` | Number of z planes. |
| `size_y` | Number of pixels along y. |
| `physical_size_y` | Physical pixel size along y. |
| `size_x` | Number of pixels along x. |
| `physical_size_x` | Physical pixel size along x. |
| `dims_order` | OME-TIFF dimension order, such as `TCZYX`. |

## Segmentation Metadata

| Column | Meaning |
| --- | --- |
| `segmentation_date_yymmdd` | Date when segmentation was run. |
| `segmentation_name` | Name of the saved segmentation file. |
| `label` | Cell label identifier for a measurement row. |

The README states that `label` identifies cells in the cytosol segmentation
mask, while the corresponding nucleus and cell masks may have different labels
before relabeling.

## Measurements

| Column pattern | Meaning |
| --- | --- |
| `area` | Cell area in pixels. |
| `centroid-N` | N coordinate of the measured-cell centroid. |
| `centroid-M` | M coordinate of the measured-cell centroid. |
| `intensity_mean-(channel)` | Mean intensity for a cell in a channel. |
| `intensity_max-(channel)` | Maximum intensity for a cell in a channel. |
| `intensity_min-(channel)` | Minimum intensity for a cell in a channel. |
| `background_offset-(channel)` | Median background-pixel intensity for a field of view and channel. |

!!! warning "Centroid coordinates need verification"
    The README marks centroid semantics as needing review. It suggests that, for
    2D masks, `centroid-N` corresponds to y and `centroid-M` corresponds to x,
    measured from the top-left image corner.

## Background Offset

The README defines background pixels as pixels that are not present in either
the nucleus segmentation mask or the cell segmentation mask. The median value of
those pixels is reported per field of view and channel.
