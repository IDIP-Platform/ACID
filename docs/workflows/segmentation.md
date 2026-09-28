# Segmentation

The segmentation stage creates nucleus and cell masks for extracted fields of
view. The README describes preprocessing, CellPoseSAM-based segmentation,
upsampling, mask saving, metadata updates, and hyperparameter logging.

## Image Preprocessing

The README describes the following preprocessing steps:

- select the nuclear channel as `nucleus_arr`
- select and stack the nuclear and f-actin channels as `cell_arr`
- stack channels in the order nuclear channel, then f-actin channel
- apply median filtering
- downsample images by a factor of 2 using local-mean downsampling

For `cell_arr`, the channel-stacking axis is not downsampled.

## Segmentation

The README describes CellPoseSAM-based segmentation for:

- cells, using the stacked `cell_arr`
- nuclei, using `nucleus_arr`

Segmentation masks are upsampled back to the original image size using nearest
neighbor interpolation.

## Saved Masks and Metadata

Segmentation masks are saved as OME-TIFF files under the output `seg`
directory. The metadata table is then updated with segmentation-mask
information.

The README also describes saving run hyperparameters in `secondary_output`.

!!! note "Version references"
    The README mentions CellPoseSAM versions from active development notes. Use
    `pyproject.toml` and `uv.lock` as the source of truth for dependency
    versions in the current repository.

## Current Notebook

Use:

```text
notebooks/part5_segment_object.ipynb
```

The README also describes earlier workflow tasks that now live in separate
notebooks, including image quality control and background correction.
