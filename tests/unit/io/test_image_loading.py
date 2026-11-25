import numpy as np
import pytest

from acid.io.image_loading import (
    load_field_of_view,
    load_segmentation_mask,
    load_tiff,
)


def test_load_tiff_reads_full_path(write_fov):
    directory, image = write_fov()

    np.testing.assert_array_equal(load_tiff(directory / "a.ome.tif"), image)


def test_load_field_of_view_reads_image_from_directory(write_fov):
    directory, image = write_fov()

    result = load_field_of_view("a.ome.tif", directory)

    np.testing.assert_array_equal(result, image)


def test_load_field_of_view_missing_file_raises_oserror_with_path(tmp_path):
    with pytest.raises(OSError, match="Could not load field of view TIFF: .*missing"):
        load_field_of_view("missing.ome.tif", tmp_path)


def test_load_segmentation_mask_reads_mask_from_directory(write_fov):
    directory, image = write_fov(filename="mask.ome.tif")

    result = load_segmentation_mask("mask.ome.tif", directory)

    np.testing.assert_array_equal(result, image)


def test_load_segmentation_mask_missing_file_names_mask(tmp_path):
    with pytest.raises(OSError, match="Could not load segmentation mask TIFF"):
        load_segmentation_mask("missing.ome.tif", tmp_path)
