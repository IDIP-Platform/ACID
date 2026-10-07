import numpy as np

from acid.io.image_loading import load_tiff


def test_load_tiff_reads_full_path(write_fov):
    directory, image = write_fov()

    np.testing.assert_array_equal(load_tiff(directory / "a.ome.tif"), image)
