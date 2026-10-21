import numpy as np
from scipy.ndimage import gaussian_filter

from acid.image_quality_control.measure_plls import compute_plls


def random_image(shape=(2, 512, 512)):
    return np.random.default_rng(0).random(shape) * 1000


def test_compute_plls_is_reproducible():
    image = random_image()

    runs = [np.asarray(compute_plls(image=image, axis=0)) for _ in range(5)]

    for run in runs[1:]:
        np.testing.assert_array_equal(run, runs[0])


def test_compute_plls_blurred_image_has_steeper_slope():
    image = random_image()
    blurred = gaussian_filter(image, sigma=(0, 4, 4))

    sharp_slope = np.asarray(compute_plls(image=image, axis=0))
    blurred_slope = np.asarray(compute_plls(image=blurred, axis=0))

    assert np.all(blurred_slope < sharp_slope)
