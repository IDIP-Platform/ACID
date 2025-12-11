import numpy as np
import pytest
from your_module import compute_plls


# ------------------------------------------------------------
# Helper: create synthetic images with known properties
# ------------------------------------------------------------

def make_gaussian_image(size=128, sigma=3.0):
    """Generate a simple 2D Gaussian spot image."""
    y, x = np.indices((size, size))
    cy = cx = size // 2
    img = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return img


def make_blurry_image(size=128, blur_factor=6.0):
    """Generate a more strongly blurred Gaussian (lower frequencies)."""
    return make_gaussian_image(size=size, sigma=blur_factor)


# ------------------------------------------------------------
# Core correctness tests
# ------------------------------------------------------------

def test_plls_runs_on_random_image():
    """PLLS should compute without error on a random 2D array."""
    img = np.random.rand(128, 128)
    slope, intercept, freqs, power = compute_plls(img)
    assert np.isfinite(slope)
    assert np.isfinite(intercept)
    assert freqs.ndim == 1
    assert power.ndim == 1


def test_plls_blurry_vs_sharp_slope():
    """Blurrier images should yield more negative slopes."""
    sharp = make_gaussian_image(size=128, sigma=2.0)
    blurry = make_gaussian_image(size=128, sigma=8.0)

    slope_sharp, *_ = compute_plls(sharp)
    slope_blurry, *_ = compute_plls(blurry)

    # Expect blurry to have a steeper falloff → more negative slope
    assert slope_blurry < slope_sharp


# ------------------------------------------------------------
# Axis slicing behavior
# ------------------------------------------------------------

def test_plls_axis_on_stack():
    """Computing PLLS along an axis should return one value per slice."""
    stack = np.random.rand(5, 64, 64)  # 5 slices
    slopes, intercepts, freqs, powers = compute_plls(stack, axis=0)

    assert slopes.shape == (5,)
    assert intercepts.shape == (5,)
    assert isinstance(powers, list)
    assert len(powers) == 5


def test_plls_axis_on_channels():
    """Test PLLS on multi-channel data."""
    img = np.random.rand(3, 128, 128)  # pretend 3-channel image
    slopes, intercepts, freqs, power_list = compute_plls(img, axis=0)

    assert slopes.size == 3
    for p in power_list:
        # Each entry may be None if slice was invalid, but slope should *exist*.
        assert isinstance(p, (np.ndarray, type(None)))


# ------------------------------------------------------------
# Error handling tests
# ------------------------------------------------------------

def test_plls_rejects_1d_input():
    """A 1D array should not be accepted when axis=None."""
    arr = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        compute_plls(arr)


def test_plls_rejects_3d_without_axis():
    """3D input must specify an axis; otherwise error."""
    arr = np.random.rand(4, 4, 4)
    with pytest.raises(ValueError):
        compute_plls(arr)  # Should expect 2D only


# ------------------------------------------------------------
# Edge cases & numerical stability
# ------------------------------------------------------------

def test_plls_constant_image_returns_nan_or_extreme():
    """Constant images have no frequency structure. Slope should be NaN or extreme."""
    img = np.ones((128, 128))
    slope, *_ = compute_plls(img)
    # Either NaN or a very large negative slope
    assert (np.isnan(slope)) or (slope < -0.1)


def test_plls_handles_zeros_without_crashing():
    """Images with large zero regions should not crash."""
    img = np.zeros((128, 128))
    slope, *_ = compute_plls(img)
    assert True  # Test passes if no exception is raised


def test_plls_small_image():
    """Tiny images should work or fail gracefully."""
    img = np.random.rand(8, 8)
    slope, intercept, freqs, power = compute_plls(img)
    # freqs may be very short but slope should be finite
    assert np.isfinite(slope)


def test_plls_with_mask_zero_off():
    """Check behavior when mask_zero=False."""
    img = make_gaussian_image(size=64, sigma=3.0)
    slope, intercept, freqs, power = compute_plls(img, mask_zero=False)
    assert np.isfinite(slope)
    assert np.isfinite(intercept)