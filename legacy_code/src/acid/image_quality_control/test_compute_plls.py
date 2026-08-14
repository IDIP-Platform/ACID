import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from image_quality_control.measure_plls import compute_plls


def run_all_tests():
    """Run an expanded suite of PLLS tests and print results."""

    def run_test(name, func):
        print(f"▶ {name} ... ", end="")
        try:
            func()
            print("OK")
        except AssertionError as e:
            print("FAIL")
            raise e
        except Exception as e:
            print("ERROR")
            print(e)
            raise e

    # ----------------------------------------
    # 1 — Gradient image
    # ----------------------------------------
    def test_gradient_image():
        img = np.tile(np.linspace(0, 1, 128), (128, 1))
        slope = compute_plls(img)
        assert np.isfinite(slope), "Slope should be finite for gradient"

    # ----------------------------------------
    # 2 — Noise image
    # ----------------------------------------
    def test_noise_image():
        rng = np.random.default_rng(123)
        img = rng.normal(0, 1, (128, 128))
        slope = compute_plls(img)
        assert np.isfinite(slope)
        assert -0.5 < slope < 0.5

    # ----------------------------------------
    # 3 — Constant image
    # ----------------------------------------
    def test_constant_image():
        img = np.ones((128, 128))
        slope = compute_plls(img)
        assert np.isnan(slope)

    # ----------------------------------------
    # 4 — Z-stack slicing
    # ----------------------------------------
    def test_axis_slicing():
        img = np.zeros((3, 64, 64))
        img[0] = np.random.random((64, 64))
        img[1] = np.linspace(0, 1, 64).reshape(64,1)
        img[2] = 5
        slopes = compute_plls(img, axis=0)
        assert len(slopes) == 3
        assert np.isfinite(slopes[0])
        assert np.isfinite(slopes[1])
        assert np.isnan(slopes[2])

    # ----------------------------------------
    # 5 — Verbose mode
    # ----------------------------------------
    def test_verbose_mode():
        img = np.random.random((64, 64))
        slope = compute_plls(img, verbose=2)
        assert np.isfinite(slope)

    # ----------------------------------------
    # 6 — Tiny images
    # ----------------------------------------
    def test_tiny_images():
        for size in [1, 2, 3]:
            img = np.random.random((size, size))
            slope = compute_plls(img)
            assert slope is None or np.isfinite(slope) or np.isnan(slope)

    # ----------------------------------------
    # 7 — Image with NaNs
    # ----------------------------------------
    def test_nan_image():
        img = np.ones((64, 64))
        img[10,10] = np.nan
        slope = compute_plls(img)
        assert np.isnan(slope) or np.isfinite(slope)

    # ----------------------------------------
    # 8 — Single-slice z-stack
    # ----------------------------------------
    def test_single_slice_zstack():
        img = np.random.random((1, 128, 128))
        slopes = compute_plls(img, axis=0)
        assert slopes.shape[0] == 1
        assert np.isfinite(slopes[0])

    # ----------------------------------------
    # 9 — Only DC component
    # ----------------------------------------
    def test_dc_only_image():
        img = np.full((128, 128), 7.0)
        slope = compute_plls(img)
        assert np.isnan(slope)

    # ----------------------------------------
    # 10 — Extreme aspect ratio
    # ----------------------------------------
    def test_extreme_aspect_ratio():
        img_wide = np.random.random((1, 128))
        img_tall = np.random.random((128, 1))
        slope_w = compute_plls(img_wide)
        slope_t = compute_plls(img_tall)
        assert np.isfinite(slope_w) or np.isnan(slope_w)
        assert np.isfinite(slope_t) or np.isnan(slope_t)

    # ----------------------------------------
    # 11 — Non-numeric input
    # ----------------------------------------
    def test_invalid_input():
        try:
            compute_plls(None)
        except Exception:
            pass
        else:
            assert False, "Should raise an error for None input"

        try:
            compute_plls("string")
        except Exception:
            pass
        else:
            assert False, "Should raise an error for string input"

    # ----------------------------------------
    # 12 — Negative values
    # ----------------------------------------
    def test_negative_values():
        img = np.random.random((64,64)) - 0.5  # [-0.5,0.5]
        slope = compute_plls(img)
        assert np.isfinite(slope) or np.isnan(slope)

    # ----------------------------------------
    # 13 — Very large image
    # ----------------------------------------
    def test_large_image():
        img = np.random.random((512,512))
        slope = compute_plls(img)
        assert np.isfinite(slope) or np.isnan(slope)

    # ----------------------------------------
    # 14 — Multi-channel image (RGB)
    # ----------------------------------------
    def test_multichannel_image():
        img = np.random.random((3, 64, 64))
        slopes = compute_plls(img, axis=0)
        assert slopes.shape[0] == 3
        for s in slopes:
            assert np.isfinite(s) or np.isnan(s)

    # ----------------------------------------
    # 15 — Single-pixel slices in z-stack
    # ----------------------------------------
    def test_single_pixel_zstack():
        img = np.random.random((5, 1, 1))
        slopes = compute_plls(img, axis=0)
        assert slopes.shape[0] == 5
        for s in slopes:
            assert np.isfinite(s) or np.isnan(s)

    # ----------------------------------------
    # 16 — Min/max intensity
    # ----------------------------------------
    def test_min_max_intensity():
        img_min = np.full((64,64), np.finfo(np.float32).min)
        img_max = np.full((64,64), np.finfo(np.float32).max)
        slope_min = compute_plls(img_min)
        slope_max = compute_plls(img_max)
        assert np.isnan(slope_min) or np.isfinite(slope_min)
        assert np.isnan(slope_max) or np.isfinite(slope_max)


    # ----------------------------------------
    # 17 — Plotting mode (does not crash)
    # ----------------------------------------
    def test_plotting_mode():
        img = np.random.random((64,64))
        for p in (False,"all", "fft", "radial", "loglog"):
            for ps in ("first","all"):
                slope, _, freqs_fit, powers_fit = compute_plls(img, plot=p, plot_slices=ps, return_details=True)
                assert np.isfinite(slope) or np.isnan(slope)
                assert len(freqs_fit) == len(powers_fit)

    # ----------------------------------------
    # Run all tests
    # ----------------------------------------
    test_functions = [
        test_gradient_image,
        test_noise_image,
        test_constant_image,
        test_axis_slicing,
        test_verbose_mode,
        test_tiny_images,
        test_nan_image,
        test_single_slice_zstack,
        test_dc_only_image,
        test_extreme_aspect_ratio,
        test_invalid_input,
        test_negative_values,
        test_large_image,
        test_multichannel_image,
        test_single_pixel_zstack,
        test_min_max_intensity,
        test_plotting_mode
    ]

    for tf in test_functions:
        run_test(tf.__name__, tf)

    print("\nAll tests completed successfully ✅")



def plls_vs_blur(
    image,
    plls_func,
    sigmas,
    return_values=False,
    plls_func_kwargs={},
):
    """
    Apply progressively stronger Gaussian blur to an image,
    compute PLLS for each blur level, and plot PLLS vs blur strength.

    Parameters
    ----------
    image : 2D numpy array
        Input image (grayscale).
    plls_func : callable
        Function that computes PLLS.
        Must return (slope, intercept, freqs, power).
    sigmas : array-like
        Gaussian blur sigmas to apply (in pixels).
    mask_zero : bool, optional
        Passed to PLLS function.
    return_values : bool, optional
        If True, return sigmas and slopes.

    Returns
    -------
    sigmas, slopes : arrays (optional)
    """

    slopes = []

    for sigma in sigmas:
        # Apply Gaussian blur
        if sigma == 0:
            blurred = image.copy()
        else:
            blurred = gaussian_filter(image, sigma=sigma)

        # Compute PLLS
        slope, _, _, _ = plls_func(blurred,**plls_func_kwargs)
        slopes.append(slope)

    slopes = np.array(slopes)

    # ----- Plot -----
    plt.figure(figsize=(6, 4))
    plt.plot(sigmas, slopes, marker="o")
    plt.xlabel("Gaussian blur σ (pixels)")
    plt.ylabel("PLLS (slope)")
    plt.title("PLLS vs blur strength")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(r'C:\Users\aless\OneDrive\Desktop\Ale\lab\CIID_IDIP\others\MIDAscience_bioimage_consultation\20260120_IDIP_MIDAscience_meeting\plls_vs_blur_test.png')

    if return_values:
        return np.array(sigmas), slopes
