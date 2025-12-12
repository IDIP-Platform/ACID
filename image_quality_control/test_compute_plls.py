import numpy as np
from compute_plls import compute_plls  # adjust path if needed


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
        slope, _, _, _ = compute_plls(img)
        assert np.isfinite(slope), "Slope should be finite for gradient"

    # ----------------------------------------
    # 2 — Noise image
    # ----------------------------------------
    def test_noise_image():
        rng = np.random.default_rng(123)
        img = rng.normal(0, 1, (128, 128))
        slope, _, _, _ = compute_plls(img)
        assert np.isfinite(slope)
        assert -0.5 < slope < 0.5

    # ----------------------------------------
    # 3 — Constant image
    # ----------------------------------------
    def test_constant_image():
        img = np.ones((128, 128))
        slope, _, _, _ = compute_plls(img)
        assert np.isnan(slope)

    # ----------------------------------------
    # 4 — Z-stack slicing
    # ----------------------------------------
    def test_axis_slicing():
        img = np.zeros((3, 64, 64))
        img[0] = np.random.random((64, 64))
        img[1] = np.linspace(0, 1, 64).reshape(64,1)
        img[2] = 5
        slopes, _, _, _ = compute_plls(img, axis=0)
        assert len(slopes) == 3
        assert np.isfinite(slopes[0])
        assert np.isfinite(slopes[1])
        assert np.isnan(slopes[2])

    # ----------------------------------------
    # 5 — Verbose mode
    # ----------------------------------------
    def test_verbose_mode():
        img = np.random.random((64, 64))
        slope, _, _, _ = compute_plls(img, verbose=2)
        assert np.isfinite(slope)

    # ----------------------------------------
    # 6 — Tiny images
    # ----------------------------------------
    def test_tiny_images():
        for size in [1, 2, 3]:
            img = np.random.random((size, size))
            slope, _, _, _ = compute_plls(img)
            assert slope is None or np.isfinite(slope) or np.isnan(slope)

    # ----------------------------------------
    # 7 — Image with NaNs
    # ----------------------------------------
    def test_nan_image():
        img = np.ones((64, 64))
        img[10,10] = np.nan
        slope, _, _, _ = compute_plls(img)
        assert np.isnan(slope) or np.isfinite(slope)

    # ----------------------------------------
    # 8 — Single-slice z-stack
    # ----------------------------------------
    def test_single_slice_zstack():
        img = np.random.random((1, 128, 128))
        slopes, _, _, _ = compute_plls(img, axis=0)
        assert slopes.shape[0] == 1
        assert np.isfinite(slopes[0])

    # ----------------------------------------
    # 9 — Only DC component
    # ----------------------------------------
    def test_dc_only_image():
        img = np.full((128, 128), 7.0)
        slope, _, _, _ = compute_plls(img)
        assert np.isnan(slope)

    # ----------------------------------------
    # 10 — Extreme aspect ratio
    # ----------------------------------------
    def test_extreme_aspect_ratio():
        img_wide = np.random.random((1, 128))
        img_tall = np.random.random((128, 1))
        slope_w, _, _, _ = compute_plls(img_wide)
        slope_t, _, _, _ = compute_plls(img_tall)
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
        slope, _, _, _ = compute_plls(img)
        assert np.isfinite(slope) or np.isnan(slope)

    # ----------------------------------------
    # 13 — Very large image
    # ----------------------------------------
    def test_large_image():
        img = np.random.random((512,512))
        slope, _, _, _ = compute_plls(img)
        assert np.isfinite(slope) or np.isnan(slope)

    # ----------------------------------------
    # 14 — Multi-channel image (RGB)
    # ----------------------------------------
    def test_multichannel_image():
        img = np.random.random((3, 64, 64))
        slopes, _, _, _ = compute_plls(img, axis=0)
        assert slopes.shape[0] == 3
        for s in slopes:
            assert np.isfinite(s) or np.isnan(s)

    # ----------------------------------------
    # 15 — Single-pixel slices in z-stack
    # ----------------------------------------
    def test_single_pixel_zstack():
        img = np.random.random((5, 1, 1))
        slopes, _, _, _ = compute_plls(img, axis=0)
        assert slopes.shape[0] == 5
        for s in slopes:
            assert np.isfinite(s) or np.isnan(s)

    # ----------------------------------------
    # 16 — Min/max intensity
    # ----------------------------------------
    def test_min_max_intensity():
        img_min = np.full((64,64), np.finfo(np.float32).min)
        img_max = np.full((64,64), np.finfo(np.float32).max)
        slope_min, _, _, _ = compute_plls(img_min)
        slope_max, _, _, _ = compute_plls(img_max)
        assert np.isnan(slope_min) or np.isfinite(slope_min)
        assert np.isnan(slope_max) or np.isfinite(slope_max)

    # ----------------------------------------
    # 17 — Numba vs fallback consistency
    # ----------------------------------------
    def test_numba_vs_fallback():
        # Only run if your compute_plls has a switch for Numba
        img = np.random.random((64,64))
        # Force Numba off
        slopes1, _, _, _ = compute_plls(img, use_numba=False)
        # Force Numba on
        slopes2, _, _, _ = compute_plls(img, use_numba=True)
        assert np.allclose(slopes1, slopes2, atol=1e-6)

    # ----------------------------------------
    # 18 — Plotting mode (does not crash)
    # ----------------------------------------
    def test_plotting_mode():
        img = np.random.random((64,64))
        slope, _, freqs_fit, powers_fit = compute_plls(img, plot=True)
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
        test_numba_vs_fallback,
        test_plotting_mode
    ]

    for tf in test_functions:
        run_test(tf.__name__, tf)

    print("\nAll tests completed successfully ✅")
