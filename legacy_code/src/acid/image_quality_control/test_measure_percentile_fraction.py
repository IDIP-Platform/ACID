import numpy as np
import warnings
import matplotlib.pyplot as plt
from typing import Tuple

from image_quality_control.measure_percentile_fraction import (
    fraction_in_extreme_percentiles,
)


def run_all_tests():
    """Run a small suite of percentile-fraction tests and print results."""

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

    # ------------------------------------------------------------------
    # 1 — Simple increasing image
    # ------------------------------------------------------------------
    def test_linear_image():
        img = np.linspace(0, 100, 1000)
        bottom, top = fraction_in_extreme_percentiles(img, percentiles=(1, 99))
        # bottom should be very close to 0.01 (but inclusive comparisons
        # mean it may be slightly higher).  top likewise.
        assert abs(bottom - 0.01) < 0.005
        assert abs(top - 0.01) < 0.005

    # ------------------------------------------------------------------
    # 2 — Constant image
    # ------------------------------------------------------------------
    def test_constant_image():
        img = np.ones((50, 50)) * 7
        bottom, top, lowt, hight = fraction_in_extreme_percentiles(
            img, return_thresholds=True
        )
        assert bottom == 1.0
        assert top == 1.0
        assert lowt == 7.0
        assert hight == 7.0

    # ------------------------------------------------------------------
    # 3 — Axis slicing with NaN slice
    # ------------------------------------------------------------------
    def test_axis_slicing():
        img = np.zeros((3, 4, 4))
        img[0] = np.arange(16).reshape(4, 4)
        img[1] = 5
        img[2] = np.nan
        bot, top = fraction_in_extreme_percentiles(img, axis=0)
        assert len(bot) == 3
        assert np.isfinite(bot[0])
        assert bot[1] == 1.0
        assert np.isnan(bot[2])

    # ------------------------------------------------------------------
    # 4 — All-NaN input
    # ------------------------------------------------------------------
    def test_all_nan():
        img = np.full((10, 10), np.nan)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bottom, top = fraction_in_extreme_percentiles(img)
            assert bottom is np.nan or bottom == np.nan
            assert top is np.nan or top == np.nan
            assert len(w) >= 1

    # ------------------------------------------------------------------
    # 5 — Invalid inputs
    # ------------------------------------------------------------------
    def test_invalid_input():
        for bad in (None, "string", object()):
            try:
                fraction_in_extreme_percentiles(bad)
            except Exception:
                pass
            else:
                assert False, f"input {bad!r} should raise"

        # invalid percentiles
        try:
            fraction_in_extreme_percentiles(np.arange(10), percentiles=(50, 50))
        except ValueError:
            pass
        else:
            assert False, "equal percentiles should raise ValueError"

    # ------------------------------------------------------------------
    # 6 — Negative axis index
    # ------------------------------------------------------------------
    def test_negative_axis():
        img = np.arange(8).reshape(2, 2, 2)
        bot_pos, top_pos = fraction_in_extreme_percentiles(img, axis=0)
        bot_neg, top_neg = fraction_in_extreme_percentiles(img, axis=-3)
        assert np.allclose(bot_pos, bot_neg)
        assert np.allclose(top_pos, top_neg)

    # ------------------------------------------------------------------
    # 7 — Single-element input
    # ------------------------------------------------------------------
    def test_single_element():
        val = 42
        bot, top, lt, ht = fraction_in_extreme_percentiles(
            np.array([val]), return_thresholds=True
        )
        assert bot == 1.0
        assert top == 1.0
        assert lt == val
        assert ht == val

    # ------------------------------------------------------------------
    # 8 — Return thresholds shape with slicing
    # ------------------------------------------------------------------
    def test_return_thresholds_slicing():
        img = np.stack([np.linspace(0, 1, 10), np.linspace(1, 2, 10)])
        b, t, lowt, hight = fraction_in_extreme_percentiles(img, axis=0, return_thresholds=True)
        assert b.shape == (2,)
        assert lowt.shape == (2,)
        assert hight.shape == (2,)
        assert t.shape == (2,)

    # run all tests list
    tests = [
        test_linear_image,
        test_constant_image,
        test_axis_slicing,
        test_all_nan,
        test_invalid_input,
        test_negative_axis,
        test_single_element,
        test_return_thresholds_slicing,
    ]

    for t in tests:
        run_test(t.__name__, t)

    print("\nAll percentile-fraction tests passed ✅")


# ------------------------------------------------------------------
# exploratory contamination/plot function
# ------------------------------------------------------------------
def contamination_vs_saturation(
    image: np.ndarray,
    n_steps: int = 21,
    max_fraction: float = 0.5,
    percentile_tuple: Tuple[float, float] = (1.0, 99.0),
    max_fallback_percentile: float = 99.0,
    min_fallback_percentile: float = 1.0,
    axes=None,
    randomize: bool = True,
) -> None:
    """Analyze effect of adding saturated/zero pixels and plot results.

    Parameters
    ----------
    image : ndarray
        Base image that will be corrupted.  The array is converted to float.
    n_steps : int
        Number of contamination fractions to test (including 0 and ``max_fraction``).
    max_fraction : float
        Maximum fraction of pixels to corrupt (default 0.5 = 50%).
    percentile_tuple : tuple of two floats
        Percentile pair passed to :func:`fraction_in_extreme_percentiles`.
        Defaults to (1.0, 99.0) as used previously.
    max_fallback_percentile : float
        Percentile used to compute a substitute maximum when the real maximum
        is not finite.  Defaults to 99.
    min_fallback_percentile : float
        Percentile used to compute a substitute minimum when the real minimum
        is not finite.  Defaults to 1.
    axes : sequence or None
        Optional pair of :class:`matplotlib.axes.Axes` objects on which to
        draw the percentile-fraction curves.  If ``None`` the function
        creates its own figure.
    randomize : bool
        Flag controlling which pixels are corrupted.

        * If ``False`` (the default) pixels are selected by constructing a
          progressively larger central square within the un-flattened image
          (a 1×1 block at first, then 2×2, etc.).  This produces a spatially
          coherent corruption around the centre.

        * If ``True`` a purely random subset of pixels is chosen at each
          contamination level (using a fixed RNG for reproducibility).

        The previous semantics (random on ``True``) have been reversed
        per the latest user request.

    The routine selects sets of pixels whose size grows from 0 up to
    ``max_fraction`` of all pixels.  If ``randomize`` is False (the default)
    the pixels form a central square of the appropriate area; when True a
    random subset is chosen.  For each level we create two corruptions: one in
    which those pixels are set to the image maximum (saturation) and another
    where they are set to the minimum value.  For each corrupted image it
    calls :func:`fraction_in_extreme_percentiles` with ``percentile_tuple`` and
    records the bottom/top fractions.  If the raw maximum or minimum of the
    image is not finite, fallback percentiles (``max_fallback_percentile`` /
    ``min_fallback_percentile``) are used to compute surrogate extreme
    values.  Results are plotted in a 1x2 figure; if ``axes`` is supplied the
    existing axes are used instead of creating new ones.
    Returns
    -------
    tuple
        A pair ``(max_corrupted_sat, max_corrupted_min)`` giving the corrupted
        images (reshaped to the original image dimensions) corresponding to the
        highest tested contamination fraction for saturation and minimum-value
        corruption respectively.  These may be ``None`` if ``n_steps`` is
        zero.    """

    # check that an image was actually provided
    if image is None:
        raise ValueError("image must be provided")

    # convert input to a floating numpy array so we can assign extreme values
    arr = np.asarray(image, dtype=float)

    # guard against zero-sized images because nothing sensible can be done
    if arr.size == 0:
        raise ValueError("image must contain at least one element")

    # flatten the array for easier random pixel selection
    flat = arr.ravel()

    # total number of pixels we'll possibly corrupt
    n_pixels = flat.size

    # determine a saturation value: ordinarily the max of the data
    maxval = flat.max()
    if not np.isfinite(maxval):
        # if the maximum is NaN or infinite, fall back to a high percentile
        # of the valid pixels.  ``max_fallback_percentile`` is provided by the
        # caller and replaces the previous single ``fallback_percentile``.
        valid = flat[~np.isnan(flat)]
        if valid.size == 0:
            raise ValueError("cannot determine saturation value from all-NaN image")
        maxval = np.nanpercentile(valid, max_fallback_percentile)

    # likewise determine a minimum value for the "zeroed" corruption.  here we
    # use ``min_fallback_percentile`` directly rather than computing 100 - p.
    minval = flat.min()
    if not np.isfinite(minval):
        valid = flat[~np.isnan(flat)]
        if valid.size == 0:
            raise ValueError("cannot determine minimum value from all-NaN image")
        minval = np.nanpercentile(valid, min_fallback_percentile)

    # fractions of the image to corrupt (from 0 to 0.5 in ``n_steps`` steps)
    fractions = np.linspace(0, max_fraction, n_steps)

    # lists that will collect results for each corruption level
    bottom_sat = []  # bottom-percentile for saturated case
    top_sat = []     # top-percentile for saturated case
    bottom_zero = [] # bottom-percentile for zero case
    top_zero = []    # top-percentile for zero case

    # fixed RNG ensures reproducibility of which pixels are chosen
    rng = np.random.default_rng(0)

    # placeholders for the two extreme corrupted images
    max_corrupted_sat = None
    max_corrupted_min = None

    for i, frac in enumerate(fractions):
        # number of pixels to replace at this corruption level
        count = int(np.round(frac * n_pixels))

        # select pixel indices depending on randomize flag
        # ``randomize`` controls whether we choose pixels randomly or in a
        # structured, deterministic pattern.
        if randomize:
            # When randomize is True we draw ``count`` distinct indices at
            # random from the set of all pixel positions.  A fixed random
            # number generator seed ensures the same sequence on every call.
            idx = rng.choice(n_pixels, size=count, replace=False)
        else:
            # When randomize is False we create a square block centered within
            # the two-dimensional image.  The block is enlarged as ``count``
            # increases, starting from a single centre pixel and growing
            # outward in a symmetric fashion.
            if arr.ndim >= 2:
                # height and width of the image
                h, w = arr.shape[0], arr.shape[1]

                # compute a side length for a square whose area is at least
                # ``count`` pixels; round up to ensure we cover enough pixels.
                side = int(np.ceil(np.sqrt(count))) if count > 0 else 0

                # make sure the square doesn't exceed image boundaries
                side = min(side, h, w)

                if side > 0:
                    # compute starting row/column so the square is centered
                    row_start = (h - side) // 2
                    col_start = (w - side) // 2

                    # generate row/column coordinates for every position in
                    # the square using meshgrid; ``rr`` and ``cc`` each have
                    # shape (side, side).
                    rr, cc = np.meshgrid(
                        np.arange(row_start, row_start + side),
                        np.arange(col_start, col_start + side),
                        indexing="ij",
                    )

                    # convert the 2‑D coordinates to flat indices into the
                    # flattened array (row * width + col) and then linearize
                    # the results with ``ravel()``.
                    flat_indices = (rr * w + cc).ravel()

                    # keep only the first ``count`` indices in case the square
                    # contained slightly more pixels than required.
                    idx = flat_indices[:count]
                else:
                    # if side is zero (count was zero) just use an empty array
                    idx = np.array([], dtype=int)
            else:
                # for 1-D inputs there is no 2-D centre; revert to the simple
                # first-N-behaviour so that the function still works.
                idx = np.arange(count, dtype=int)

        # ----- saturated version -----
        corrupted = flat.copy()              # start with original pixel values
        corrupted[idx] = maxval             # set selected pixels to saturation value
        b, t = fraction_in_extreme_percentiles(corrupted, percentiles=percentile_tuple)
        bottom_sat.append(b)                # record bottom fraction result
        top_sat.append(t)                   # record top fraction result
        if i == len(fractions) - 1:
            # reshape the highest-fraction corrupted data back into original shape
            max_corrupted_sat = corrupted.reshape(arr.shape)

        # "zeroed" version – actually set to the minimum value determined
        # earlier, which may not be literal zero (useful for signed/shifted data)
        corrupted = flat.copy()              # restore original values
        corrupted[idx] = minval             # set selected pixels to minimum
        b, t = fraction_in_extreme_percentiles(corrupted, percentiles=percentile_tuple)
        bottom_zero.append(b)               # store bottom fraction
        top_zero.append(t)                  # store top fraction
        if i == len(fractions) - 1:
            max_corrupted_min = corrupted.reshape(arr.shape)

    # if axes were not passed in, make our own figure/axes
    if axes is None:
        fig, axes = plt.subplots(1, 5, figsize=(14, 8))
    else:
        # expect an iterable with two axes objects
        try:
            ax0, ax1, ax2, ax3, ax4 = axes
        except Exception:
            raise ValueError("axes must be iterable with five elements")
        axes = (ax0, ax1, ax2, ax3, ax4)

    axes[0].imshow(arr, cmap='gray', aspect='auto')
    axes[0].set_title('Original image')
    axes[0].axis('off') # hide axes for image display

    axes[1].imshow(max_corrupted_sat, cmap='gray', aspect='auto')
    axes[1].set_title(f'Max-value/saturation \n corruption ({fractions[-1]:.2f} fraction)')
    axes[1].axis('off') # hide axes for image display

    axes[2].imshow(max_corrupted_min, cmap='gray', aspect='auto')
    axes[2].set_title(f'Minimum-value/underexposure \n corruption ({fractions[-1]:.2f} fraction)')
    axes[2].axis('off') # hide axes for image display

    # plot saturated results on left panel
    # axes[0].plot(fractions, bottom_sat, label="bottom")
    axes[3].plot(fractions, top_sat, label="top")
    axes[3].set_title("Maxval/saturation")
    axes[3].set_xlabel("fraction maxvalued")
    axes[3].set_ylabel("percentile fraction")
    axes[3].legend()

    # plot zeroed results on right panel
    axes[4].plot(fractions, bottom_zero, label="bottom")
    # axes[1].plot(fractions, top_zero, label="top")
    axes[4].set_title("Minval/underexposure")
    axes[4].set_xlabel("fraction minvalued")
    axes[4].set_ylabel("percentile fraction")
    axes[4].legend()

    # if we created our own figure, tidy layout/show; otherwise caller handles it
    if 'fig' in locals():
        fig.tight_layout()
        plt.show()
