import numpy as np
from typing import Tuple, Union, Optional, Any
from numpy.typing import ArrayLike
import warnings


def fraction_in_extreme_percentiles(
    image: ArrayLike,                                   # Input image data, any array-like structure
    percentiles: Tuple[float, float] = (1.0, 99.0),     # (low, high) percentiles defining extremes
    axis: Optional[int] = None,                          # Axis indexing independent sub-images
    return_thresholds: bool = False,                     # Whether to return percentile thresholds
    null_val: Any = np.nan                               # Value returned when fractions are undefined
) -> Union[
    Tuple[Any, Any],
    Tuple[np.ndarray, np.ndarray],
    Tuple[Any, Any, Any, Any],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Compute the fraction of pixels below and above explicit percentile thresholds.

    This function is designed for robust analysis of N-dimensional images and works
    independently of data type (int or float). It is robust to NaN values and
    supports slice-wise computation via an explicit axis definition.

    ────────────────────────────────────────────────────────────────────────────
    Semantics of `axis`
    ────────────────────────────────────────────────────────────────────────────

    - axis = None
        The computation is performed on the entire image.
        The image is flattened and treated as a single set of pixels.

    - axis = k
        The image is sliced along axis `k`, and the computation is performed
        independently for each slice.

        For example:
            image.shape = (1024, 1024, 5)
            axis = 2

        In this case, each slice image[:, :, i] is treated as an independent image,
        and the function returns one value per slice.

        The output arrays therefore have length image.shape[axis].

    ────────────────────────────────────────────────────────────────────────────
    Percentile definition
    ────────────────────────────────────────────────────────────────────────────

    - percentiles = (p_low, p_high)
        Defines explicit percentile thresholds.
        Pixels with values <= p_low percentile contribute to the bottom fraction.
        Pixels with values >= p_high percentile contribute to the top fraction.

    - Percentiles are computed using numpy.nanpercentile, so NaN values are ignored.

    ────────────────────────────────────────────────────────────────────────────
    NaN handling
    ────────────────────────────────────────────────────────────────────────────

    - NaN values are excluded from:
        - percentile computation
        - fraction numerators
        - fraction denominators

    - If an image or slice contains no valid (non-NaN) pixels:
        - A warning is issued
        - The returned fraction(s) are set to `null_val`
        - If return_thresholds=True, thresholds are also set to `null_val`

    ────────────────────────────────────────────────────────────────────────────
    Edge cases and guarantees
    ────────────────────────────────────────────────────────────────────────────

    - Empty input arrays:
        Raise an AssertionError.

    - Non-numeric input arrays:
        Raise a TypeError.

    - Single-element arrays:
        All percentiles collapse to the single value.
        Both bottom and top fractions are equal to 1.0.

    - Constant-valued arrays:
        All percentiles collapse to the constant value.
        Both bottom and top fractions are equal to 1.0.

    - Slices of size 1 along the selected axis:
        Behavior is identical to single-element arrays.

    - Ties at percentile boundaries:
        Inclusive comparisons (<=, >=) are used.
        Fractions may therefore slightly exceed the nominal percentile values.

    - Inf and -Inf values:
        Treated as valid numeric values.

    - Sparse inputs:
        Converted to dense arrays; memory usage may increase.

    - Multi-axis slicing:
        Not supported in this implementation.

    ────────────────────────────────────────────────────────────────────────────
    Return values
    ────────────────────────────────────────────────────────────────────────────

    - If axis is None:
        bottom_fraction, top_fraction are scalars.

    - If axis is specified:
        bottom_fraction, top_fraction are 1D arrays with length image.shape[axis].

    - If return_thresholds=True:
        low_threshold and high_threshold are also returned, matching the shape of
        the fraction outputs.

    ────────────────────────────────────────────────────────────────────────────
    Intended use
    ────────────────────────────────────────────────────────────────────────────

    This function is intended for robust intensity distribution analysis,
    quality control, and saturation detection in multidimensional image data.
    """

    # Convert input to NumPy array
    image = np.asarray(image)

    # Assert non-empty input
    assert image.size > 0, "Input image must contain at least one element."

    # Assert numeric dtype
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Input image must be numeric, got {image.dtype}")

    # Unpack percentiles
    p_low, p_high = percentiles

    # Validate percentile bounds
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

    # Handle whole-image computation
    if axis is None:

        # Flatten image
        pixels = image.ravel()

        # Mask valid values
        valid = ~np.isnan(pixels)

        # Count valid pixels
        total = np.count_nonzero(valid)

        # Handle all-NaN case
        if total == 0:
            warnings.warn("No valid (non-NaN) pixels found; returning null_val.")
            if return_thresholds:
                return null_val, null_val, null_val, null_val
            else:
                return null_val, null_val

        # Compute percentile thresholds
        low_thresh = np.nanpercentile(pixels, p_low)
        high_thresh = np.nanpercentile(pixels, p_high)

        # Compute fractions
        bottom_fraction = np.count_nonzero((pixels <= low_thresh) & valid) / total
        top_fraction = np.count_nonzero((pixels >= high_thresh) & valid) / total

        # Return results
        if return_thresholds:
            return bottom_fraction, top_fraction, low_thresh, high_thresh
        else:
            return bottom_fraction, top_fraction

    # Handle slice-wise computation
    else:

        # Normalize negative axis
        axis = axis % image.ndim

        # Number of slices
        n_slices = image.shape[axis]

        # Prepare output arrays
        bottom_fraction = np.empty(n_slices, dtype=float)
        top_fraction = np.empty(n_slices, dtype=float)
        low_thresh = np.empty(n_slices, dtype=float)
        high_thresh = np.empty(n_slices, dtype=float)

        # Iterate over slices
        for i in range(n_slices):

            # Extract slice
            slc = np.take(image, i, axis=axis)

            # Flatten slice
            pixels = slc.ravel()

            # Mask valid values
            valid = ~np.isnan(pixels)

            # Count valid pixels
            total = np.count_nonzero(valid)

            # Handle all-NaN slice
            if total == 0:
                bottom_fraction[i] = null_val
                top_fraction[i] = null_val
                low_thresh[i] = null_val
                high_thresh[i] = null_val
                continue

            # Compute percentile thresholds
            low = np.nanpercentile(pixels, p_low)
            high = np.nanpercentile(pixels, p_high)

            # Store thresholds
            low_thresh[i] = low
            high_thresh[i] = high

            # Compute fractions
            bottom_fraction[i] = np.count_nonzero((pixels <= low) & valid) / total
            top_fraction[i] = np.count_nonzero((pixels >= high) & valid) / total

        # Warn if any slice was invalid
        if np.any(bottom_fraction == null_val):
            warnings.warn("One or more slices contained no valid (non-NaN) pixels.")

        # Return results
        if return_thresholds:
            return bottom_fraction, top_fraction, low_thresh, high_thresh
        else:
            return bottom_fraction, top_fraction
