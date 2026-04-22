import numpy as np  # Import NumPy for numerical operations
from typing import Tuple, Union, Optional, Any  # Import typing utilities for clarity
from numpy.typing import ArrayLike  # Allow flexible array-like inputs
import warnings  # Used to emit warnings for edge cases
from skimage.filters import median  # Import median filter from scikit-image


def fraction_in_extreme_percentiles(
    image: ArrayLike,                                   # Input image data (can be list, NumPy array, etc.)
    percentiles: Tuple[float, float] = (1.0, 99.0),     # Lower and upper percentile thresholds
    axis: Optional[int] = None,                         # Axis along which to compute independent slices
    return_thresholds: bool = False,                    # Whether to also return the computed thresholds
    null_val: Any = np.nan,                             # Value returned when computation is undefined
    median_smooth: bool = True,                         # Whether to apply median filtering before analysis
    footprint: Optional[np.ndarray] = None,             # Neighborhood shape used for median filtering
    median_kwargs: Optional[dict] = None                # Additional keyword arguments for median filter
) -> Union[
    Tuple[Any, Any],
    Tuple[np.ndarray, np.ndarray],
    Tuple[Any, Any, Any, Any],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Compute the fraction of pixels that lie in the extreme intensity ranges
    of an image, optionally after applying median smoothing.

    ────────────────────────────────────────────────────────────────────────────
    What this function does (plain language)
    ────────────────────────────────────────────────────────────────────────────

    This function analyzes the distribution of pixel values in an image and
    answers two questions:

        1. What fraction of pixels are among the darkest values?
        2. What fraction of pixels are among the brightest values?

    These are defined using percentiles:
        - The "bottom fraction" = pixels below a lower percentile threshold
        - The "top fraction" = pixels above an upper percentile threshold

    Before computing these, the image can optionally be smoothed using a
    median filter to reduce noise.

    ────────────────────────────────────────────────────────────────────────────
    Median smoothing
    ────────────────────────────────────────────────────────────────────────────

    - If `median_smooth=True`, a median filter is applied before analysis.
    - This replaces each pixel with the median value of its neighborhood.
    - The neighborhood is defined by `footprint`.
    - Additional parameters can be passed via `median_kwargs`.

    Important note:
        Median filtering does NOT ignore NaN values by default.
        NaNs may propagate or affect results near them.

    ────────────────────────────────────────────────────────────────────────────
    Semantics of `axis`
    ────────────────────────────────────────────────────────────────────────────

    - axis = None:
        The entire image is treated as one dataset.

    - axis = k:
        The image is split into slices along axis `k`, and each slice is
        analyzed independently.

        Example:
            image.shape = (1024, 1024, 5), axis = 2

        → The function processes 5 independent images.

    ────────────────────────────────────────────────────────────────────────────
    Percentile definition
    ────────────────────────────────────────────────────────────────────────────

    - percentiles = (p_low, p_high)
        Defines the thresholds for "extreme" values.

    - Pixels ≤ p_low percentile → bottom fraction
    - Pixels ≥ p_high percentile → top fraction

    - Percentiles are computed using np.nanpercentile (ignores NaNs).

    ────────────────────────────────────────────────────────────────────────────
    NaN handling
    ────────────────────────────────────────────────────────────────────────────

    - NaN values are excluded from:
        - percentile computation
        - fraction calculation

    - If no valid pixels exist:
        → return `null_val` and emit a warning

    ────────────────────────────────────────────────────────────────────────────
    Returns
    ────────────────────────────────────────────────────────────────────────────

    - If axis is None:
        Returns scalars

    - If axis is specified:
        Returns arrays of length image.shape[axis]

    - If return_thresholds=True:
        Also returns the computed percentile thresholds

    ────────────────────────────────────────────────────────────────────────────
    Intended use
    ────────────────────────────────────────────────────────────────────────────

    Useful for:
        - detecting saturation (too bright / too dark regions)
        - quality control in imaging pipelines
        - analyzing intensity distributions robustly
    """

    image = np.asarray(image)  # Convert input to a NumPy array (ensures consistent behavior)

    assert image.size > 0, "Input image must contain at least one element."  # Ensure input is not empty

    if not np.issubdtype(image.dtype, np.number):  # Check that data is numeric
        raise TypeError(f"Input image must be numeric, got {image.dtype}")  # Raise error if not numeric

    p_low, p_high = percentiles  # Unpack lower and upper percentile values

    if not (0.0 <= p_low < p_high <= 100.0):  # Validate percentile range
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")  # Raise error if invalid

    if median_kwargs is None:  # If no extra median arguments provided
        median_kwargs = {}  # Use empty dictionary
    else:
        assert 'footprint' not in median_kwargs, "footprint can't be passed to median_kwargs, use the dedicated parameter instead"
    
    # ─────────────────────────────────────────────────────────────
    # Whole-image computation
    # ─────────────────────────────────────────────────────────────
    if axis is None:  # If no axis is specified

        data = image  # Work on the full image

        if median_smooth:  # If smoothing is enabled
            data = median(data, footprint=footprint, **median_kwargs)  # Apply median filter

        pixels = data.ravel()  # Flatten image into 1D array
        valid = ~np.isnan(pixels)  # Create mask of valid (non-NaN) pixels
        total = np.count_nonzero(valid)  # Count how many valid pixels exist

        if total == 0:  # If no valid pixels are present
            warnings.warn("No valid (non-NaN) pixels found; returning null_val.")  # Emit warning
            if return_thresholds:  # If thresholds requested
                return null_val, null_val, null_val, null_val  # Return null values
            else:
                return null_val, null_val  # Return null fractions

        low_thresh = np.nanpercentile(pixels, p_low)  # Compute lower percentile threshold
        high_thresh = np.nanpercentile(pixels, p_high)  # Compute upper percentile threshold

        bottom_fraction = np.count_nonzero((pixels <= low_thresh) & valid) / total  # Fraction below threshold
        top_fraction = np.count_nonzero((pixels >= high_thresh) & valid) / total  # Fraction above threshold

        if return_thresholds:  # If thresholds should be returned
            return bottom_fraction, top_fraction, low_thresh, high_thresh  # Return everything
        else:
            return bottom_fraction, top_fraction  # Return only fractions

    # ─────────────────────────────────────────────────────────────
    # Slice-wise computation
    # ─────────────────────────────────────────────────────────────
    else:

        axis = axis % image.ndim  # Normalize axis (handle negative values)
        n_slices = image.shape[axis]  # Determine number of slices

        bottom_fraction = np.empty(n_slices, dtype=float)  # Allocate array for bottom fractions
        top_fraction = np.empty(n_slices, dtype=float)  # Allocate array for top fractions
        low_thresh = np.empty(n_slices, dtype=float)  # Allocate array for lower thresholds
        high_thresh = np.empty(n_slices, dtype=float)  # Allocate array for upper thresholds

        for i in range(n_slices):  # Loop over each slice

            slc = np.take(image, i, axis=axis)  # Extract slice along chosen axis

            if median_smooth:  # If smoothing is enabled
                slc = median(slc, footprint=footprint, **median_kwargs)  # Apply median filter to slice

            pixels = slc.ravel()  # Flatten slice into 1D array
            valid = ~np.isnan(pixels)  # Identify valid pixels
            total = np.count_nonzero(valid)  # Count valid pixels

            if total == 0:  # If slice contains no valid data
                bottom_fraction[i] = null_val  # Assign null value
                top_fraction[i] = null_val  # Assign null value
                low_thresh[i] = null_val  # Assign null value
                high_thresh[i] = null_val  # Assign null value
                continue  # Skip to next slice

            low = np.nanpercentile(pixels, p_low)  # Compute lower threshold
            high = np.nanpercentile(pixels, p_high)  # Compute upper threshold

            low_thresh[i] = low  # Store lower threshold
            high_thresh[i] = high  # Store upper threshold

            bottom_fraction[i] = np.count_nonzero((pixels <= low) & valid) / total  # Compute bottom fraction
            top_fraction[i] = np.count_nonzero((pixels >= high) & valid) / total  # Compute top fraction

        if np.any(bottom_fraction == null_val):  # Check if any slice failed
            warnings.warn("One or more slices contained no valid (non-NaN) pixels.")  # Emit warning

        if return_thresholds:  # If thresholds requested
            return bottom_fraction, top_fraction, low_thresh, high_thresh  # Return everything
        else:
            return bottom_fraction, top_fraction  # Return only fractions

