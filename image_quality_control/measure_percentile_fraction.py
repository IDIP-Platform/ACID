import numpy as np
from typing import Tuple, Union, Optional, Any
from numpy.typing import ArrayLike
import warnings


def fraction_in_extreme_percentiles(
    image: ArrayLike,                                   # Input image data, any array-like structure
    percentiles: Tuple[float, float] = (1.0, 99.0),     # (low, high) percentiles defining extremes
    axis: Optional[int] = None,                          # Axis along which to compute statistics
    return_thresholds: bool = False,                     # Whether to return percentile thresholds
    null_val: Any = np.nan                               # Value returned when fractions are undefined
) -> Union[
    Tuple[Any, Any],
    Tuple[np.ndarray, np.ndarray],
    Tuple[Any, Any, Any, Any],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Compute the fraction of pixels below and above explicit percentile thresholds,
    ignoring NaN values and handling edge cases safely.

    Parameters
    ----------
    image : ArrayLike
        N-dimensional numeric image (int or float), possibly containing NaNs.
        Non-numeric arrays will raise an error.
    percentiles : tuple of float
        (low_percentile, high_percentile), values between 0 and 100.
    axis : int or None
        Axis along which percentiles and fractions are computed.
        If None, the computation is done on the flattened image.
        Only a single axis is supported; negative axis values are allowed.
    return_thresholds : bool
        If True, the percentile threshold values are also returned.
    null_val : any
        Value returned when fractions cannot be calculated (e.g., all NaNs).

    Returns
    -------
    bottom_fraction : float or np.ndarray
        Fraction of non-NaN values less than or equal to the low percentile.
    top_fraction : float or np.ndarray
        Fraction of non-NaN values greater than or equal to the high percentile.
    low_threshold : float or np.ndarray, optional
        Value corresponding to the low percentile (returned if return_thresholds=True).
    high_threshold : float or np.ndarray, optional
        Value corresponding to the high percentile (returned if return_thresholds=True).

    Notes / Additional Edge Cases
    -----------------------------
    - Constant-valued images: fractions = 1.0, thresholds = constant value.
    - Single-element arrays: fractions = 1.0, thresholds = the element value.
    - Axis of size 1: fractions = 1.0, thresholds = broadcast correctly.
    - Mixed NaNs: NaNs are ignored in both fraction computation and percentile calculation.
    - Negative axes: supported and behave as expected.
    - Inf/-Inf values: treated as valid numeric values.
    - Extremely small arrays relative to percentile: fractions may be quantized.
    - Ties at percentile thresholds: fraction may slightly exceed requested percentile.
    - Multi-axis computation (tuple of axes) is NOT supported in this version.
    - Sparse arrays are converted to dense arrays (could be memory-intensive).
    """

    # Convert the input image to a NumPy array
    image = np.asarray(image)

    # Assert that the image contains at least one element
    assert image.size > 0, "Input image must contain at least one element."

    # Assert that the image has a numeric dtype
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Input image must be numeric (int or float), got {image.dtype}")

    # Unpack the low and high percentile values from the tuple
    p_low, p_high = percentiles

    # Validate that percentiles are within valid bounds and ordered correctly
    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

    # Compute the low percentile threshold while ignoring NaN values
    low_thresh = np.nanpercentile(image, p_low, axis=axis)

    # Compute the high percentile threshold while ignoring NaN values
    high_thresh = np.nanpercentile(image, p_high, axis=axis)

    # Check whether computation is done on the entire image
    if axis is None:

        # Create a boolean mask identifying non-NaN values
        valid_mask = ~np.isnan(image)

        # Count the number of valid (non-NaN) elements
        total = np.count_nonzero(valid_mask)

        # Handle the case where no valid pixels are present
        if total == 0:
            warnings.warn("No valid (non-NaN) pixels found; returning null_val results.")
            if return_thresholds:
                return null_val, null_val, null_val, null_val
            else:
                return null_val, null_val

        # Count valid elements less than or equal to the low threshold
        bottom_fraction = np.count_nonzero((image <= low_thresh) & valid_mask) / total

        # Count valid elements greater than or equal to the high threshold
        top_fraction = np.count_nonzero((image >= high_thresh) & valid_mask) / total

    else:

        # Insert a singleton dimension so low thresholds broadcast correctly
        low_broadcast = np.expand_dims(low_thresh, axis)

        # Insert a singleton dimension so high thresholds broadcast correctly
        high_broadcast = np.expand_dims(high_thresh, axis)

        # Create a boolean mask identifying non-NaN values
        valid_mask = ~np.isnan(image)

        # Count valid (non-NaN) elements along the specified axis
        total = np.sum(valid_mask, axis=axis)

        # Identify slices with zero valid pixels
        zero_valid = total == 0

        # Count valid values below or equal to the low threshold along the axis
        bottom_fraction = np.sum((image <= low_broadcast) & valid_mask, axis=axis) / total

        # Count valid values above or equal to the high threshold along the axis
        top_fraction = np.sum((image >= high_broadcast) & valid_mask, axis=axis) / total

        # Replace undefined fractions with the specified null value
        bottom_fraction = np.where(zero_valid, null_val, bottom_fraction)

        # Replace undefined fractions with the specified null value
        top_fraction = np.where(zero_valid, null_val, top_fraction)

        # Emit a warning if any slice had no valid pixels
        if np.any(zero_valid):
            warnings.warn("One or more slices contained no valid (non-NaN) pixels.")

    # Decide whether percentile thresholds should be returned
    if return_thresholds:

        # Replace undefined thresholds with the specified null value if needed
        if axis is not None:
            low_thresh = np.where(zero_valid, null_val, low_thresh)
            high_thresh = np.where(zero_valid, null_val, high_thresh)

        # Return fractions together with the computed threshold values
        return bottom_fraction, top_fraction, low_thresh, high_thresh

    else:

        # Return only the fraction values
        return bottom_fraction, top_fraction
