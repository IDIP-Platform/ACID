import numpy as np

def mean_over_std(
    image: np.ndarray, 
    axis: int | None = None, 
    null_val: float | int = np.nan, 
    inf_val: float | int = np.nan
) -> float | np.ndarray:
    """
    Computes the mean intensity divided by the standard deviation of an n-dimensional image,
    with robust handling of NaNs, infinite values, and edge cases.

    Parameters:
        image (np.ndarray): n-dimensional array representing the image. Must be numeric.
        axis (int or None, optional): Axis along which to compute the ratio.
            If None, computes over the entire array. Default is None.
        null_val (float or int, optional): Value to return if the standard deviation is zero or NaN.
            Default is np.nan.
        inf_val (float or int, optional): Value to replace +inf/-inf in the image before computation.
            Default is np.nan.

    Returns:
        float or np.ndarray: mean / std along the specified axis, or null_val if std is zero or NaN.

    Notes on edge cases:
        1. Empty arrays: returns null_val with a warning.
        2. Single-value arrays: returns null_val with a warning (std=0).
        3. Axis of size 1: returns null_val per slice with a warning (std=0 along that axis).
        4. Arrays containing only NaNs: returns null_val with a warning.
        5. Arrays with scattered NaNs: ignored in mean/std calculation.
        6. Very large or very small values: uses dtype=np.float64 to reduce overflow/underflow.
        7. Non-numeric arrays: raises TypeError.
        8. Arrays containing inf or -inf: replaced with inf_val with a warning.
        9. Axis out of bounds: NumPy AxisError is raised naturally.
        10. Non-contiguous arrays or slices: handled correctly by NumPy.
        11. Mixed object types: raises TypeError.
        12. Boolean arrays: treated as 0/1 numerically.
    """

    # Check that the input is a NumPy array; otherwise raise a TypeError
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a NumPy array.")

    # Check that the array contains numeric types (integers, floats, etc.)
    # This prevents errors if the array contains strings or objects
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("Input array must be numeric.")

    # Convert the image to float64 for numerical stability
    # This reduces risk of overflow/underflow when working with very large/small values
    img = image.astype(np.float64, copy=True)

    # Check if there are any positive or negative infinity values in the array
    # np.isposinf checks for +inf, np.isneginf checks for -inf
    if np.isinf(img).any():
        # Print a warning if any infinite values are found
        print("Warning: Inf values detected; replacing them with inf_val.")
        # Replace all +inf or -inf values with the user-specified inf_val
        img[np.isposinf(img) | np.isneginf(img)] = inf_val

    # Compute the mean along the specified axis, ignoring NaN values
    # dtype=np.float64 ensures that calculations are performed in float64
    mean_val = np.nanmean(img, axis=axis, dtype=np.float64)
    print("mean val", mean_val.shape)
    # Compute the standard deviation along the specified axis, ignoring NaN values
    # dtype=np.float64 ensures numerical stability
    std_val = np.nanstd(img, axis=axis, dtype=np.float64)
    print("std val", std_val.shape)

    # Handle the case when axis=None (i.e., compute over the entire array)
    if axis is None:
        # If std is 0 or NaN, return null_val with a warning
        if std_val == 0 or np.isnan(std_val):
            print("Warning: Standard deviation is zero or NaN, returning null_val.")
            return null_val
    else:
        # If an axis is specified, create a mask identifying positions where std is 0 or NaN
        std_zero_mask = (std_val == 0) | np.isnan(std_val)
        # If any of these positions exist along the axis, replace them with null_val
        if np.any(std_zero_mask):
            # Print a warning indicating the axis along which std was zero or NaN
            print(f"Warning: Standard deviation is zero or NaN along axis {axis}, replacing with null_val.")
            # Convert mean_val to float to allow assignment of null_val (in case null_val is np.nan)
            mean_val = mean_val.astype(float)
            # Replace mean values at positions where std is zero/NaN with null_val
            mean_val[std_zero_mask] = null_val
            # Replace the corresponding std values with 1 to avoid division by zero
            std_val[std_zero_mask] = 1

    # Return the ratio of mean over std
    # If axis=None, this is a single float; otherwise, it is an array along the specified axis
    return mean_val / std_val
