import numpy as np
from typing import Union
from scipy.stats import skew

def _validate_image(image: np.ndarray) -> None:
    """
    Validate that the input is a proper image array.

    Parameters
    ----------
    image : np.ndarray
        The array to validate.

    Raises
    ------
    TypeError
        If image is not a NumPy ndarray.
    ValueError
        If image has fewer than 2 dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"image must be a NumPy ndarray, got {type(image).__name__} instead."
        )
    if image.ndim < 2:
        raise ValueError(
            f"image must have at least 2 dimensions, got {image.ndim}."
        )


def image_stat(image: np.ndarray, stat: callable, **kwargs) -> float:
    """
    Apply a scalar-valued statistical function to an image array.

    Parameters
    ----------
    image : np.ndarray
        Input image as an n-dimensional NumPy array (grayscale, RGB, etc.).
    stat : callable
        A function that takes a NumPy array and returns a single scalar value.
        Examples: np.mean, np.median, np.std, or any custom function.
    **kwargs
        Optional keyword arguments forwarded to stat.

    Returns
    -------
    float
        The scalar result of applying stat to the image.

    Examples
    --------
    >>> image_stat(img, np.mean)
    127.4
    >>> image_stat(img, np.percentile, q=95)
    231.0
    """
    return stat(image, **kwargs)


def masked_image_stat(
    image: np.ndarray,
    stat: callable,
    mask: np.ndarray = None,
    **kwargs
) -> float:
    """
    Apply a scalar-valued statistical function to an image array,
    optionally restricted to a masked region.

    Parameters
    ----------
    image : np.ndarray
        Input image as an n-dimensional NumPy array (grayscale, RGB, etc.).
    stat : callable
        A function that takes a NumPy array and returns a single scalar value.
        Examples: np.mean, np.median, np.std, or any custom function.
    mask : np.ndarray or None, optional
        A mask array with the same shape as image. Values must be >= 0.
        0 is treated as background (excluded). Positive values are included.
        If None, stat is applied to the entire image.
    kwargs
        Optional keyword arguments forwarded to stat.

    Returns
    -------
    float
        The scalar result of applying stat to the (masked) image.

    Raises
    ------
    ValueError
        If mask shape does not match image shape.
        If mask contains negative values.

    Examples
    --------
    >>> masked_image_stat(img, np.mean)
    127.4
    >>> masked_image_stat(img, np.mean, mask=roi_mask)
    143.2
    >>> masked_image_stat(img, np.percentile, mask=roi_mask, q=95)
    221.0
    """
    assert 'mask' not in kwargs, "mask can't be passed to kwargs, use the dedicated paramenter instead"
    
    if mask is None:
        return image_stat(image, stat, **kwargs)

    if mask.shape != image.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image.shape}."
        )
    if np.any(mask < 0):
        raise ValueError("Mask contains negative values. All mask values must be >= 0.")

    return image_stat(image[mask > 0], stat, **kwargs)


def axis_image_stat(
    image: np.ndarray,
    stat: callable,
    axis: int = None,
    mask: np.ndarray = None,
    **kwargs
) -> Union[float, np.ndarray]:
    """
    Apply a scalar-valued statistical function along an axis of an image array,
    optionally restricted to a masked region, returning one value per slice.
    If axis is None, falls back to masked_image_stat.

    Parameters
    ----------
    image : np.ndarray
        Input image as an n-dimensional NumPy array (grayscale, RGB, etc.).
    stat : callable
        A function that takes a NumPy array and returns a single scalar value.
        Examples: np.mean, np.median, np.std, or any custom function.
    axis : int or None, optional
        The axis along which to iterate. Each slice along this axis is passed
        independently to masked_image_stat.
        If None, falls back to masked_image_stat on the whole image.
    mask : np.ndarray or None, optional
        A mask array with the same shape as image. Values must be >= 0.
        0 is treated as background (excluded). Positive values are included.
        If None, stat is applied to the entire slice.
    kwargs
        Optional keyword arguments forwarded to stat.

    Returns
    -------
    float or np.ndarray
        A scalar if axis is None, or a 1D array of scalars if axis is provided.

    Raises
    ------
    ValueError
        If mask shape does not match image shape.
        If mask contains negative values.

    Examples
    --------
    >>> axis_image_stat(img, np.mean)
    127.4
    >>> axis_image_stat(img, np.mean, axis=0)
    array([120.1, 134.5, 98.2, ...])
    >>> axis_image_stat(img, np.mean, axis=2, mask=roi_mask)
    array([113.0, 128.7, 145.3])
    """
    if axis is None:
        return masked_image_stat(image, stat, mask=mask, **kwargs)

    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {image.shape}."
            )
        if np.any(mask < 0):
            raise ValueError("Mask contains negative values. All mask values must be >= 0.")

    slices      = np.moveaxis(image, axis, 0)
    mask_slices = np.moveaxis(mask, axis, 0) if mask is not None else [None] * image.shape[axis]

    return np.array([
        masked_image_stat(sub_image, stat, mask=sub_mask, **kwargs)
        for sub_image, sub_mask in zip(slices, mask_slices)
    ])


def broadcast_mask_image_stat(
    image: np.ndarray,
    stat: callable,
    axis: int = None,
    mask: np.ndarray = None,
    **kwargs
) -> Union[float, np.ndarray]:
    """
    Apply a scalar-valued statistical function along an axis of an image array,
    broadcasting a single slice mask across all slices along that axis.

    Parameters
    ----------
    image : np.ndarray
        Input image as an n-dimensional NumPy array (grayscale, RGB, etc.).
    stat : callable
        A function that takes a NumPy array and returns a single scalar value.
        Examples: np.mean, np.median, np.std, or any custom function.
    axis : int or None, optional
        The axis along which to iterate. The same mask is applied to every
        slice along this axis.
        If None and mask is None, falls back to axis_image_stat.
        If None and mask is not None, raises an error.
    mask : np.ndarray or None, optional
        A mask array with the same shape as a single slice of image along axis.
        Values must be >= 0. 0 is treated as background (excluded).
        Positive values are included.
        If None, falls back to axis_image_stat.
    **kwargs
        Optional keyword arguments forwarded to stat.

    Returns
    -------
    float or np.ndarray
        A scalar if axis is None, or a 1D array of scalars if axis is provided.

    Raises
    ------
    ValueError
        If axis is None and mask is not None.
        If mask shape does not match the shape of a single slice along axis.
        If mask contains negative values.

    Examples
    --------
    >>> broadcast_mask_image_stat(img, np.mean)
    127.4
    >>> broadcast_mask_image_stat(img, np.mean, axis=0, mask=slice_mask)
    array([120.1, 134.5, 98.2, ...])
    """
    if axis is None and mask is None:
        return axis_image_stat(image, stat, **kwargs)

    if axis is None and mask is not None:
        raise ValueError(
            "mask cannot be provided without an axis. "
            "Please provide an axis along which to broadcast the mask."
        )

    if mask is None:
        return axis_image_stat(image, stat, axis=axis, **kwargs)

    # Validate mask against a single slice shape
    slices = np.moveaxis(image, axis, 0)
    expected_shape = slices[0].shape
    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match slice shape {expected_shape} "
            f"of image along axis {axis}."
        )
    if np.any(mask < 0):
        raise ValueError("Mask contains negative values. All mask values must be >= 0.")

    return np.array([
        masked_image_stat(sub_image, stat, mask=mask, **kwargs)
        for sub_image in slices
    ])


def aggregated_axis_image_stat(
    image: np.ndarray,
    stat: callable,
    axis: int = None,
    mask: np.ndarray = None,
    agg: callable = None,
    agg_kwargs: dict = None,
    **kwargs
) -> Union[float, np.ndarray]:
    """
    Apply axis_image_stat and optionally aggregate the resulting values.

    Parameters
    ----------
    image : np.ndarray
        Input image as an n-dimensional NumPy array (grayscale, RGB, etc.).
    stat : callable
        A function that takes a NumPy array and returns a single scalar value.
    axis : int or None, optional
        The axis along which to iterate. If None, falls back to masked_image_stat.
    mask : np.ndarray or None, optional
        A mask array with the same shape as image. Values must be >= 0.
        0 is treated as background (excluded). Positive values are included.
    agg : callable or None, optional
        A function to aggregate the array of per-slice results into a single scalar.
        Examples: np.mean, np.sum, np.percentile.
        If None, the full array of per-slice results is returned as-is.
    agg_kwargs : dict or None, optional
        Keyword arguments to forward to agg. For example, {"q": 95} for np.percentile.
        If None, agg is called with no extra arguments.
    **kwargs
        Optional keyword arguments forwarded to stat.

    Returns
    -------
    float or np.ndarray
        A scalar if agg is provided or axis is None, otherwise a 1D array of scalars.

    Examples
    --------
    >>> aggregated_axis_image_stat(img, np.mean, axis=0)
    array([120.1, 134.5, 98.2, ...])
    >>> aggregated_axis_image_stat(img, np.mean, axis=0, agg=np.mean)
    117.3
    >>> aggregated_axis_image_stat(img, np.mean, axis=0, agg=np.percentile, agg_kwargs={"q": 95})
    201.4
    """
    result = axis_image_stat(image, stat, axis=axis, mask=mask, **kwargs)

    if agg is None:
        return result

    return agg(result, **(agg_kwargs or {}))


def intensity_skewness(image: np.ndarray, ravel_kwargs: dict = None, **kwargs) -> float:
    """
    Compute the skewness of the image intensity histogram.

    Parameters
    ----------
    image : np.ndarray
        Input image as an n-dimensional NumPy array (grayscale, RGB, etc.).
        The skewness is computed over all pixel values flattened.
    ravel_kwargs : dict or None, optional
        Keyword arguments forwarded to np.ndarray.ravel (e.g. {"order": "F"}).
        If None, ravel is called with default arguments.
    **kwargs
        Keyword arguments forwarded to scipy.stats.skew
        (e.g. bias=False, nan_policy='omit').

    Returns
    -------
    float
        Skewness of the intensity distribution.
        0 indicates symmetry, >0 right-skewed, <0 left-skewed.

    Examples
    --------
    >>> intensity_skewness(img)
    0.43
    >>> intensity_skewness(img, bias=False)
    0.45
    >>> image_stat(img, intensity_skewness)
    0.43
    >>> masked_image_stat(img, intensity_skewness, mask=roi_mask)
    0.61
    >>> axis_image_stat(img, intensity_skewness, axis=2)
    array([0.43, 0.21, 0.61])
    """
    return float(skew(image.ravel(**(ravel_kwargs or {})), **kwargs))


def normalized_intensity_skewness(
    image: np.ndarray,
    normalize: bool = True,
    ravel_kwargs: dict = None,
    **kwargs
) -> float:
    """
    Compute the skewness of the image intensity histogram,
    optionally after min-max normalization.

    Parameters
    ----------
    image : np.ndarray
        Input image as an n-dimensional NumPy array (grayscale, RGB, etc.).
    normalize : bool, optional
        If True (default), min-max normalizes the image to [0, 1] before
        computing skewness. If False, behaves exactly as intensity_skewness.
    ravel_kwargs : dict or None, optional
        Keyword arguments forwarded to np.ndarray.ravel (e.g. {"order": "F"}).
        If None, ravel is called with default arguments.
    **kwargs
        Keyword arguments forwarded to scipy.stats.skew
        (e.g. bias=False, nan_policy='omit').

    Returns
    -------
    float
        Skewness of the (optionally normalized) intensity distribution.
        0 indicates symmetry, >0 right-skewed, <0 left-skewed.

    Examples
    --------
    >>> normalized_intensity_skewness(img)
    0.43
    >>> normalized_intensity_skewness(img, normalize=False)
    0.43
    >>> image_stat(img, normalized_intensity_skewness)
    0.43
    >>> image_stat(img, normalized_intensity_skewness, normalize=False)
    0.43
    >>> masked_image_stat(img, normalized_intensity_skewness, mask=roi_mask)
    0.61
    """
    if normalize:
        min_val = image.min()
        max_val = image.max()
        image = (image - min_val) / (max_val - min_val) if max_val > min_val else image

    return intensity_skewness(image, ravel_kwargs=ravel_kwargs, **kwargs)
