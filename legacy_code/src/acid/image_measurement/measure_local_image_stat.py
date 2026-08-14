from typing import Any, Callable, Optional, Sequence, Union
from scipy.ndimage import generic_filter
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


def local_map(
    image: ArrayLike,
    func: Callable[[NDArray[np.number]], Union[int, float, np.number]],
    size: Optional[Union[int, Sequence[int]]] = None,
    footprint: Optional[ArrayLike] = None,
    axes: Optional[Sequence[int]] = None,
    dtype: Optional[DTypeLike] = None,
    **kwargs: Any,
) -> NDArray:
    """
    Apply a scalar-returning local function over an image using a moving neighborhood.

    This function is a thin wrapper around
    :contentReference[oaicite:0]{index=0}
    `scipy.ndimage.generic_filter`, with additional validation and optional
    output casting.

    The function evaluates a user-defined callable locally across an image.
    For each pixel position, a neighborhood is extracted according to either
    `size` or `footprint`, flattened into a 1D array, and passed to `func`.

    Parameters
    ----------
    image : ArrayLike
        Input image to process.

        Accepted types include:
        - numpy.ndarray
        - list
        - tuple
        - array-like objects convertible to ndarray

        Shape:
        - arbitrary dimensionality
        - examples:
            (H, W)
            (Z, H, W)
            (T, Z, H, W)

    func : Callable[[NDArray], scalar]
        Scalar-returning callable applied to each neighborhood.

        Expected signature:

            func(values) -> scalar

        where:
        - `values` is a 1D NumPy array
        - values contain pixels selected by `size` or `footprint`
        - output must be scalar-compatible

        Accepted return types:
        - int
        - float
        - numpy scalar types

        Examples:
        - np.mean
        - np.std
        - np.median
        - custom local statistics

    size : int or Sequence[int], optional
        Window size passed to scipy.ndimage.generic_filter.

        Accepted forms:
        - int
            same size applied to all filtered axes
        - tuple/list of ints
            per-axis neighborhood size

        Examples:
        - 5
        - (5, 5)
        - (3, 5, 5)

        Notes:
        - mutually exclusive with `footprint`
        - passed directly to generic_filter

    footprint : ArrayLike, optional
        Boolean neighborhood mask defining local support.

        Accepted forms:
        - ndarray
        - list-like arrays convertible to ndarray

        Internally converted to:
        - np.ndarray(dtype=bool)

        Examples:
        - square mask
        - circular disk
        - anisotropic footprint
        - sparse neighborhood

        Notes:
        - mutually exclusive with `size`
        - shape determines neighborhood geometry

    axes : Sequence[int], optional
        Axes along which filtering is applied.

        Accepted forms:
        - tuple of ints
        - list of ints

        Examples:
        - (0, 1)
        - (1, 2)

        Notes:
        - forwarded to generic_filter
        - useful for selectively filtering dimensions

    dtype : DTypeLike, optional
        Output dtype conversion.

        Accepted forms:
        - numpy dtype
        - Python type
        - string dtype name

        Examples:
        - np.float32
        - np.uint16
        - "float64"

        Notes:
        - applied after filtering
        - if None, original generic_filter dtype is preserved

    **kwargs : Any
        Additional keyword arguments forwarded directly to
        scipy.ndimage.generic_filter.

        Common options include:
        - mode : str
        - cval : float
        - origin : int or tuple
        - output : ndarray or dtype
        - extra_arguments : tuple
        - extra_keywords : dict

        Reserved keywords not allowed in kwargs:
        - function
        - size
        - footprint
        - axes

        Passing these through kwargs raises TypeError.

    Returns
    -------
    output : NDArray
        Filtered image with same shape as input.

        Type:
        - numpy.ndarray

        Shape:
        - identical to input image shape

        Dtype:
        - generic_filter default dtype
        - or `dtype` if provided

    Raises
    ------
    TypeError
        If reserved parameters are duplicated inside kwargs.

    Notes
    -----
    Neighborhood values passed to `func` are flattened to 1D.

    This function is intended as a reusable primitive for local image metrics,
    including:
    - local contrast
    - local variance
    - entropy-like measures
    - percentile filters
    - microscopy artifact scoring

    Examples
    --------
    Local mean using footprint:

    >>> from skimage.morphology import disk
    >>> result = local_map(image, np.mean, footprint=disk(5))

    Local standard deviation using size:

    >>> result = local_map(image, np.std, size=7)

    Custom local contrast:

    >>> def contrast(values):
    ...     center = values[len(values) // 2]
    ...     return center - values.mean()
    >>> result = local_map(image, contrast, footprint=disk(7))
    """

    # Convert input image to a NumPy array for compatibility.
    image = np.asarray(image)

    # Convert footprint to boolean ndarray if provided.
    if footprint is not None:
        footprint = np.asarray(footprint, dtype=bool)

    # Define argument names managed explicitly by this wrapper.
    reserved = {"function", "size", "footprint", "axes"}

    # Detect duplicate arguments passed both explicitly and via kwargs.
    duplicates = reserved.intersection(kwargs)

    # Raise a clear error if duplicated parameters are detected.
    if duplicates:
        duplicate_str = ", ".join(sorted(duplicates))
        raise TypeError(
            f"Arguments passed both explicitly and via kwargs: {duplicate_str}"
        )

    # Apply scipy generic_filter using explicit and forwarded arguments.
    output = generic_filter(
        image=image,
        function=func,
        size=size,
        footprint=footprint,
        axes=axes,
        **kwargs,
    )

    # Cast output dtype if requested by caller.
    if dtype is not None:
        output = output.astype(dtype)

    # Return filtered image.
    return output


from typing import Any, Callable, Optional, Sequence, Union
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


def aggregate_local_stat(
    image: ArrayLike,
    local_func: Callable[[NDArray[np.number]], Union[int, float, np.number]],
    aggregate_func: Callable[..., Union[int, float, np.number]],
    size: Optional[Union[int, Sequence[int]]] = None,
    footprint: Optional[ArrayLike] = None,
    axes: Optional[Sequence[int]] = None,
    dtype: Optional[DTypeLike] = None,
    agg_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[int, float, np.number]:
    """
    Compute an aggregate statistic over a local-function map.

    This function performs two sequential operations:

    1. Compute a local-response image using `local_map`
    2. Aggregate the resulting local map into a single scalar

    Parameters
    ----------
    image : ArrayLike
        Input image.

    local_func : Callable[[NDArray], scalar]
        Function applied locally to neighborhoods.

    aggregate_func : Callable[..., scalar]
        Function applied globally to the local-response image.

        Signature:

            aggregate_func(values, **agg_kwargs) -> scalar

    size : int or Sequence[int], optional
        Window size passed to local_map.

    footprint : ArrayLike, optional
        Boolean neighborhood mask passed to local_map.

    axes : Sequence[int], optional
        Axes along which filtering is applied.

    dtype : DTypeLike, optional
        Optional dtype conversion applied to the local map before aggregation.

    agg_kwargs : dict[str, Any], optional
        Additional keyword arguments passed to `aggregate_func`.

        Examples:
        - {"q": 99} for np.percentile
        - {"bias": False} for scipy.stats.skew
        - {"axis": None}

    **kwargs : Any
        Additional keyword arguments forwarded to local_map.

    Returns
    -------
    scalar : int | float | numpy scalar
        Aggregate statistic computed over the local-response image.
    """

    # Initialize aggregate kwargs if not provided.
    if agg_kwargs is None:
        agg_kwargs = {}

    # Compute the local-response image using local_map.
    local_response = local_map(
        image=image,
        func=local_func,
        size=size,
        footprint=footprint,
        axes=axes,
        dtype=dtype,
        **kwargs,
    )

    # Flatten the local map into a 1D vector.
    flattened = np.ravel(local_response)

    # Apply aggregate function with optional kwargs.
    result = aggregate_func(flattened, **agg_kwargs)

    # Return scalar aggregate measurement.
    return result


def aggregate_local_stat_along_axis(
    image: ArrayLike,
    local_func: Callable,
    aggregate_func: Callable,
    axis: Optional[int] = None,
    size: Optional[Union[int, Sequence[int]]] = None,
    footprint: Optional[ArrayLike] = None,
    dtype: Optional[DTypeLike] = None,
    agg_kwargs: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> Union[int, float, NDArray]:
    """
    Axis-aware wrapper around aggregate_local_stat.

    This function simplifies the conceptual model by removing the notion of
    multi-axis filtering. Instead, it introduces a single optional axis
    along which independent computations are performed.

    Conceptual behavior
    -------------------

    CASE 1: axis is None
        - The entire image is treated as a single unit
        - Equivalent to:
              aggregate_local_stat(image, ...)

    CASE 2: axis is an integer
        - The image is split into independent sub-arrays along `axis`
        - aggregate_local_stat is applied independently to each slice
        - Returns a 1D array of results

    Important design simplification
    --------------------------------

    - `axes` is NOT supported anywhere
    - `axes` is also NOT allowed inside kwargs (will be ignored/removed if present)
    - Only a single axis-based decomposition is supported

    Footprint / size behavior
    -------------------------

    - If axis is None:
        * footprint and size are applied globally over full image

    - If axis is int:
        * each slice is processed independently
        * footprint and size are applied within each slice only
        * i.e. local neighborhoods do NOT cross slice boundaries

    Parameters
    ----------
    image : ArrayLike
        Input image (N-D array).

    local_func : Callable
        Local neighborhood function.

    aggregate_func : Callable
        Global aggregation function applied after local mapping.

    axis : int or None, optional
        Axis along which independent computations are performed.

        - None → full image is one unit
        - int  → independent processing per slice

    size : int or sequence, optional
        Local window size passed to local_map via aggregate_local_stat.

    footprint : ArrayLike, optional
        Local neighborhood mask.

    dtype : dtype, optional
        Optional dtype conversion of local map.

    agg_kwargs : dict, optional
        Keyword arguments passed to aggregate_func.

    **kwargs : Any
        Forwarded to aggregate_local_stat.

        NOTE:
        - 'axes' is NOT supported and will be removed if present.

    Returns
    -------
    scalar or ndarray
        - scalar if axis is None
        - 1D ndarray if axis is int

    """

    # initialize aggregate kwargs
    if agg_kwargs is None:
        agg_kwargs = {}

    # explicitly forbid axes in kwargs for conceptual clarity
    kwargs.pop("axes", None)

    # CASE 1: no axis → full-image computation
    if axis is None:
        return aggregate_local_stat(
            image=image,
            local_func=local_func,
            aggregate_func=aggregate_func,
            size=size,
            footprint=footprint,
            dtype=dtype,
            agg_kwargs=agg_kwargs,
            **kwargs,
        )

    # CASE 2: axis-wise computation
    axis = int(axis)

    image = np.asarray(image)

    # move axis to front for clean iteration
    image = np.moveaxis(image, axis, 0)

    results = []

    # iterate over slices along axis
    for i in range(image.shape[0]):
        slice_ = image[i]

        value = aggregate_local_stat(
            image=slice_,
            local_func=local_func,
            aggregate_func=aggregate_func,
            size=size,
            footprint=footprint,
            dtype=dtype,
            agg_kwargs=agg_kwargs,
            **kwargs,
        )

        results.append(value)

    return np.asarray(results)
