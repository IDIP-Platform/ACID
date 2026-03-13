import numpy as np
from skimage.restoration import rolling_ball
from skimage.util import invert
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d
from typing import Optional, Dict, Union, Any


def compute_simple_background_2d(
    image: np.ndarray,
    ball_radius: int = 50,
    white_background: bool = False,
    _rb_kwargs: Optional[Dict[str, Any]] = None,
    invert_kwargs: Optional[Dict[str, Any]] = None,
    gau_smooth: Optional[Union[int, np.ndarray]] = None,
    gaussian_kwargs: Optional[Dict[str, Any]] = None,
    convolve_kwargs: Optional[Dict[str, Any]] = None,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Estimate the background of a 2D microscopy image using the rolling-ball algorithm,
    optionally applying Gaussian or kernel-based smoothing. Useful for illumination
    correction and preprocessing before quantitative analysis.

    This function only supports 2D arrays. For 3D microscopy stacks, process each
    slice independently.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image array. This is the raw microscopy image from which the
        background will be estimated.
    ball_radius : int, optional
        Radius of the rolling ball in pixels used for background estimation.
        Larger values remove larger structures.
    white_background : bool, optional
        If True, the image is assumed to have dark features on a bright background.
        Otherwise, the default is bright features on a dark background.
    _rb_kwargs : dict, optional
        Extra keyword arguments passed directly to skimage.restoration.rolling_ball.
        Do not include 'radius' here; it is controlled separately.
    invert_kwargs : dict, optional
        Extra keyword arguments passed to skimage.util.invert, used when
        white_background=True.
    gau_smooth : int, ndarray, or None, optional
        Optional smoothing applied to the estimated background:
        - None: no smoothing
        - int: Gaussian smoothing applied with sigma = gau_smooth
        - ndarray: user-provided 2D kernel for convolution
    gaussian_kwargs : dict, optional
        Additional keyword arguments passed to scipy.ndimage.gaussian_filter.
        The 'sigma' parameter is not allowed here (controlled by gau_smooth).
    convolve_kwargs : dict, optional
        Additional keyword arguments passed to scipy.signal.convolve2d.
        Defaults include mode='same' and boundary='symm'.
    dtype : np.dtype, optional
        If specified, the output background array is converted to this data type
        before being returned. If None, the original data type is preserved.

    Returns
    -------
    background : np.ndarray
        2D array of the estimated background, optionally smoothed and converted
        to the specified dtype.

    Edge Cases
    ----------
    - If the input image is empty, the function returns an empty array.
    - If ball_radius is larger than the image dimensions, the function will
      return a nearly flat background equal to the average image intensity.
    - If gau_smooth is zero or an empty kernel, no smoothing is applied.
    - Input arrays with uniform intensity are returned unchanged.
    - The function handles different image types (uint8, uint16, float) correctly
      due to skimage.util.invert usage.
    """

    # Initialize dictionaries if None were provided
    if _rb_kwargs is None:
        _rb_kwargs = {}

    if invert_kwargs is None:
        invert_kwargs = {}

    if gaussian_kwargs is None:
        gaussian_kwargs = {}

    if convolve_kwargs is None:
        convolve_kwargs = {}

    # Prevent accidental overriding of core parameters
    assert "radius" not in _rb_kwargs, "Do not pass 'radius' in _rb_kwargs; use ball_radius."
    assert "sigma" not in gaussian_kwargs, "Do not pass 'sigma' in gaussian_kwargs; use gau_smooth."

    # Set default convolution behavior if not overridden
    convolve_defaults = {"mode": "same", "boundary": "symm"}
    convolve_kwargs = {**convolve_defaults, **convolve_kwargs}

    # Make a copy of the input image to avoid modifying the original
    img = np.copy(image)

    # If the background is white, invert the image before processing
    if white_background:
        img = invert(img, **invert_kwargs)
        background = rolling_ball(img, radius=ball_radius, **_rb_kwargs)
        background = invert(background, **invert_kwargs)
    else:
        # Otherwise, directly compute the rolling-ball background
        background = rolling_ball(img, radius=ball_radius, **_rb_kwargs)

    # Apply optional smoothing if requested
    if gau_smooth is not None:
        if isinstance(gau_smooth, int):
            # Apply Gaussian smoothing with sigma equal to the provided integer
            background = gaussian_filter(background, sigma=gau_smooth, **gaussian_kwargs)
        else:
            # Apply convolution with a user-provided kernel
            kernel = np.asarray(gau_smooth)
            background = convolve2d(background, kernel, **convolve_kwargs)

    # Convert background to specified dtype if requested
    if dtype is not None:
        background = background.astype(dtype)

    # Return the final estimated background
    return background


def compute_simple_background_2d_stack(
    image: np.ndarray,
    ball_radius: int = 50,
    white_background: bool = False,
    _rb_kwargs: Optional[Dict[str, Any]] = None,
    invert_kwargs: Optional[Dict[str, Any]] = None,
    gau_smooth: Optional[Union[int, np.ndarray]] = None,
    gaussian_kwargs: Optional[Dict[str, Any]] = None,
    convolve_kwargs: Optional[Dict[str, Any]] = None,
    dtype: Optional[np.dtype] = None,
    axis: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the background of a 2D or 3D microscopy image using the rolling-ball
    algorithm, optionally applying Gaussian or kernel-based smoothing.

    This function wraps simple_background_computation. It can process single 2D
    images or stacks of 2D images along a specified axis.

    Parameters
    ----------
    image : np.ndarray
        Input image array. Can be 2D or 3D depending on axis parameter.
    ball_radius : int, optional
        Radius of the rolling ball in pixels used for background estimation.
    white_background : bool, optional
        If True, the image is assumed to have dark features on a bright background.
    _rb_kwargs : dict, optional
        Extra keyword arguments passed to simple_background_computation.
    invert_kwargs : dict, optional
        Extra keyword arguments passed to simple_background_computation.
    gau_smooth : int, ndarray, or None, optional
        Optional smoothing applied to the estimated background.
    gaussian_kwargs : dict, optional
        Extra parameters for Gaussian smoothing.
    convolve_kwargs : dict, optional
        Extra parameters for kernel convolution smoothing.
    dtype : np.dtype, optional
        Data type to convert the output background(s) to.
    axis : int, optional
        Axis along which to compute background for a stack of 2D images.
        If None, image must be 2D.

    Returns
    -------
    np.ndarray
        Background image if 2D, or stack of background images if 3D.

    Raises
    ------
    ValueError
        If axis=None and image is not 2D.
        If axis is specified and image is not 3D.
        If axis is out of bounds for the input array.
    """

    # Check axis and input dimensions
    if axis is None:
        # Expect a 2D image
        if image.ndim > 2:
            raise ValueError("axis=None requires a 2D input image.")
        # Compute background using compute_simple_background_2d
        from __main__ import compute_simple_background_2d  # Avoid circular import if in the same module
        return compute_simple_background_2d(
            image=image,
            ball_radius=ball_radius,
            white_background=white_background,
            _rb_kwargs=_rb_kwargs,
            invert_kwargs=invert_kwargs,
            gau_smooth=gau_smooth,
            gaussian_kwargs=gaussian_kwargs,
            convolve_kwargs=convolve_kwargs,
            dtype=dtype,
        )
    else:
        # Expect a 3D image
        if image.ndim != 3:
            raise ValueError("axis specified requires a 3D input image.")
        if axis < 0 or axis >= image.ndim:
            raise ValueError(f"axis {axis} is out of bounds for image with shape {image.shape}.")

        # Prepare output array of the same shape as input
        output_shape = list(image.shape)
        backgrounds = np.empty_like(image, dtype=dtype if dtype is not None else image.dtype)

        # Iterate over the slices along the specified axis
        for idx in range(image.shape[axis]):
            # Extract the 2D slice
            if axis == 0:
                slice_2d = image[idx, :, :]
            elif axis == 1:
                slice_2d = image[:, idx, :]
            else:
                slice_2d = image[:, :, idx]

            # Compute background for this slice
            bg_slice = compute_simple_background_2d(
                image=slice_2d,
                ball_radius=ball_radius,
                white_background=white_background,
                _rb_kwargs=_rb_kwargs,
                invert_kwargs=invert_kwargs,
                gau_smooth=gau_smooth,
                gaussian_kwargs=gaussian_kwargs,
                convolve_kwargs=convolve_kwargs,
                dtype=dtype,
            )

            # Place the result in the output array
            if axis == 0:
                backgrounds[idx, :, :] = bg_slice
            elif axis == 1:
                backgrounds[:, idx, :] = bg_slice
            else:
                backgrounds[:, :, idx] = bg_slice

        return backgrounds



def compute_simple_background_nd(
    image: np.ndarray,
    ball_radius: int = 50,
    white_background: bool = False,
    _rb_kwargs: Optional[Dict[str, Any]] = None,
    invert_kwargs: Optional[Dict[str, Any]] = None,
    gau_smooth: Optional[Union[int, np.ndarray]] = None,
    gaussian_kwargs: Optional[Dict[str, Any]] = None,
    convolve_kwargs: Optional[Dict[str, Any]] = None,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Estimate the background of an n-dimensional microscopy image using the rolling-ball algorithm,
    optionally applying Gaussian or kernel-based smoothing. Useful for illumination
    correction and preprocessing before quantitative analysis.

    Parameters
    ----------
    image : np.ndarray
        Input n-dimensional image array.
    ball_radius : int, optional
        Radius of the rolling ball in pixels.
    white_background : bool, optional
        If True, assume dark features on bright background.
    _rb_kwargs : dict, optional
        Extra kwargs for rolling_ball (except 'radius').
    invert_kwargs : dict, optional
        Extra kwargs for invert.
    gau_smooth : int, ndarray, or None, optional
        - int: Gaussian smoothing sigma
        - ndarray: custom kernel for convolution
    gaussian_kwargs : dict, optional
        Extra kwargs for gaussian_filter (sigma is controlled by gau_smooth).
    convolve_kwargs : dict, optional
        Extra kwargs for scipy.ndimage.convolve.
    dtype : np.dtype, optional
        Desired output dtype.

    Returns
    -------
    background : np.ndarray
        Estimated background.
    """

    # Initialize kwargs if None
    _rb_kwargs = _rb_kwargs or {}
    invert_kwargs = invert_kwargs or {}
    gaussian_kwargs = gaussian_kwargs or {}
    convolve_kwargs = convolve_kwargs or {}

    # Prevent overriding key params
    assert "radius" not in _rb_kwargs, "Do not pass 'radius' in _rb_kwargs; use ball_radius."
    assert "sigma" not in gaussian_kwargs, "Do not pass 'sigma' in gaussian_kwargs; use gau_smooth."

    # Copy input image
    img = np.copy(image)

    # Invert if white background
    if white_background:
        img = invert(img, **invert_kwargs)
        background = rolling_ball(img, radius=ball_radius, **_rb_kwargs)
        background = invert(background, **invert_kwargs)
    else:
        background = rolling_ball(img, radius=ball_radius, **_rb_kwargs)

    # Apply optional smoothing
    if gau_smooth is not None:
        if isinstance(gau_smooth, int):
            # Gaussian filter supports n-dimensional data
            background = gaussian_filter(background, sigma=gau_smooth, **gaussian_kwargs)
        else:
            # Use n-dimensional convolution
            kernel = np.asarray(gau_smooth)
            background = convolve(background, kernel, **convolve_kwargs)

    # Convert dtype if requested
    if dtype is not None:
        background = background.astype(dtype)

    return background


