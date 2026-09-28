"""
bgcorrection_syn_img_dask.py
------------------------
Dask-backed equivalents of the functions in ``bgcorrection_syn_img.py``.

All functions return ``dask.array.Array`` objects and are fully lazy:
no computation is triggered until the caller explicitly calls ``.compute()``.

The public API mirrors ``bgcorrection_syn_img.py`` exactly; the only
additional parameter is ``chunks``, which controls how each array is
partitioned into Dask chunks.

Typical usage::

    from bgcorrection_syn_img_dask import build_test_suite

    suite = build_test_suite(shape=(1024, 1024), chunks=(512, 512))

    # Nothing has been computed yet.
    observed = suite["linear"]["observed"]

    # Trigger computation when needed:
    result = my_correction(observed).compute()
"""

import numpy as np
import dask.array as da
from dask.array import Array as DaskArray

# Type alias for chunk specifications accepted by dask.array
ChunkSpec = int | tuple[int, int] | str


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def generate_signal(
    shape: tuple[int, int],
    n_spots: int = 20,
    spot_intensity: float = 1.0,
    spot_radius: int = 3,
    seed: int | None = 42,
    chunks: ChunkSpec = (512, 512),
) -> DaskArray:
    """
    Generate a 2-D Dask array containing random bright spots on a zero
    background.

    Each spot is modelled as a filled circle (all pixels within the radius
    are set to ``spot_intensity``). The array is built eagerly in numpy and
    then wrapped as a lazy Dask array; the wrapping itself does not trigger
    further computation.

    Parameters
    ----------
    shape : tuple[int, int]
        Image dimensions as ``(height, width)`` in pixels.
    n_spots : int, optional
        Number of spots to place. Default is 20.
    spot_intensity : float, optional
        Pixel value assigned to every spot pixel. Default is 1.0.
    spot_radius : int, optional
        Radius of each spot in pixels. Default is 3.
    seed : int or None, optional
        Random seed for reproducibility. Pass ``None`` for a random result.
        Default is 42.
    chunks : int, tuple[int, int], or str, optional
        Dask chunk shape for the output array. Accepts any value that
        ``dask.array.from_array`` accepts for its ``chunks`` parameter
        (e.g. ``(512, 512)``, ``256``, or ``"auto"``).
        Default is ``(512, 512)``.

    Returns
    -------
    signal : DaskArray
        Lazy 2-D Dask array of shape ``(height, width)`` with dtype
        ``float64``. Background pixels are 0; spot pixels are
        ``spot_intensity``.
    """
    rng = np.random.default_rng(seed)
    signal_np = np.zeros(shape, dtype=np.float64)

    h, w = shape
    ys, xs = np.ogrid[:h, :w]

    for _ in range(n_spots):
        cy = rng.integers(spot_radius, h - spot_radius)
        cx = rng.integers(spot_radius, w - spot_radius)
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= spot_radius ** 2
        signal_np[mask] = spot_intensity

    return da.from_array(signal_np, chunks=chunks)


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------

def generate_background_linear(
    shape: tuple[int, int],
    slope_y: float = 0.5,
    slope_x: float = 0.3,
    intercept: float = 0.1,
    baseline: float = 2.0,
    chunks: ChunkSpec = (512, 512),
) -> DaskArray:
    """
    Generate a lazy background array with a linear (planar) gradient.

    The background value at pixel ``(r, c)`` is::

        B(r, c) = baseline + intercept + slope_y * r_norm + slope_x * c_norm

    where ``r_norm`` and ``c_norm`` are row and column indices normalised
    to the range ``[0, 1]``. The ``baseline`` offset ensures the background
    is bounded away from zero, preventing division instability in correction
    methods of the form ``(signal - offset) / background``.

    Parameters
    ----------
    shape : tuple[int, int]
        Image dimensions as ``(height, width)`` in pixels.
    slope_y : float, optional
        Rate of intensity change along the vertical axis (normalised).
        Default is 0.5.
    slope_x : float, optional
        Rate of intensity change along the horizontal axis (normalised).
        Default is 0.3.
    intercept : float, optional
        Background value at the top-left corner, before baseline is added.
        Default is 0.1.
    baseline : float, optional
        Constant offset added to the entire background to ensure a safe
        minimum value. Default is 2.0.
    chunks : int, tuple[int, int], or str, optional
        Dask chunk shape for the output array. Default is ``(512, 512)``.

    Returns
    -------
    background : DaskArray
        Lazy 2-D Dask array of shape ``(height, width)`` with dtype
        ``float64``.
    """
    h, w = shape
    r_norm = da.linspace(0, 1, h, chunks=chunks[0] if isinstance(chunks, tuple) else chunks)[:, np.newaxis]
    c_norm = da.linspace(0, 1, w, chunks=chunks[1] if isinstance(chunks, tuple) else chunks)[np.newaxis, :]
    return (baseline + intercept + slope_y * r_norm + slope_x * c_norm).rechunk(chunks)


def generate_background_gaussian(
    shape: tuple[int, int],
    amplitude: float = 1.0,
    center_y: float = 0.5,
    center_x: float = 0.5,
    sigma_y: float = 0.3,
    sigma_x: float = 0.3,
    baseline: float = 2.0,
    chunks: ChunkSpec = (512, 512),
) -> DaskArray:
    """
    Generate a lazy background array shaped as a 2-D Gaussian blob.

    The background value at pixel ``(r, c)`` is::

        B(r, c) = baseline + amplitude * exp(
            -((r_norm - center_y)^2 / (2 * sigma_y^2)
            + (c_norm - center_x)^2 / (2 * sigma_x^2))
        )

    where ``r_norm`` and ``c_norm`` are normalised to ``[0, 1]``. The
    ``baseline`` offset ensures the background is bounded away from zero,
    preventing division instability in correction methods of the form
    ``(signal - offset) / background``.

    Parameters
    ----------
    shape : tuple[int, int]
        Image dimensions as ``(height, width)`` in pixels.
    amplitude : float, optional
        Peak intensity of the Gaussian. Default is 1.0.
    center_y : float, optional
        Vertical position of the peak in normalised coordinates ``[0, 1]``.
        Default is 0.5.
    center_x : float, optional
        Horizontal position of the peak in normalised coordinates ``[0, 1]``.
        Default is 0.5.
    sigma_y : float, optional
        Standard deviation along the vertical axis in normalised units.
        Default is 0.3.
    sigma_x : float, optional
        Standard deviation along the horizontal axis in normalised units.
        Default is 0.3.
    baseline : float, optional
        Constant offset added to the entire background to ensure a safe
        minimum value. Default is 2.0.
    chunks : int, tuple[int, int], or str, optional
        Dask chunk shape for the output array. Default is ``(512, 512)``.

    Returns
    -------
    background : DaskArray
        Lazy 2-D Dask array of shape ``(height, width)`` with dtype
        ``float64``.
    """
    h, w = shape
    r_norm = da.linspace(0, 1, h, chunks=chunks[0] if isinstance(chunks, tuple) else chunks)[:, np.newaxis]
    c_norm = da.linspace(0, 1, w, chunks=chunks[1] if isinstance(chunks, tuple) else chunks)[np.newaxis, :]
    exponent = (
        ((r_norm - center_y) ** 2) / (2 * sigma_y ** 2)
        + ((c_norm - center_x) ** 2) / (2 * sigma_x ** 2)
    )
    return (baseline + amplitude * da.exp(-exponent)).rechunk(chunks)


def generate_background_polynomial(
    shape: tuple[int, int],
    coefficients: dict[tuple[int, int], float] | None = None,
    baseline: float = 2.0,
    chunks: ChunkSpec = (512, 512),
) -> DaskArray:
    """
    Generate a lazy background array as a 2-D polynomial surface.

    The background value at pixel ``(r, c)`` is::

        B(r, c) = baseline + sum_{ (i,j) in coefficients } coeff * r_norm^i * c_norm^j

    where ``r_norm`` and ``c_norm`` are normalised to ``[0, 1]``. The
    ``baseline`` offset ensures the background is bounded away from zero,
    preventing division instability in correction methods of the form
    ``(signal - offset) / background``.

    Parameters
    ----------
    shape : tuple[int, int]
        Image dimensions as ``(height, width)`` in pixels.
    coefficients : dict[tuple[int, int], float] or None, optional
        Mapping from ``(power_y, power_x)`` to coefficient value.
        Defaults to a gentle quadratic surface::

            {(0, 0): 0.1, (1, 0): 0.4, (0, 1): 0.3, (2, 0): 0.2, (0, 2): 0.15}

    baseline : float, optional
        Constant offset added to the entire background to ensure a safe
        minimum value. Default is 2.0.
    chunks : int, tuple[int, int], or str, optional
        Dask chunk shape for the output array. Default is ``(512, 512)``.

    Returns
    -------
    background : DaskArray
        Lazy 2-D Dask array of shape ``(height, width)`` with dtype
        ``float64``.
    """
    if coefficients is None:
        coefficients = {
            (0, 0): 0.1,
            (1, 0): 0.4,
            (0, 1): 0.3,
            (2, 0): 0.2,
            (0, 2): 0.15,
        }

    h, w = shape
    chunk_y = chunks[0] if isinstance(chunks, tuple) else chunks
    chunk_x = chunks[1] if isinstance(chunks, tuple) else chunks
    r_norm = da.linspace(0, 1, h, chunks=chunk_y)[:, np.newaxis]
    c_norm = da.linspace(0, 1, w, chunks=chunk_x)[np.newaxis, :]

    background = da.zeros(shape, dtype=np.float64, chunks=chunks)
    for (pow_y, pow_x), coeff in coefficients.items():
        background = background + coeff * (r_norm ** pow_y) * (c_norm ** pow_x)

    return (baseline + background).rechunk(chunks)


def generate_background_multi_blob(
    shape: tuple[int, int],
    blobs: list[dict] | None = None,
    baseline: float = 2.0,
    chunks: ChunkSpec = (512, 512),
) -> DaskArray:
    """
    Generate a lazy background array as a superposition of multiple 2-D
    Gaussian blobs, simulating uneven illumination with several bright patches.

    The ``baseline`` is applied once to the final superposition, not to each
    individual blob, to avoid accumulating the offset multiple times.

    Parameters
    ----------
    shape : tuple[int, int]
        Image dimensions as ``(height, width)`` in pixels.
    blobs : list[dict] or None, optional
        List of blob parameter dictionaries. Each dictionary is passed as
        keyword arguments to :func:`generate_background_gaussian`, so it
        may contain any subset of ``amplitude``, ``center_y``, ``center_x``,
        ``sigma_y``, ``sigma_x``. The ``baseline`` key is intentionally
        excluded — use the ``baseline`` parameter of this function instead.
        Missing keys use the defaults of :func:`generate_background_gaussian`.

        Defaults to three blobs arranged to create an asymmetric
        illumination pattern::

            [
                {"amplitude": 1.0, "center_y": 0.2, "center_x": 0.3, "sigma_y": 0.2, "sigma_x": 0.25},
                {"amplitude": 0.7, "center_y": 0.7, "center_x": 0.6, "sigma_y": 0.3, "sigma_x": 0.2},
                {"amplitude": 0.5, "center_y": 0.4, "center_x": 0.8, "sigma_y": 0.15, "sigma_x": 0.2},
            ]

    baseline : float, optional
        Constant offset added to the entire background to ensure a safe
        minimum value. Default is 2.0.
    chunks : int, tuple[int, int], or str, optional
        Dask chunk shape for the output array. Default is ``(512, 512)``.

    Returns
    -------
    background : DaskArray
        Lazy 2-D Dask array of shape ``(height, width)`` with dtype
        ``float64``.
    """
    if blobs is None:
        blobs = [
            {"amplitude": 1.0, "center_y": 0.2, "center_x": 0.3, "sigma_y": 0.2, "sigma_x": 0.25},
            {"amplitude": 0.7, "center_y": 0.7, "center_x": 0.6, "sigma_y": 0.3, "sigma_x": 0.2},
            {"amplitude": 0.5, "center_y": 0.4, "center_x": 0.8, "sigma_y": 0.15, "sigma_x": 0.2},
        ]

    background = da.zeros(shape, dtype=np.float64, chunks=chunks)
    for blob_params in blobs:
        background = background + generate_background_gaussian(shape, baseline=0.0, chunks=chunks, **blob_params)

    return baseline + background


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def generate_mask(
    signal: DaskArray,
) -> DaskArray:
    """
    Derive a lazy boolean spot mask from a signal array.

    Spot pixels are defined as pixels with a value greater than zero.
    This convention matches :func:`generate_signal`, where background
    pixels are exactly 0 and spot pixels are ``spot_intensity > 0``.

    Parameters
    ----------
    signal : DaskArray
        Lazy 2-D Dask array of shape ``(height, width)``, as returned by
        :func:`generate_signal`.

    Returns
    -------
    mask : DaskArray
        Lazy boolean Dask array of shape ``(height, width)``. ``True``
        where a pixel belongs to a spot, ``False`` otherwise.
    """
    return signal > 0


# ---------------------------------------------------------------------------
# Test image factory
# ---------------------------------------------------------------------------

def make_test_image(
    background: DaskArray,
    signal: DaskArray,
) -> DaskArray:
    """
    Combine a signal and a background into a single lazy synthetic test image.

    The two arrays must have the same shape. The result is::

        observed = signal + background

    No computation is triggered; the returned array is lazy.

    Parameters
    ----------
    background : DaskArray
        Lazy 2-D Dask array of shape ``(height, width)``.
    signal : DaskArray
        Lazy 2-D Dask array of shape ``(height, width)``.

    Returns
    -------
    observed : DaskArray
        Lazy 2-D Dask array of shape ``(height, width)`` with dtype
        ``float64``.

    Raises
    ------
    ValueError
        If ``signal`` and ``background`` do not have the same shape.
    """
    if signal.shape != background.shape:
        raise ValueError(
            f"signal shape {signal.shape} does not match "
            f"background shape {background.shape}."
        )
    return signal + background


# ---------------------------------------------------------------------------
# Convenience: build the full test suite
# ---------------------------------------------------------------------------

def build_test_suite(
    shape: tuple[int, int] = (256, 256),
    n_spots: int = 20,
    spot_intensity: float = 1.0,
    spot_radius: int = 3,
    seed: int | None = 42,
    baseline: float = 2.0,
    chunks: ChunkSpec = (512, 512),
) -> dict[str, dict[str, DaskArray]]:
    """
    Build the complete set of synthetic test images as lazy Dask arrays.

    A single signal array (random bright spots) is generated once and
    combined with each of the four canonical backgrounds:
    ``linear``, ``gaussian``, ``polynomial``, and ``multi_blob``.

    No computation is triggered until the caller calls ``.compute()``
    on an array.

    Parameters
    ----------
    shape : tuple[int, int], optional
        Image dimensions as ``(height, width)`` in pixels. Default is
        ``(256, 256)``.
    n_spots : int, optional
        Number of bright spots in the signal layer. Default is 20.
    spot_intensity : float, optional
        Pixel intensity of each spot. Default is 1.0.
    spot_radius : int, optional
        Radius of each spot in pixels. Default is 3.
    seed : int or None, optional
        Random seed for the signal generator. Default is 42.
    baseline : float, optional
        Constant offset added to every background to ensure a safe minimum
        value, preventing division instability during correction. The same
        value is passed to all background generators. Default is 2.0.
    chunks : int, tuple[int, int], or str, optional
        Dask chunk shape applied to all generated arrays. Default is
        ``(512, 512)``.

    Returns
    -------
    suite : dict[str, dict[str, DaskArray]]
        A dictionary keyed by background type name. Each value is itself
        a dictionary with the following keys:

        * ``"observed"``   – lazy synthetic image (signal + background)
        * ``"signal"``     – lazy ground-truth signal
        * ``"background"`` – lazy ground-truth background
        * ``"mask"``       – lazy boolean array, ``True`` at spot pixels

        Example access::

            suite["linear"]["observed"].compute()   # trigger computation
            suite["linear"]["signal"]               # still lazy
    """
    signal = generate_signal(
        shape=shape,
        n_spots=n_spots,
        spot_intensity=spot_intensity,
        spot_radius=spot_radius,
        seed=seed,
        chunks=chunks,
    )
    mask = generate_mask(signal)

    backgrounds = {
        "linear":     generate_background_linear(shape, baseline=baseline, chunks=chunks),
        "gaussian":   generate_background_gaussian(shape, baseline=baseline, chunks=chunks),
        "polynomial": generate_background_polynomial(shape, baseline=baseline, chunks=chunks),
        "multi_blob": generate_background_multi_blob(shape, baseline=baseline, chunks=chunks),
    }

    suite = {}
    for name, bg in backgrounds.items():
        suite[name] = {
            "observed":   make_test_image(bg, signal),
            "signal":     signal,
            "background": bg,
            "mask":       mask,
        }

    return suite
