"""
bgcorrection_syn_img.py
-------------------
Functions to generate synthetic microscopy images for testing
background correction algorithms.

Each test image is composed of:
    observed = signal + background

where the signal is a set of random bright spots and the background
is a known, deterministic surface. No noise is added, so that
correction functions can be tested on their pure technical implementation.
"""

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def generate_signal(
    shape: tuple[int, int],
    n_spots: int = 20,
    spot_intensity: float = 1.0,
    spot_radius: int = 3,
    seed: int | None = 42,
) -> NDArray[np.float64]:
    """
    Generate a 2-D image containing random bright spots on a zero background.

    Each spot is modelled as a filled circle (all pixels inside the radius
    are set to ``spot_intensity``).

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

    Returns
    -------
    signal : NDArray[np.float64]
        2-D array of shape ``(height, width)`` with dtype ``float64``.
        Background pixels are 0; spot pixels are ``spot_intensity``.
    """
    rng = np.random.default_rng(seed)
    signal = np.zeros(shape, dtype=np.float64)

    h, w = shape
    ys, xs = np.ogrid[:h, :w]

    for _ in range(n_spots):
        cy = rng.integers(spot_radius, h - spot_radius)
        cx = rng.integers(spot_radius, w - spot_radius)
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= spot_radius ** 2
        signal[mask] = spot_intensity

    return signal


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------

def generate_background_linear(
    shape: tuple[int, int],
    slope_y: float = 0.5,
    slope_x: float = 0.3,
    intercept: float = 0.1,
    baseline: float = 2.0,
) -> NDArray[np.float64]:
    """
    Generate a background with a linear (planar) gradient.

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

    Returns
    -------
    background : NDArray[np.float64]
        2-D array of shape ``(height, width)`` with dtype ``float64``.
    """
    h, w = shape
    r_norm = np.linspace(0, 1, h)[:, np.newaxis]
    c_norm = np.linspace(0, 1, w)[np.newaxis, :]
    return baseline + intercept + slope_y * r_norm + slope_x * c_norm


def generate_background_gaussian(
    shape: tuple[int, int],
    amplitude: float = 1.0,
    center_y: float = 0.5,
    center_x: float = 0.5,
    sigma_y: float = 0.3,
    sigma_x: float = 0.3,
    baseline: float = 2.0,
) -> NDArray[np.float64]:
    """
    Generate a background shaped as a 2-D Gaussian blob.

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
        Default is 0.5 (image centre).
    center_x : float, optional
        Horizontal position of the peak in normalised coordinates ``[0, 1]``.
        Default is 0.5 (image centre).
    sigma_y : float, optional
        Standard deviation along the vertical axis in normalised units.
        Default is 0.3.
    sigma_x : float, optional
        Standard deviation along the horizontal axis in normalised units.
        Default is 0.3.
    baseline : float, optional
        Constant offset added to the entire background to ensure a safe
        minimum value. Default is 2.0.

    Returns
    -------
    background : NDArray[np.float64]
        2-D array of shape ``(height, width)`` with dtype ``float64``.
    """
    h, w = shape
    r_norm = np.linspace(0, 1, h)[:, np.newaxis]
    c_norm = np.linspace(0, 1, w)[np.newaxis, :]
    exponent = (
        ((r_norm - center_y) ** 2) / (2 * sigma_y ** 2)
        + ((c_norm - center_x) ** 2) / (2 * sigma_x ** 2)
    )
    return baseline + amplitude * np.exp(-exponent)


def generate_background_polynomial(
    shape: tuple[int, int],
    coefficients: dict[tuple[int, int], float] | None = None,
    baseline: float = 2.0,
) -> NDArray[np.float64]:
    """
    Generate a background as a 2-D polynomial surface.

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

    Returns
    -------
    background : NDArray[np.float64]
        2-D array of shape ``(height, width)`` with dtype ``float64``.
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
    r_norm = np.linspace(0, 1, h)[:, np.newaxis]
    c_norm = np.linspace(0, 1, w)[np.newaxis, :]

    background = np.zeros(shape, dtype=np.float64)
    for (pow_y, pow_x), coeff in coefficients.items():
        background += coeff * (r_norm ** pow_y) * (c_norm ** pow_x)

    return baseline + background


def generate_background_multi_blob(
    shape: tuple[int, int],
    blobs: list[dict] | None = None,
    baseline: float = 2.0,
) -> NDArray[np.float64]:
    """
    Generate a background as a superposition of multiple 2-D Gaussian blobs,
    simulating uneven illumination with several bright patches.

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

    Returns
    -------
    background : NDArray[np.float64]
        2-D array of shape ``(height, width)`` with dtype ``float64``.
    """
    if blobs is None:
        blobs = [
            {"amplitude": 1.0, "center_y": 0.2, "center_x": 0.3, "sigma_y": 0.2, "sigma_x": 0.25},
            {"amplitude": 0.7, "center_y": 0.7, "center_x": 0.6, "sigma_y": 0.3, "sigma_x": 0.2},
            {"amplitude": 0.5, "center_y": 0.4, "center_x": 0.8, "sigma_y": 0.15, "sigma_x": 0.2},
        ]

    background = np.zeros(shape, dtype=np.float64)
    for blob_params in blobs:
        background += generate_background_gaussian(shape, baseline=0.0, **blob_params)

    return baseline + background


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def generate_mask(
    signal: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """
    Derive a boolean spot mask from a signal array.

    Spot pixels are defined as pixels with a value greater than zero.
    This convention matches :func:`generate_signal`, where background
    pixels are exactly 0 and spot pixels are ``spot_intensity > 0``.

    Parameters
    ----------
    signal : NDArray[np.float64]
        2-D signal array of shape ``(height, width)``, as returned by
        :func:`generate_signal`.

    Returns
    -------
    mask : NDArray[np.bool_]
        2-D boolean array of shape ``(height, width)``. ``True`` where
        a pixel belongs to a spot, ``False`` otherwise.
    """
    return signal > 0


# ---------------------------------------------------------------------------
# Test image factory
# ---------------------------------------------------------------------------

def make_test_image(
    background: NDArray[np.float64],
    signal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Combine a signal and a background into a single synthetic test image.

    The two arrays must have the same shape. The result is::

        observed = signal + background

    Parameters
    ----------
    background : NDArray[np.float64]
        2-D background array of shape ``(height, width)``.
    signal : NDArray[np.float64]
        2-D signal array of shape ``(height, width)``.

    Returns
    -------
    observed : NDArray[np.float64]
        2-D array of shape ``(height, width)`` representing the synthetic
        microscopy image.

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
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """
    Build the complete set of synthetic test images.

    A single signal array (random bright spots) is generated once and
    combined with each of the four canonical backgrounds:
    ``linear``, ``gaussian``, ``polynomial``, and ``multi_blob``.

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

    Returns
    -------
    suite : dict[str, dict[str, NDArray[np.float64]]]
        A dictionary keyed by background type name. Each value is itself
        a dictionary with the following keys:

        * ``"observed"``   – the synthetic image (signal + background)
        * ``"signal"``     – the ground-truth signal
        * ``"background"`` – the ground-truth background
        * ``"mask"``       – boolean array, ``True`` at spot pixels

        Example access::

            suite["linear"]["observed"]   # the test image
            suite["linear"]["signal"]     # ground truth signal
            suite["linear"]["background"] # ground truth background
            suite["linear"]["mask"]       # spot pixel mask
    """
    signal = generate_signal(
        shape=shape,
        n_spots=n_spots,
        spot_intensity=spot_intensity,
        spot_radius=spot_radius,
        seed=seed,
    )
    mask = generate_mask(signal)

    backgrounds = {
        "linear":     generate_background_linear(shape, baseline=baseline),
        "gaussian":   generate_background_gaussian(shape, baseline=baseline),
        "polynomial": generate_background_polynomial(shape, baseline=baseline),
        "multi_blob": generate_background_multi_blob(shape, baseline=baseline),
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
