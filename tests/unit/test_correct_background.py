from __future__ import annotations
import pytest
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label


"""
-------------------
Functions to generate synthetic microscopy images for testing
background correction algorithms.

Each test image is composed of:
    observed = signal + background

where the signal is a set of random bright spots and the background
is a known, deterministic surface. No noise is added, so that
correction functions can be tested on their pure technical implementation.
"""

def _validate_shape(
    shape: tuple[int, int],
    name: str = "shape",
) -> None:
    """
    Validate a 2-D image shape.
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError(f"{name} must be a tuple of length 2, got {shape!r}.")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"{name} must contain positive integers, got {shape!r}.")


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
    _validate_shape(shape, "shape")

    if n_spots < 0:
        raise ValueError(f"n_spots must be >= 0, got {n_spots!r}.")
    if spot_intensity <= 0:
        raise ValueError(f"spot_intensity must be > 0, got {spot_intensity!r}.")
    if spot_radius <= 0:
        raise ValueError(f"spot_radius must be > 0, got {spot_radius!r}.")

    h, w = shape
    if h <= 2 * spot_radius or w <= 2 * spot_radius:
        raise ValueError(
            "shape is too small to place spots with the requested radius: "
            f"shape={shape}, spot_radius={spot_radius}."
        )

    rng = np.random.default_rng(seed)
    signal = np.zeros(shape, dtype=np.float64)

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
    """
    _validate_shape(shape, "shape")

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
    """
    _validate_shape(shape, "shape")

    if sigma_y <= 0 or sigma_x <= 0:
        raise ValueError(
            "sigma_y and sigma_x must be > 0, "
            f"got sigma_y={sigma_y!r}, sigma_x={sigma_x!r}."
        )
    if not 0.0 <= center_y <= 1.0:
        raise ValueError(f"center_y must be in [0, 1], got {center_y!r}.")
    if not 0.0 <= center_x <= 1.0:
        raise ValueError(f"center_x must be in [0, 1], got {center_x!r}.")

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
    """
    _validate_shape(shape, "shape")

    if coefficients is None:
        coefficients = {
            (0, 0): 0.1,
            (1, 0): 0.4,
            (0, 1): 0.3,
            (2, 0): 0.2,
            (0, 2): 0.15,
        }

    for key, coeff in coefficients.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                f"Coefficient keys must be (power_y, power_x), got {key!r}."
            )
        if not all(isinstance(val, int) and val >= 0 for val in key):
            raise ValueError(
                f"Coefficient powers must be non-negative ints, got {key!r}."
            )
        if not isinstance(coeff, (int, float)):
            raise TypeError(f"Coefficient values must be numeric, got {coeff!r}.")

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
    """
    _validate_shape(shape, "shape")

    if blobs is None:
        blobs = [
            {
                "amplitude": 1.0,
                "center_y": 0.2,
                "center_x": 0.3,
                "sigma_y": 0.2,
                "sigma_x": 0.25,
            },
            {
                "amplitude": 0.7,
                "center_y": 0.7,
                "center_x": 0.6,
                "sigma_y": 0.3,
                "sigma_x": 0.2,
            },
            {
                "amplitude": 0.5,
                "center_y": 0.4,
                "center_x": 0.8,
                "sigma_y": 0.15,
                "sigma_x": 0.2,
            },
        ]

    if not isinstance(blobs, list):
        raise TypeError(f"blobs must be a list[dict] or None, got {type(blobs).__name__}.")

    background = np.zeros(shape, dtype=np.float64)
    for blob_params in blobs:
        if not isinstance(blob_params, dict):
            raise TypeError(
                f"Each blob entry must be a dict, got {type(blob_params).__name__}."
            )
        background += generate_background_gaussian(
            shape,
            baseline=0.0,
            **blob_params,
        )

    return baseline + background


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def generate_mask(
    signal: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """
    Derive a boolean spot mask from a signal array.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError(
            "signal must be a NumPy array, got "
            f"{type(signal).__name__}."
        )
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
    """
    if not isinstance(background, np.ndarray):
        raise TypeError(
            f"background must be a NumPy array, got {type(background).__name__}."
        )
    if not isinstance(signal, np.ndarray):
        raise TypeError(
            f"signal must be a NumPy array, got {type(signal).__name__}."
        )

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
    """
    _validate_shape(shape, "shape")

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

    suite: dict[str, dict[str, NDArray[np.float64]]] = {}
    for name, bg in backgrounds.items():
        suite[name] = {
            "observed":   make_test_image(bg, signal),
            "signal":     signal,
            "background": bg,
            "mask":       mask,
        }

    return suite


"""
------------------------
Dask-backed equivalents of the numpy-based functions above.

Functions to generate synthetic microscopy images for testing
background correction algorithms.

Each test image is composed of:
    observed = signal + background

where the signal is a set of random bright spots and the background
is a known, deterministic surface. No noise is added, so that
correction functions can be tested on their pure technical implementation.

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


def _normalise_chunks(
    chunks: ChunkSpec,
    shape: tuple[int, int],
) -> tuple[ChunkSpec, ChunkSpec]:
    """
    Normalise a chunk specification to a (row_chunks, col_chunks) pair.

    This keeps the Dask code consistent across int, tuple, and string chunk
    specifications and makes pytest inputs easier to validate and reason about.
    """
    if isinstance(chunks, tuple):
        if len(chunks) != 2:
            raise ValueError(
                f"chunks tuple must have len 2, got {chunks!r} for shape={shape}."
            )
        row_chunks, col_chunks = chunks
        if not isinstance(row_chunks, (int, str)) or not isinstance(col_chunks, (int, str)):
            raise TypeError(
                "chunks tuple entries must be int or str, "
                f"got {type(row_chunks).__name__} and {type(col_chunks).__name__}."
            )
        return row_chunks, col_chunks

    if isinstance(chunks, (int, str)):
        return chunks, chunks

    raise TypeError(
        f"Unsupported chunk specification {chunks!r}. "
        "Expected int, tuple[int, int], or str."
    )


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------

def generate_signal_dask(
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
    """
    _validate_shape(shape, "shape")

    if n_spots < 0:
        raise ValueError(f"n_spots must be >= 0, got {n_spots!r}.")
    if spot_intensity <= 0:
        raise ValueError(f"spot_intensity must be > 0, got {spot_intensity!r}.")
    if spot_radius <= 0:
        raise ValueError(f"spot_radius must be > 0, got {spot_radius!r}.")

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

def generate_background_linear_dask(
    shape: tuple[int, int],
    slope_y: float = 0.5,
    slope_x: float = 0.3,
    intercept: float = 0.1,
    baseline: float = 2.0,
    chunks: ChunkSpec = (512, 512),
) -> DaskArray:
    """
    Generate a lazy background array with a linear (planar) gradient.
    """
    _validate_shape(shape, "shape")

    chunk_y, chunk_x = _normalise_chunks(chunks, shape)
    h, w = shape

    r_norm = da.linspace(0, 1, h, chunks=chunk_y)[:, np.newaxis]
    c_norm = da.linspace(0, 1, w, chunks=chunk_x)[np.newaxis, :]

    return (baseline + intercept + slope_y * r_norm + slope_x * c_norm).rechunk(chunks)


def generate_background_gaussian_dask(
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
    """
    _validate_shape(shape, "shape")

    if sigma_y <= 0 or sigma_x <= 0:
        raise ValueError(
            f"sigma_y and sigma_x must be > 0, got sigma_y={sigma_y!r}, sigma_x={sigma_x!r}."
        )

    chunk_y, chunk_x = _normalise_chunks(chunks, shape)
    h, w = shape

    r_norm = da.linspace(0, 1, h, chunks=chunk_y)[:, np.newaxis]
    c_norm = da.linspace(0, 1, w, chunks=chunk_x)[np.newaxis, :]

    exponent = (
        ((r_norm - center_y) ** 2) / (2 * sigma_y ** 2)
        + ((c_norm - center_x) ** 2) / (2 * sigma_x ** 2)
    )
    return (baseline + amplitude * da.exp(-exponent)).rechunk(chunks)


def generate_background_polynomial_dask(
    shape: tuple[int, int],
    coefficients: dict[tuple[int, int], float] | None = None,
    baseline: float = 2.0,
    chunks: ChunkSpec = (512, 512),
) -> DaskArray:
    """
    Generate a lazy background array as a 2-D polynomial surface.
    """
    _validate_shape(shape, "shape")

    if coefficients is None:
        coefficients = {
            (0, 0): 0.1,
            (1, 0): 0.4,
            (0, 1): 0.3,
            (2, 0): 0.2,
            (0, 2): 0.15,
        }

    for key, coeff in coefficients.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Coefficient keys must be (power_y, power_x), got {key!r}.")
        if not all(isinstance(val, int) and val >= 0 for val in key):
            raise ValueError(f"Coefficient powers must be non-negative ints, got {key!r}.")
        if not isinstance(coeff, (int, float)):
            raise TypeError(f"Coefficient values must be numeric, got {coeff!r}.")

    chunk_y, chunk_x = _normalise_chunks(chunks, shape)
    h, w = shape
    r_norm = da.linspace(0, 1, h, chunks=chunk_y)[:, np.newaxis]
    c_norm = da.linspace(0, 1, w, chunks=chunk_x)[np.newaxis, :]

    background = da.zeros(shape, dtype=np.float64, chunks=chunks)
    for (pow_y, pow_x), coeff in coefficients.items():
        background = background + coeff * (r_norm ** pow_y) * (c_norm ** pow_x)

    return (baseline + background).rechunk(chunks)


def generate_background_multi_blob_dask(
    shape: tuple[int, int],
    blobs: list[dict] | None = None,
    baseline: float = 2.0,
    chunks: ChunkSpec = (512, 512),
) -> DaskArray:
    """
    Generate a lazy background array as a superposition of multiple 2-D
    Gaussian blobs, simulating uneven illumination with several bright patches.
    """
    _validate_shape(shape, "shape")

    if blobs is None:
        blobs = [
            {"amplitude": 1.0, "center_y": 0.2, "center_x": 0.3, "sigma_y": 0.2, "sigma_x": 0.25},
            {"amplitude": 0.7, "center_y": 0.7, "center_x": 0.6, "sigma_y": 0.3, "sigma_x": 0.2},
            {"amplitude": 0.5, "center_y": 0.4, "center_x": 0.8, "sigma_y": 0.15, "sigma_x": 0.2},
        ]

    if not isinstance(blobs, list):
        raise TypeError(f"blobs must be a list[dict] or None, got {type(blobs).__name__}.")

    background = da.zeros(shape, dtype=np.float64, chunks=chunks)
    for blob_params in blobs:
        if not isinstance(blob_params, dict):
            raise TypeError(f"Each blob entry must be a dict, got {type(blob_params).__name__}.")
        background = background + generate_background_gaussian_dask(
            shape,
            baseline=0.0,
            chunks=chunks,
            **blob_params,
        )

    return baseline + background


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def generate_mask_dask(
    signal: DaskArray,
) -> DaskArray:
    """
    Derive a lazy boolean spot mask from a signal array.
    """
    if not hasattr(signal, "shape"):
        raise TypeError("signal must be a Dask array-like object with a shape attribute.")
    return signal > 0


# ---------------------------------------------------------------------------
# Test image factory
# ---------------------------------------------------------------------------

def make_test_image_dask(
    background: DaskArray,
    signal: DaskArray,
) -> DaskArray:
    """
    Combine a signal and a background into a single lazy synthetic test image.
    """
    if signal.shape != background.shape:
        raise ValueError(
            f"signal shape {signal.shape} does not match background shape {background.shape}."
        )
    return signal + background


# ---------------------------------------------------------------------------
# Convenience: build the full test suite
# ---------------------------------------------------------------------------

def build_test_suite_dask(
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
    """
    _validate_shape(shape, "shape")

    signal = generate_signal_dask(
        shape=shape,
        n_spots=n_spots,
        spot_intensity=spot_intensity,
        spot_radius=spot_radius,
        seed=seed,
        chunks=chunks,
    )
    mask = generate_mask_dask(signal)

    backgrounds = {
        "linear":     generate_background_linear_dask(shape, baseline=baseline, chunks=chunks),
        "gaussian":   generate_background_gaussian_dask(shape, baseline=baseline, chunks=chunks),
        "polynomial": generate_background_polynomial_dask(shape, baseline=baseline, chunks=chunks),
        "multi_blob": generate_background_multi_blob_dask(shape, baseline=baseline, chunks=chunks),
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


"""
------------------
Spot uniformity metric for evaluating microscopy background correction.

The metric measures whether the mean intensity of individual spots becomes
more consistent (lower standard deviation) after background correction.
A good correction should equalize spot brightness across the image,
regardless of where each spot sits relative to the illumination gradient.

The module provides three public functions:

* :func:`compute_spot_uniformity` — labels individual spots in the mask,
  computes the mean intensity of each spot in an image, and returns the
  standard deviation of those per-spot means.
* :func:`assert_spot_uniformity` — asserts that the std after correction
  is lower than the std before correction.
* :func:`evaluate_spot_uniformity` — applies a correction function
  internally and orchestrates the two functions above.

All functions accept both ``numpy.ndarray`` and ``dask.array.Array``
inputs. Dask arrays are materialised with ``.compute()`` before being
passed to ``scipy.ndimage.label``, which requires concrete numpy arrays.
"""


def _to_numpy(array: Any) -> NDArray:
    """
    Convert an array to a concrete numpy array, triggering Dask computation
    if needed.

    Parameters
    ----------
    array : NDArray or DaskArray
        Input array.

    Returns
    -------
    array_np : NDArray
        Concrete numpy array.
    """
    if hasattr(array, "compute"):
        return array.compute()
    return np.asarray(array)


def _normalise(image: Any) -> NDArray[np.float64]:
    """
    Normalise an image to the ``[0, 1]`` range using min-max normalisation.

    The transformation is::

        normalised = (image - min) / (max - min)

    This maps both the observed and corrected images to exactly ``[0, 1]``
    regardless of their original offset or scale, ensuring a fair comparison
    in :func:`evaluate_spot_uniformity`.

    Parameters
    ----------
    image : NDArray[np.float64] or DaskArray
        2-D image of shape ``(height, width)``.

    Returns
    -------
    normalised : NDArray[np.float64]
        2-D array of shape ``(height, width)`` with values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``max - min`` is zero, making normalisation undefined (i.e. the
        image is completely flat).
    """
    image_np = _to_numpy(image)
    min_val = image_np.min()
    max_val = image_np.max()
    denom = max_val - min_val

    if denom == 0:
        raise ValueError(
            "Cannot normalise a flat image whose max and min are equal."
        )

    return (image_np - min_val) / denom


def _validate_same_shape(
    left: Any,
    right: Any,
    left_name: str,
    right_name: str,
) -> None:
    """
    Validate that two arrays share the same shape.
    """
    left_np = _to_numpy(left)
    right_np = _to_numpy(right)

    if left_np.shape != right_np.shape:
        raise ValueError(
            f"{left_name} and {right_name} must have the same shape: "
            f"{left_np.shape} != {right_np.shape}"
        )


def compute_spot_uniformity(
    image: Any,
    mask: Any,
) -> float:
    """
    Compute the standard deviation of per-spot mean intensities.

    Individual spots are identified as connected components in ``mask``
    using :func:`scipy.ndimage.label`. For each spot the mean pixel
    intensity in ``image`` is computed. The returned value is the standard
    deviation of those per-spot means — a lower value indicates more
    uniform spot brightness across the image.

    Parameters
    ----------
    image : NDArray[np.float64] or DaskArray
        2-D image of shape ``(height, width)`` to evaluate. This is
        typically either the observed image (before correction) or the
        corrected image (after correction).
    mask : NDArray[np.bool_] or DaskArray
        Boolean array of shape ``(height, width)``. ``True`` at spot
        pixels, as returned by ``generate_mask()`` from
        ``bgcorrection_syn_img``.

    Returns
    -------
    std_spot_means : float
        Standard deviation of the per-spot mean intensities. Returns
        ``0.0`` if fewer than two spots are found, since std is
        undefined for a single value.
    """
    image_np = _to_numpy(image)
    mask_np = _to_numpy(mask)

    if image_np.shape != mask_np.shape:
        raise ValueError(
            "image and mask must have the same shape: "
            f"{image_np.shape} != {mask_np.shape}"
        )

    mask_bool = mask_np.astype(bool)
    labeled, n_spots = label(mask_bool)

    if n_spots < 2:
        return 0.0

    spot_means = [
        image_np[labeled == spot_id].mean()
        for spot_id in range(1, n_spots + 1)
    ]

    return float(np.std(spot_means))


def assert_spot_uniformity(
    std_observed: float,
    std_corrected: float,
) -> None:
    """
    Assert that spot intensity uniformity improves after correction.

    The assertion passes when the standard deviation of per-spot means
    in the corrected image is strictly lower than in the observed image,
    indicating that the correction has equalised spot brightness.

    Parameters
    ----------
    std_observed : float
        Standard deviation of per-spot mean intensities in the observed
        (uncorrected) image.
    std_corrected : float
        Standard deviation of per-spot mean intensities in the corrected
        image.

    Raises
    ------
    AssertionError
        If ``std_corrected >= std_observed``, with a message reporting
        both values.
    """
    if not np.isfinite(std_observed) or not np.isfinite(std_corrected):
        raise ValueError(
            "Spot uniformity values must be finite numbers. "
            f"Got std_observed={std_observed!r}, std_corrected={std_corrected!r}."
        )

    if std_corrected >= std_observed:
        raise AssertionError(
            "Spot uniformity did not improve after correction: "
            f"std_corrected={std_corrected:.6g} >= std_observed={std_observed:.6g}."
        )


def evaluate_spot_uniformity(
    correction_fn: Callable,
    observed: Any,
    background: Any,
    mask: Any,
    correction_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Apply a correction function and evaluate whether spot uniformity improves.

    This function combines :func:`compute_spot_uniformity` and
    :func:`assert_spot_uniformity`. Both the observed and corrected images
    are normalised to ``[0, 1]`` via :func:`_normalise` before computing
    the std of per-spot means, ensuring the comparison is independent of
    the absolute intensity scale of each image. The function asserts that
    the corrected std is lower and returns both values for inspection.

    Parameters
    ----------
    correction_fn : Callable
        Background correction function with the signature::

            corrected = correction_fn(observed, background, **correction_kwargs)

    observed : NDArray[np.float64] or DaskArray
        Synthetic observed image of shape ``(height, width)``
        (i.e. ``signal + background``).
    background : NDArray[np.float64] or DaskArray
        Ground-truth background image of shape ``(height, width)``.
    mask : NDArray[np.bool_] or DaskArray
        Boolean spot mask of shape ``(height, width)``, as returned by
        ``generate_mask()`` from ``bgcorrection_syn_img``.
    correction_kwargs : dict[str, Any] or None, optional
        Keyword arguments forwarded to ``correction_fn``. Default is ``{}``.

    Returns
    -------
    results : dict[str, float]
        Dictionary with the following keys:

        * ``"std_observed"``  – std of per-spot means before correction
        * ``"std_corrected"`` – std of per-spot means after correction

    Raises
    ------
    AssertionError
        If spot uniformity does not improve after correction.
    """
    if correction_kwargs is None:
        correction_kwargs = {}

    _validate_same_shape(observed, background, "observed", "background")
    _validate_same_shape(observed, mask, "observed", "mask")

    try:
        corrected = correction_fn(observed, background, **correction_kwargs)
    except TypeError as exc:
        raise TypeError(
            "correction_fn must accept "
            "(observed, background, **correction_kwargs)"
        ) from exc

    corrected_np = _to_numpy(corrected)
    observed_np = _to_numpy(observed)

    if corrected_np.shape != observed_np.shape:
        raise ValueError(
            "correction_fn returned an array with an unexpected shape: "
            f"{corrected_np.shape} != {observed_np.shape}"
        )

    observed_norm = _normalise(observed)
    corrected_norm = _normalise(corrected)

    std_observed = compute_spot_uniformity(observed_norm, mask)
    std_corrected = compute_spot_uniformity(corrected_norm, mask)

    assert_spot_uniformity(std_observed, std_corrected)

    return {
        "std_observed": std_observed,
        "std_corrected": std_corrected,
    }


"""
-------------
Validation utility for microscopy background correction functions.

The module provides a single public function :func:`validate_correction`
which runs :func:`evaluate_spot_uniformity` from ``spot_uniformity.py``
over the full synthetic test suite and reports all failures at the end.

The script is compatible with suites generated by both ``bgcorrection_syn_img``
(numpy) and ``bgcorrection_syn_img_dask`` (Dask), since :func:`evaluate_spot_uniformity`
handles both array types transparently.
"""

def _validate_suite_structure(suite: dict[str, dict[str, Any]]) -> None:
    """
    Validate that a synthetic suite has the expected nested structure.
    """
    if not isinstance(suite, dict):
        raise TypeError(f"suite must be a dict, got {type(suite).__name__}.")

    for bg_name, images in suite.items():
        if not isinstance(images, dict):
            raise TypeError(
                f"Suite entry for '{bg_name}' must be a dict, "
                f"got {type(images).__name__}."
            )
        required_keys = {"observed", "background", "mask"}
        missing = required_keys - images.keys()
        if missing:
            raise ValueError(
                f"Suite entry for '{bg_name}' is missing required keys: {sorted(missing)}"
            )


def validate_correction(
    correction_fn: Callable,
    suite: dict[str, dict[str, Any]] | None = None,
    correction_kwargs: dict[str, Any] | None = None,
    suite_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Validate a correction function against the full synthetic test suite
    using the spot uniformity metric.

    For each background type in the suite, :func:`evaluate_spot_uniformity`
    is called to check whether the std of per-spot mean intensities decreases
    after correction. Failures are collected across all background types and
    reported together at the end via a single ``AssertionError``.

    Parameters
    ----------
    correction_fn : Callable
        Background correction function with the signature::

            corrected = correction_fn(observed, background, **correction_kwargs)

    suite : dict[str, dict[str, Any]] or None, optional
        Pre-built test suite as returned by ``build_test_suite()`` from
        either ``bgcorrection_syn_img`` or ``bgcorrection_syn_img_dask``. Must
        contain ``"observed"``, ``"background"``, and ``"mask"`` keys for
        each background type. When ``None``, the suite is generated
        automatically using
        ``bgcorrection_syn_img.build_test_suite(**suite_kwargs)``.
    correction_kwargs : dict[str, Any] or None, optional
        Keyword arguments forwarded to ``correction_fn``. Default is ``{}``.
    suite_kwargs : dict[str, Any] or None, optional
        Keyword arguments forwarded to ``build_test_suite()`` when ``suite``
        is ``None``. Default is ``{}``.

    Returns
    -------
    all_results : dict[str, dict[str, float]]
        Nested mapping ``{background_name: {"std_observed": float, "std_corrected": float}}``.
        Returned even when some assertions fail, so the caller can inspect
        numeric results for all background types.

    Raises
    ------
    AssertionError
        If spot uniformity does not improve for one or more background types,
        with a consolidated report of all failures.
    """
    if not callable(correction_fn):
        raise TypeError(f"correction_fn must be callable, got {type(correction_fn).__name__}.")

    if correction_kwargs is None:
        correction_kwargs = {}
    if not isinstance(correction_kwargs, dict):
        raise TypeError(
            f"correction_kwargs must be a dict or None, got {type(correction_kwargs).__name__}."
        )

    if suite_kwargs is None:
        suite_kwargs = {}
    if not isinstance(suite_kwargs, dict):
        raise TypeError(
            f"suite_kwargs must be a dict or None, got {type(suite_kwargs).__name__}."
        )

    if suite is None:
        from acid.image_processing.bgcorrection_syn_img import build_test_suite

        suite = build_test_suite(**suite_kwargs)

    _validate_suite_structure(suite)

    all_results: dict[str, dict[str, float]] = {}
    failures: list[str] = []

    for bg_name, images in suite.items():
        try:
            results = evaluate_spot_uniformity(
                correction_fn=correction_fn,
                observed=images["observed"],
                background=images["background"],
                mask=images["mask"],
                correction_kwargs=correction_kwargs,
            )
            all_results[bg_name] = results
        except Exception as exc:  # noqa: BLE001
            all_results[bg_name] = {
                "std_observed": float("nan"),
                "std_corrected": float("nan"),
            }
            failures.append(f"  [{bg_name}] {type(exc).__name__}: {exc}")

    if failures:
        failure_report = "\n".join(failures)
        raise AssertionError(
            f"Validation failed for {len(failures)} background type(s):\n{failure_report}"
        )

    return all_results



def test_correct_background_nd() -> None:
    from acid.image_processing.correct_background import correct_background_nd
    results = validate_correction(
        correct_background_nd,
        correction_kwargs={
            "method": "division",
            "rescale_background": "max",
        },
    )

    assert results, "No validation results were returned."

    for bg_name, metrics in results.items():
        assert "std_observed" in metrics, f"{bg_name} missing std_observed"
        assert "std_corrected" in metrics, f"{bg_name} missing std_corrected"

        std_observed = metrics["std_observed"]
        std_corrected = metrics["std_corrected"]

        assert std_corrected < std_observed, (
            f"{bg_name}: background correction did not improve uniformity: "
            f"std_observed={std_observed}, std_corrected={std_corrected}"
        )
