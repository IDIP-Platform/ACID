"""
spot_uniformity.py
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

from __future__ import annotations

from typing import Callable, Any
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    labeled, n_spots = label(mask_np)

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
    assert std_corrected < std_observed, (
        f"Spot uniformity did not improve after correction: "
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

    corrected = correction_fn(observed, background, **correction_kwargs)

    observed_norm = _normalise(observed)
    corrected_norm = _normalise(corrected)

    std_observed = compute_spot_uniformity(observed_norm, mask)
    std_corrected = compute_spot_uniformity(corrected_norm, mask)

    assert_spot_uniformity(std_observed, std_corrected)

    return {
        "std_observed": std_observed,
        "std_corrected": std_corrected,
    }

