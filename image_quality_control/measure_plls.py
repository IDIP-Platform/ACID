import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.stats import linregress


# ---------------------------------------------------------------------
# Utility: Determine which plots user wants (Option C API)
# ---------------------------------------------------------------------
def _normalize_plot_arg(plot):
    if plot is False or plot is None:
        return set()
    if plot == "all":
        return {"fft", "radial", "loglog"}
    if isinstance(plot, str):
        return {plot}
    # assume iterable
    return set(plot)


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------
def compute_plls(image, axis=None, mask_zero=True, verbose=0,
                 plot=False, plot_slices="first"):
    """
    Compute the Power Log-Log Slope (PLLS) for image sharpness assessment,
    with optional debugging plots.

    Parameters
    ----------
    image : ndarray
        The input image. Must be 2D if axis=None. If axis is given, PLLS is
        computed for each 2D slice along that axis.
    axis : int or None
        Axis along which to compute PLLS (per-slice processing).
    mask_zero : bool
        Whether to remove zero power bins before regression.
    verbose : int
        Verbosity level 0–3.
    plot : False | str | iterable | "all"
        Option C plotting system:
            False   -> no plots
            "all"   -> all plots
            "fft"   -> only FFT magnitude
            etc.
            ["fft", "loglog"] -> multiple specific plots
    plot_slices : "first" | "all" | list of indices
        When axis is given, choose which slices to plot.

    Returns
    -------
    slopes : float or ndarray
    intercepts : float or ndarray
    freqs_out : ndarray or list
    powers_out : ndarray or list
    """

    # Interpret plotting selection
    plots = _normalize_plot_arg(plot)

    # ===============================
    # Case A: No axis → single 2D image
    # ===============================
    if axis is None:

        if verbose >= 1:
            print("[PLLS] Computing PLLS for single 2D image.")

        if image.ndim != 2:
            raise ValueError("If axis is None, image must be 2D.")

        slope, intercept, freqs, power = _plls_2d(
            image, mask_zero, verbose, plots, title="Image"
        )

        return slope, intercept, freqs, power

    # ===============================
    # Case B: axis specified → per-slice processing
    # ===============================

    if verbose >= 1:
        print(f"[PLLS] Computing PLLS per slice along axis={axis}")
        print(f"[PLLS] Input shape = {image.shape}")

    axis = int(axis)
    num_slices = image.shape[axis]

    slopes = []
    intercepts = []
    freqs_out = []
    powers_out = []

    # Determine which slices to plot
    if plot_slices == "first":
        plot_indices = {0}
    elif plot_slices == "all":
        plot_indices = set(range(num_slices))
    elif isinstance(plot_slices, (tuple, list, set)):
        plot_indices = set(plot_slices)
    else:
        raise ValueError("Invalid value for plot_slices")

    # Loop over slices
    for i in range(num_slices):

        if verbose >= 2:
            print(f"[PLLS] Processing slice {i}/{num_slices-1}")

        sl = np.take(image, i, axis=axis)

        if sl.ndim != 2:
            raise ValueError(
                f"Slice along axis {axis} is not 2D; got shape {sl.shape}"
            )

        # Decide which plots to display for this slice
        active_plots = plots if i in plot_indices else set()

        slope, intercept, freqs, power = _plls_2d(
            sl, mask_zero, verbose, active_plots, title=f"Slice {i}"
        )

        slopes.append(slope)
        intercepts.append(intercept)
        freqs_out.append(freqs)
        powers_out.append(power)

    return np.array(slopes), np.array(intercepts), freqs_out, powers_out


# ---------------------------------------------------------------------
# Internal: Compute PLLS for a single 2D image + optional plots
# ---------------------------------------------------------------------
def _plls_2d(image, mask_zero, verbose, plots, title=""):
    """
    Compute PLLS for one 2D image.
    """

    # ----- FFT -----
    fft_img = fftshift(fft2(image))
    power = np.abs(fft_img)**2

    # ----- FFT plot -----
    if "fft" in plots:
        plt.figure(figsize=(5, 4))
        plt.imshow(np.log10(1 + np.abs(fft_img)), cmap="magma")
        plt.title(f"FFT magnitude (log) - {title}")
        plt.colorbar()
        plt.show()

    # ----- Build radial distances -----
    h, w = image.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Flatten
    r_flat = r.astype(int).ravel()
    p_flat = power.ravel()

    # Radial average
    radial_sum = np.bincount(r_flat, weights=p_flat)
    radial_count = np.bincount(r_flat)
    radial_power = radial_sum / np.maximum(radial_count, 1)

    freqs = np.arange(len(radial_power))

    # Mask zeros
    valid_mask = (radial_power > 0) if mask_zero else np.ones_like(radial_power, bool)
    freqs = freqs[valid_mask]
    radial_power = radial_power[valid_mask]

    # ----- Radial plot -----
    if "radial" in plots:
        plt.figure(figsize=(5, 4))
        plt.plot(freqs, radial_power)
        plt.title(f"Radial Power Spectrum - {title}")
        plt.xlabel("Frequency (radius)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.show()

    # ----- Regression -----
    log_freqs = np.log(freqs)
    log_power = np.log(radial_power)

    slope, intercept, _, _, _ = linregress(log_freqs, log_power)

    if verbose >= 2:
        print(f"[PLLS-2D] slope = {slope:.4f}, intercept = {intercept:.4f}")

    # ----- Log-log regression plot -----
    if "loglog" in plots:
        plt.figure(figsize=(5, 4))
        plt.scatter(log_freqs, log_power, s=10, label="data")
        plt.plot(
            log_freqs,
            intercept + slope * log_freqs,
            label=f"fit (slope={slope:.3f})",
        )
        plt.title(f"Log–Log Spectrum + Regression - {title}")
        plt.xlabel("log(freq)")
        plt.ylabel("log(power)")
        plt.grid(True)
        plt.legend()
        plt.show()

    return slope, intercept, freqs, radial_power
