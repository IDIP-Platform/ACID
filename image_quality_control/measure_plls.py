import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.stats import linregress
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True

    # -------------------------------------------------------------------
    # NUMBA-ACCELERATED RADIAL BINNING
    # -------------------------------------------------------------------
    @njit(parallel=True, cache=True)
    def radial_binning_numba(power, r_int, max_r):
        """
        Fast radial binning using Numba parallel loops.
        
        Parameters
        ----------
        power : 2D array (float32 or float64)
            Power spectrum (|FFT|^2).
        r_int : 2D int array
            Precomputed integer radii for each pixel.
        max_r : int
            Maximum radius + 1.
        
        Returns
        -------
        radial_sum : 1D array
        radial_count : 1D array
        """

        h, w = power.shape
        radial_sum = np.zeros(max_r, dtype=np.float64)
        radial_count = np.zeros(max_r, dtype=np.int64)

        # Parallel raster scan
        for y in prange(h):
            for x in range(w):
                r = r_int[y, x]
                radial_sum[r] += power[y, x]
                radial_count[r] += 1

        return radial_sum, radial_count

except ImportError:
    NUMBA_AVAILABLE = False

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
                 plot=False, plot_slices="first", return_details=False):
    """
    Compute the Power Log-Log Slope (PLLS) for image sharpness assessment,
    with optional debugging plots.

    Returns the power-law log-log slope (PLLS) of the image's Fourier spectrum.

    Conceptually:

    Sharp, in-focus images have more high-frequency content, so the slope is more negative.

    Blurry or out-of-focus images have suppressed high frequencies, so the slope is closer to zero.

    This means one can use PLLS as a focus score: lower slope → better focus, higher slope → blurry.

    
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
        
        if return_details:
            return np.array(slope), np.array(intercept), freqs, power
        
        else:
            return np.array(slope)

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

    if return_details:
        return np.array(slopes), np.array(intercepts), freqs_out, powers_out
        
    else:
        return np.array(slopes)
        


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

    # ----- Build radial distances (integer radii) -----
    h, w = image.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_int = r.astype(np.int32)
    max_r = r_int.max() + 1

    # ----- Radial binning (Numba or fallback) -----
    if NUMBA_AVAILABLE:
        if verbose >= 2:
            print("[PLLS-2D] Using Numba-accelerated radial binning.")
        radial_sum, radial_count = radial_binning_numba(power, r_int, max_r)
    else:
        if verbose >= 2:
            print("[PLLS-2D] Using NumPy fallback radial binning.")
        radial_sum = np.bincount(r_int.ravel(), weights=power.ravel())
        radial_count = np.bincount(r_int.ravel())

    radial_power = radial_sum / np.maximum(radial_count, 1)
    freqs = np.arange(len(radial_power))

    # Mask zeros in power (optional)
    valid_mask = (radial_power > 0) if mask_zero else np.ones_like(radial_power, bool)

    # Exclude zero frequency (DC component)
    valid_mask &= (freqs > 0)

    # Apply mask
    freqs_fit = freqs[valid_mask]
    radial_power_fit = radial_power[valid_mask]

    # ----- Radial plot -----
    if "radial" in plots:
        plt.figure(figsize=(5, 4))
        plt.plot(freqs_fit, radial_power_fit)
        plt.title(f"Radial Power Spectrum - {title}")
        plt.xlabel("Frequency (radius)")
        plt.ylabel("Power")
        plt.grid(True)
        plt.show()

    # ----- Regression -----
    log_freqs = np.log(freqs_fit)
    log_power = np.log(radial_power_fit)

    # ----- Regression -----
    # don't calculate the linear regression is the sample is too small
    if len(log_freqs) < 2:
        slope, intercept = np.nan, np.nan
    else:
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

    return slope, intercept, freqs_fit, radial_power_fit
