import numpy as np
from scipy import stats
from skimage.filters import median


def compute_pixel_stat(image, mask, stat='mean'):
    """
    Compute a statistic over image pixels selected by a mask.

    Parameters
    ----------
    image : np.ndarray
        N-dimensional array of pixel values.
    mask : np.ndarray
        N-dimensional array of the same shape as image. Pixels where
        mask >= 0 are included in the computation.
    stat : str
        Statistic to compute. One of 'mean', 'median', 'mode', 'max', 'min'.
        Default is 'mean'.

    Returns
    -------
    float or int
        The computed statistic over the selected pixels.

    Notes
    -----
    NaN handling: NaN values in the selected pixels are propagated — if any
    selected pixel is NaN, the result will be NaN. This applies to all stats
    except 'mode', where scipy.stats.mode treats NaN as a regular value and
    may return NaN as the mode if it is the most frequent value.
    To ignore NaNs instead, replace np.mean/median/max/min with their
    np.nan* equivalents (e.g. np.nanmean).

    Raises
    ------
    ValueError
        If image and mask shapes differ, stat is invalid, or no pixels are selected.
    """
    if image.shape != mask.shape:
        raise ValueError(
            f"image and mask must have the same shape, "
            f"got {image.shape} and {mask.shape}."
        )

    valid_stats = {'mean', 'median', 'mode', 'max', 'min'}
    if stat not in valid_stats:
        raise ValueError(f"stat must be one of {valid_stats}, got '{stat}'.")

    selected = image[mask >= 0]

    if selected.size == 0:
        raise ValueError("No pixels selected — mask has no values >= 0.")

    if stat == 'mean':
        return np.mean(selected)
    elif stat == 'median':
        return np.median(selected)
    elif stat == 'mode':
        return stats.mode(selected, keepdims=False).mode
    elif stat == 'max':
        return np.max(selected)
    elif stat == 'min':
        return np.min(selected)


def measure_percentile_stat(image, perc, stat='mean', below=True):
    """
    Compute a statistic over image pixels that fall below or above a given percentile.

    Builds on compute_pixel_stat by deriving a mask from the image's own
    histogram distribution at the specified percentile threshold.

    Parameters
    ----------
    image : np.ndarray
        N-dimensional array of pixel values.
    perc : float
        Percentile threshold in the range [0, 100]. The threshold value is
        computed from the image's own distribution using np.nanpercentile,
        so NaNs are excluded when computing the threshold.
    stat : str
        Statistic to compute over the selected pixels. One of 'mean', 'median',
        'mode', 'max', 'min'. Default is 'mean'.
    below : bool
        If True (default), selects pixels whose value is strictly below the
        percentile threshold. If False, selects pixels strictly above the threshold.

    Returns
    -------
    float or int
        The computed statistic over the selected pixels.

    Notes
    -----
    NaN handling: NaN values in the selected pixels are propagated — if any
    selected pixel is NaN, the result will be NaN. This applies to all stats
    except 'mode', where scipy.stats.mode treats NaN as a regular value and
    may return NaN as the mode if it is the most frequent value.
    NaNs are excluded when computing the percentile threshold itself
    (via np.nanpercentile).

    Raises
    ------
    ValueError
        If perc is outside [0, 100], stat is invalid, or no pixels are selected
        after applying the percentile mask.
    """
    if not (0 <= perc <= 100):
        raise ValueError(f"perc must be in [0, 100], got {perc}.")

    threshold = np.nanpercentile(image, perc)

    # Build a mask compatible with compute_pixel_stat:
    # values >= 0 in the mask are included, so we use 0 for selected and -1 for excluded.
    if below:
        mask = np.where(image < threshold, 0, -1)
    else:
        mask = np.where(image > threshold, 0, -1)

    return compute_pixel_stat(image, mask, stat=stat)


def measure_low_high_perc_stat(image, percentiles, stat='mean'):
    """
    Compute a statistic separately over the low and high tails of the image distribution.

    Builds on measure_percentile_stat by applying it to both ends of the
    distribution in a single call.

    Parameters
    ----------
    image : np.ndarray
        N-dimensional array of pixel values.
    percentiles : tuple of float
        A (low_percentile, high_percentile) pair, each in [0, 100].
        low_percentile selects pixels strictly below that percentile threshold;
        high_percentile selects pixels strictly above that percentile threshold.
    stat : str
        Statistic to compute over each tail. One of 'mean', 'median', 'mode',
        'max', 'min'. Default is 'mean'.

    Returns
    -------
    stat_low : float or int
        The computed statistic over pixels strictly below low_percentile.
    stat_high : float or int
        The computed statistic over pixels strictly above high_percentile.

    Notes
    -----
    NaN handling: NaN values in the selected pixels are propagated — if any
    selected pixel is NaN, the result will be NaN. This applies to all stats
    except 'mode', where scipy.stats.mode treats NaN as a regular value and
    may return NaN as the mode if it is the most frequent value.
    NaNs are excluded when computing the percentile thresholds themselves
    (via np.nanpercentile).

    Raises
    ------
    ValueError
        If percentiles does not contain exactly two values, either percentile is
        outside [0, 100], low_percentile >= high_percentile, stat is invalid,
        or no pixels fall within either tail.
    """
    if len(percentiles) != 2:
        raise ValueError(
            f"percentiles must be a tuple of exactly 2 values (low, high), "
            f"got {len(percentiles)}."
        )

    low_perc, high_perc = percentiles

    if not (0 <= low_perc <= 100) or not (0 <= high_perc <= 100):
        raise ValueError(
            f"Both percentiles must be in [0, 100], got {low_perc} and {high_perc}."
        )

    if low_perc >= high_perc:
        raise ValueError(
            f"low_percentile must be strictly less than high_percentile, "
            f"got {low_perc} and {high_perc}."
        )

    stat_low = measure_percentile_stat(image, low_perc, stat=stat, below=True)
    stat_high = measure_percentile_stat(image, high_perc, stat=stat, below=False)

    return stat_low, stat_high


def high_low_percentiles_stat_ch(image, percentiles, stat='mean', smooth=True,
                                  footprint=None, median_kwargs=None):
    """
    Optionally smooth an image before computing low and high percentile statistics.

    Wraps measure_low_high_perc_stat with an optional median smoothing step
    via skimage.filters.median.

    Parameters
    ----------
    image : np.ndarray
        N-dimensional array of pixel values.
    percentiles : tuple of float
        A (low_percentile, high_percentile) pair, each in [0, 100].
        Passed directly to measure_low_high_perc_stat.
    stat : str
        Statistic to compute over each tail. One of 'mean', 'median', 'mode',
        'max', 'min'. Default is 'mean'.
    smooth : bool
        If True (default), applies skimage.filters.median to the image before
        computing statistics. If False, the image is used as-is.
    footprint : array-like or None
        Footprint (structuring element) passed to skimage.filters.median.
        If None, skimage's default footprint is used.
    median_kwargs : dict or None
        Additional keyword arguments passed to skimage.filters.median.
        If None, no extra arguments are passed.

    Returns
    -------
    stat_low : float or int
        The computed statistic over pixels strictly below low_percentile.
    stat_high : float or int
        The computed statistic over pixels strictly above high_percentile.

    Notes
    -----
    NaN handling: NaN values in the selected pixels are propagated through the
    stat computation. NaNs are excluded when computing percentile thresholds.
    Note that skimage.filters.median may not handle NaNs gracefully — if your
    image contains NaNs and smooth=True, consider pre-processing the image
    before calling this function.

    Raises
    ------
    ValueError
        If percentiles are invalid, stat is unrecognised, or no pixels fall
        within either tail. See measure_low_high_perc_stat for full details.
    """
    if not smooth:
        return measure_low_high_perc_stat(image, percentiles, stat=stat)

    kwargs = median_kwargs or {}
    if footprint is not None:
        smoothed = median(image, footprint=footprint, **kwargs)
    else:
        smoothed = median(image, **kwargs)

    return measure_low_high_perc_stat(smoothed, percentiles, stat=stat)


def high_low_percentiles_stat(image, percentiles, stat='mean', smooth=True,
                               footprint=None, median_kwargs=None,
                               channel_axis=None):
    """
    Compute low and high percentile statistics, optionally per channel.

    Wraps high_low_percentiles_stat_ch, applying it either to the full image
    (when channel_axis is None) or independently to each slice along the
    specified channel axis.

    Parameters
    ----------
    image : np.ndarray
        N-dimensional array of pixel values.
    percentiles : tuple of float
        A (low_percentile, high_percentile) pair, each in [0, 100].
        Passed directly to high_low_percentiles_stat_ch.
    stat : str
        Statistic to compute over each tail. One of 'mean', 'median', 'mode',
        'max', 'min'. Default is 'mean'.
    smooth : bool
        If True (default), applies skimage.filters.median to the image before
        computing statistics. If False, the image is used as-is.
    footprint : array-like or None
        Footprint (structuring element) passed to skimage.filters.median.
        If None, skimage's default footprint is used.
    median_kwargs : dict or None
        Additional keyword arguments passed to skimage.filters.median.
        If None, no extra arguments are passed.
    channel_axis : int or None
        If None (default), the statistic is computed over the full image via
        high_low_percentiles_stat_ch.
        If an int, the image is sliced along that axis and
        high_low_percentiles_stat_ch is applied independently to each channel.

    Returns
    -------
    If channel_axis is None:
        stat_low : float or int
        stat_high : float or int

    If channel_axis is int:
        tuple of (stat_low_ch0, stat_low_ch1, ...) : tuple of float or int
        tuple of (stat_high_ch0, stat_high_ch1, ...) : tuple of float or int

    Notes
    -----
    NaN handling: NaN values in the selected pixels are propagated through the
    stat computation. NaNs are excluded when computing percentile thresholds.
    Note that skimage.filters.median may not handle NaNs gracefully — if your
    image contains NaNs and smooth=True, consider pre-processing the image
    before calling this function.

    Raises
    ------
    ValueError
        If percentiles are invalid, stat is unrecognised, channel_axis is out
        of bounds, or no pixels fall within either tail for any channel.
    """
    kwargs = dict(percentiles=percentiles, stat=stat, smooth=smooth,
                  footprint=footprint, median_kwargs=median_kwargs)

    if channel_axis is None:
        return high_low_percentiles_stat_ch(image, **kwargs)

    if not (-image.ndim <= channel_axis < image.ndim):
        raise ValueError(
            f"channel_axis {channel_axis} is out of bounds for image with "
            f"{image.ndim} dimensions."
        )

    channels = np.moveaxis(image, channel_axis, 0)
    results = [high_low_percentiles_stat_ch(ch, **kwargs) for ch in channels]

    stat_lows, stat_highs = zip(*results)
    return tuple(stat_lows), tuple(stat_highs)