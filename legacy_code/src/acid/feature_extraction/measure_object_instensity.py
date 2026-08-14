import numpy as np
from scipy.stats import skew, median_abs_deviation

def median_intensity(mask_image:np.array,
                     intensity_image:np.array,
                     **kwargs)-> float:
    """
    Returns the median of the intensity distribution of pixels within a specific region of interest defined
    by image mask.

    When mask_image is a label image, it is possible to pass this function to skimage.measure.regionprops
    extra_properties parameters (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).

    Inputs:
    - mask_image. n-dimensional np.array. Binary boolean mask or label image (if the function should be passed to
    skimage.measure.regionprops as one of the extra_properties). Background values are
    assumed to be 0. NOTE: if a binary mask is passed make sure that it is intepreted by numpy as a boolean mask
    (e.g. change the dytpe to bool or np.uint8) alternatively it could be interpreted as fancy indexing.

    - intensity_image. n-dimensional np.array. If the function is used as it is, intensity_image must have the
    same shape as mask_image. When the function is passed to skimage.measure.regionprops as one of the
    extra_properties it is possible to have extra-channels as long as the channel is on the last axis (position
    -1). In this case the measurement which will returned per each channel.

    - kwargs. Keyword arguments passed to the np.median function.

    Outputs:
    If image_mask is a binary boolean mask, the result is a float value (likely np.float64) corresponding to
    median of the intensity distribution of the pixel values in intensity_image which
    are segmented by the mask_image (which correspond to positive values in mask_image).
    """
    return np.median(intensity_image[mask_image], **kwargs)


def skew_intensity(mask_image:np.array,
                   intensity_image:np.array,
                   **kwargs)-> float:
    """
    Returns the skewness of the intensity distribution of pixels within a specific region of interest defined
    by image mask.

    This function is taken from https://stackoverflow.com/questions/63481913/add-extra-properties-to-regionprops-in-skimage

    When mask_image is a label image, it is possible to pass this function to skimage.measure.regionprops
    extra_properties parameters (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).

    Inputs:
    - mask_image. n-dimensional np.array. Binary boolean mask or label image (if the function should be passed to
    skimage.measure.regionprops as one of the extra_properties). Background values are
    assumed to be 0. NOTE: if a binary mask is passed make sure that it is intepreted by numpy as a boolean mask
    (e.g. change the dytpe to bool or np.uint8) alternatively it could be interpreted as fancy indexing.

    - intensity_image. n-dimensional np.array. If the function is used as it is, intensity_image must have the
    same shape as mask_image. When the function is passed to skimage.measure.regionprops as one of the
    extra_properties it is possible to have extra-channels as long as the channel is on the last axis (position
    -1). In this case the measurement which will returned per each channel.


    - kwargs. Keyword arguments passed to the skew function.

    Outputs:
    If image_mask is a binary boolean mask, the result is a float value (likely np.float64) corresponding to
    skewness of the intensity distribution of the pixel values in intensity_image which
    are segmented by the mask_image (which correspond to positive values in mask_image).
    """

    return skew(intensity_image[mask_image], **kwargs)

def quantile_intensity(mask_image:np.array,
                       intensity_image:np.array,
                       q:float=0.5,
                       **kwargs)-> float:
    """
    Returns the q-th quantile of the intensity distribution of pixels within a specific region of interest defined
    by image mask.

    When mask_image is a label image, it is possible to pass this function to skimage.measure.regionprops
    extra_properties parameters (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).

    Inputs:
    - mask_image. n-dimensional np.array. Binary boolean mask or label image (if the function should be passed to
    skimage.measure.regionprops as one of the extra_properties). Background values are
    assumed to be 0. NOTE: if a binary mask is passed make sure that it is intepreted by numpy as a boolean mask
    (e.g. change the dytpe to bool or np.uint8) alternatively it could be interpreted as fancy indexing.

    - intensity_image. n-dimensional np.array. If the function is used as it is, intensity_image must have the
    same shape as mask_image. When the function is passed to skimage.measure.regionprops as one of the
    extra_properties it is possible to have extra-channels as long as the channel is on the last axis (position
    -1). In this case the measurement which will returned per each channel.

    - q. float. Optional. Default 0.5. The probability of the quantile to compute.
    Values must be between 0 and 1 inclusive.

    - kwargs. Keyword arguments passed to the np.quantile function.

    Outputs:
    If image_mask is a binary boolean mask, the result is a float value (likely np.float64) corresponding to
    q-th quantile of the intensity distribution of the pixel values in intensity_image which
    are segmented by the mask_image (which correspond to positive values in mask_image).
    """
    assert q not in kwargs, "q should not be passed as a keyword argument"
    assert 0 <= q <= 1, "Quantile must be between 0 and 1 inclusive"
    return np.quantile(intensity_image[mask_image], q=q, **kwargs)


def lower_quartile_intensity(mask_image:np.array,
                             intensity_image:np.array,
                             q:float=0.25,
                             **kwargs)-> float:
    """
    Returns the lower quartile of the intensity distribution of pixels within a specific region of interest defined
    by image mask.

    When mask_image is a label image, it is possible to pass this function to skimage.measure.regionprops
    extra_properties parameters (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).

    Inputs:
    - mask_image. n-dimensional np.array. Binary boolean mask or label image (if the function should be passed to
    skimage.measure.regionprops as one of the extra_properties). Background values are
    assumed to be 0. NOTE: if a binary mask is passed make sure that it is intepreted by numpy as a boolean mask
    (e.g. change the dytpe to bool or np.uint8) alternatively it could be interpreted as fancy indexing.

    - intensity_image. n-dimensional np.array. If the function is used as it is, intensity_image must have the
    same shape as mask_image. When the function is passed to skimage.measure.regionprops as one of the
    extra_properties it is possible to have extra-channels as long as the channel is on the last axis (position
    -1). In this case the measurement which will returned per each channel.

    - q. float. Optional. Default 0.25. The probability of the quantile to compute.
    Values must be between 0 and 1 inclusive.

    - kwargs. Keyword arguments passed to the np.quantile function.

    Outputs:
    If image_mask is a binary boolean mask, the result is a float value (likely np.float64) corresponding to
    lower quartile (0.25 quantile) of the intensity distribution of the pixel values in intensity_image which
    are segmented by the mask_image (which correspond to positive values in mask_image).
    """
    assert q not in kwargs, "q should not be passed as a keyword argument"
    assert 0 <= q <= 1, "Quantile must be between 0 and 1 inclusive"
    return np.quantile(intensity_image[mask_image], q=q, **kwargs)


def upper_quartile_intensity(mask_image:np.array,
                              intensity_image:np.array,
                              q:float=0.75,
                              **kwargs) -> float:
    """
    Returns the upper quartile of the intensity distribution of pixels within a specific region of interest defined
    by image mask.

    When mask_image is a label image, it is possible to pass this function to skimage.measure.regionprops
    extra_properties parameters (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).

    Inputs:
    - mask_image. n-dimensional np.array. Binary boolean mask or label image (if the function should be passed to
    skimage.measure.regionprops as one of the extra_properties). Background values are
    assumed to be 0. NOTE: if a binary mask is passed make sure that it is intepreted by numpy as a boolean mask
    (e.g. change the dytpe to bool or np.uint8) alternatively it could be interpreted as fancy indexing.

    - intensity_image. n-dimensional np.array. If the function is used as it is, intensity_image must have the
    same shape as mask_image. When the function is passed to skimage.measure.regionprops as one of the
    extra_properties it is possible to have extra-channels as long as the channel is on the last axis (position
    -1). In this case the measurement which will returned per each channel.

    - q. float. Optional. Default 0.75. The probability of the quantile to compute.
    Values must be between 0 and 1 inclusive.

    - kwargs. Keyword arguments passed to the np.quantile function.

    Outputs:
    If image_mask is a binary boolean mask, the result is a float value (likely np.float64) corresponding to
    upper quartile (0.25 quantile) of the intensity distribution of the pixel values in intensity_image which
    are segmented by the mask_image (which correspond to positive values in mask_image).
    """
    assert 'q' not in kwargs, "q should not be passed as a keyword argument"
    assert 0 <= q <= 1, "Quantile must be between 0 and 1 inclusive"
    return np.quantile(intensity_image[mask_image], q=q, **kwargs)


def mad_intensity(mask_image:np.array,
                   intensity_image:np.array,
                   **kwargs) -> float:
    """
    Returns the median absolute deviation (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_abs_deviation.html)
    of the intensity distribution of pixels within a specific region of interest defined by image mask.

    When mask_image is a label image, it is possible to pass this function to skimage.measure.regionprops
    extra_properties parameters (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).

    Inputs:
    - mask_image. n-dimensional np.array. Binary boolean mask or label image (if the function should be passed to
    skimage.measure.regionprops as one of the extra_properties). Background values are
    assumed to be 0. NOTE: if a binary mask is passed make sure that it is intepreted by numpy as a boolean mask
    (e.g. change the dytpe to bool or np.uint8) alternatively it could be interpreted as fancy indexing.

    - intensity_image. n-dimensional np.array. If the function is used as it is, intensity_image must have the
    same shape as mask_image. When the function is passed to skimage.measure.regionprops as one of the
    extra_properties it is possible to have extra-channels as long as the channel is on the last axis (position
    -1). In this case the measurement which will returned per each channel.

    - kwargs. Keyword arguments passed to the median_abs_deviation function.

    Outputs:
    If image_mask is a binary boolean mask, the result is a float value (likely np.float64) corresponding to
    median_absolute_deviation of the intensity distribution of the pixel values in intensity_image which
    are segmented by the mask_image (which correspond to positive values in mask_image).

    TO DO: add kwargs.
    """
    return median_abs_deviation(intensity_image[mask_image], **kwargs)


def integrated_intensity(mask_image:np.array,
                         intensity_image:np.array,
                         **kwargs) -> float:
    """
    Returns the integrated intensity (sum of all pixel intensity values) of pixels within a specific region of interest
    in intensity_image defined by image mask.

    When mask_image is a label image, it is possible to pass this function to skimage.measure.regionprops
    extra_properties parameters (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).

    Inputs:
    - mask_image. n-dimensional np.array. Binary boolean mask or label image (if the function should be passed to
    skimage.measure.regionprops as one of the extra_properties). Background values are
    assumed to be 0. NOTE: if a binary mask is passed make sure that it is intepreted by numpy as a boolean mask
    (e.g. change the dytpe to bool or np.uint8) alternatively it could be interpreted as fancy indexing.

    - intensity_image. n-dimensional np.array. If the function is used as it is, intensity_image must have the
    same shape as mask_image. When the function is passed to skimage.measure.regionprops as one of the
    extra_properties it is possible to have extra-channels as long as the channel is on the last axis (position
    -1). In this case the measurement which will returned per each channel.

    - kwargs. Keyword arguments passed to the np.sum function.

    Outputs:
    If image_mask is a binary boolean mask, the result is a single value (the dtype depends on the initial
    intensity_image dtype, likely a np.int64) corresponding to the sum of all pixel values in intensity_image
    which are segmented by the mask_image (which correspond to positive values in mask_image).

    TO DO: add kwargs.
    """
    return np.sum(intensity_image[mask_image], **kwargs)
