# from collections.abc import Callable
import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import gaussian

# def mean_filter_image(image:np.array,
#                       function:Callable=np.mean,
#                       size:tuple=(3,3),
#                       axis:int|None=None,
#                       **kwargs):

#     if axis==None:
#         return generic_filter(image, function, size=size, **kwargs)

#     else:
#         i_axes = tuple([a for a in range(len(image.shape)) if a!=axis])
#         print(i_axes)

#         filtered_stack = generic_filter(image, function, size=size, axes=i_axes, **kwargs)
#         return filtered_stack


def median_filter_image(image: np.array, **kwargs) -> np.array:
    if "size" not in kwargs and "footprint" not in kwargs:
        kwargs["size"] = 3
        print(
            "neither size nor footprint specified for median filter, using default size of 3"
        )

    return median_filter(image, **kwargs)


def gaussian_filter_image(
    image: np.array, sigma: float, channel_axis: int | None = None, kwargs: dict = {}
) -> np.array:

    assert (
        "sigma" not in kwargs
    ), "sigma can't be passed to kwargs, use the dedicated parameter instead"
    assert (
        "channel_axis" not in kwargs
    ), "channel_axis can't be passed to kwargs, use the dedicated parameter instead"
    return gaussian(image, sigma=sigma, channel_axis=channel_axis, **kwargs)
