import numpy as np
from skimage.exposure import rescale_intensity


def quantize_image(image:np.ndarray,
                   levels:int=8,
                   pmin:int|float=1,
                   pmax:int|float=99,
                   out_bottom:int=0,
                   out_dtype=np.uint8,
                   percentile_kwargs:dict|None=None,
                   clip_kwargs:dict|None=None,
                   rescale_kwargs:dict|None=None,
                   floor_kwargs:dict|None=None
                   ) -> np.ndarray:

    """Quantizes the image to a specified number of intensity levels,
    after clipping the intensity values to a specified percentile range.

    NOTE: the function is conceptualized to rescale the image in a positive integer range,
    which is the expected input for the GLCM calculation"""

    # Set defaults
    if percentile_kwargs is None:
        percentile_kwargs={}

    if clip_kwargs is None:
        clip_kwargs={}

    if rescale_kwargs is None:
        rescale_kwargs={}

    if floor_kwargs is None:
        floor_kwargs={}

    lo, hi = np.percentile(image, (pmin, pmax), **percentile_kwargs)
    image = np.clip(image, lo, hi, **clip_kwargs)
    image = rescale_intensity(image, in_range=(lo, hi), out_range=(out_bottom, levels-1), **rescale_kwargs)
    return np.floor(image, **floor_kwargs).astype(out_dtype)
