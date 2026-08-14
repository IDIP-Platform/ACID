import os
import numpy as np
import skimage.io
import dask.array as da
import dask
from .feature_extraction.measure_haralick import glcm_feature_map


input_dir_image = ""
input_dir_mask = ""
filenames = os.listdir(input_dir_image)

@dask.delayed
def load(filename,
         input_dir_image=input_dir_image,
         input_dir_mask=input_dir_mask):
    """
    needs to return an array. Expects filename to be the same for
    the image and segmentation mask.
    """
    image = skimage.io.imread(os.path.join(input_dir_image, filename))
    mask = skimage.io.imread(os.path.join(input_dir_mask, filename))

    return image, mask

@dask.delayed
def process_image_mask(image, mask, input_channel_axis=-1):
    """
    needs to receive an array and return an array.
    Expects the mask to be concatenated to the image data along the channel axis.

    This function is meant to prepare image and mask in a standard pipeline (aka
    no mask-based computation of the feature map).
    """
    data = image.copy() # this is a placeholder for any processing that needs to be done
    mask = image.copy() # this is a placeholder for any processing that needs to be done

    return data, mask


@dask.delayed
def compute_feature_map(data,
                        feature_funct,
                        chunks=(256, 256, -1),
                        depth=None,
                        dtype=float,
                        boundary='reflect',
                        kwargs=None):
    """
    needs to receive an array and return an array.
    Expects channel axis in the last position.
    """
    if kwargs is None:
        kwargs = {}

    if depth is None:
        depth = {0: 10, 1: 10, 2: 0}

    data_da = da.from_array(data, chunks=chunks) # chunks in xy but not on the channel axis

    def _wrapper(data_block, **w_kwargs):
        # data_block is a numpy array corresponding to a chunk of the input data
        # it has the same structure as the input data (channel axis in the last position)

        # call the function that computes the feature map for the given
        # data block. Note that the feature map is computed for the etire
        # block and the individual channels. No mask is provided.
        return feature_funct(image=data_block, **w_kwargs)

    fm = data_da.map_overlap(_wrapper,
                             depth=depth,
                             boundary=boundary, # no padding is applied, this is handled internally by the glcm_feature_map function
                             dtype=dtype,
                             trim=True,
                             **kwargs) # trim the overlapping regions after computation to avoid double-counting

    return fm.compute() # returns a numpy array


@dask.delayed
def measure_feature_map(feature_map, mask):
    # generates measurements from the feature map and the mask
    return # returns a dataframe of measurements


def f_mbfm(filenames,
      chunks=(256, 256, -1),
      kwargs=None):

    if kwargs is None:
        kwargs = {}

    results = []

    for filename in filenames:

        image, mask = load(filename)

        data, mask = process_image_mask(image, mask)

        fm = compute_feature_map(data,
                                 chunks=chunks,
                                 **kwargs)

        measurements = measure_feature_map(fm,mask)

        results.append(measurements)

    return results


dask.compute(f_mbfm(filenames))
