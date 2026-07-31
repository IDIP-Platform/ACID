import os
from pathlib import Path
import numpy as np
import skimage.io
import dask.array as da
import dask

# from .feature_extraction.measure_haralick import glcm_feature_map


input_dir_image = r""
input_dir_mask = r""
filenames = os.listdir(input_dir_image)


@dask.delayed
def load(filename, input_dir_image=input_dir_image, input_dir_mask=input_dir_mask):
    """
    needs to return an array. Expects filename to be the same for
    the image and segmentation mask.
    """
    image = skimage.io.imread(os.path.join(Path(input_dir_image), filename))
    mask = skimage.io.imread(os.path.join(Path(input_dir_mask), filename))

    return image, mask


@dask.delayed
def process_image_mask_mbfm(image, mask, input_channel_axis=-1):
    """
    needs to receive an array and return an array.
    Expects the mask to be concatenated to the image data along the channel axis.

    This function is meant to prepare image and mask for the mask-based computation
    of the feature map, by concatenating the mask to the image data along the channel axis.
    """
    mask = np.expand_dims(
        mask, axis=input_channel_axis
    )  # add a channel axis to the mask if it doesn't have one
    data = np.concatenate([image, mask], axis=input_channel_axis)

    # ensure the correct order of the axes (channel axis in the last position)
    if input_channel_axis != -1:
        data = np.moveaxis(data, input_channel_axis, -1)

    return data, mask


@dask.delayed
def compute_mask_based_feature_map(
    data,
    feature_funct,
    window_size=11,
    boundary="reflect",
    chunks=(256, 256, -1),
    kwargs=None,
    dtype=float,
):
    """
    needs to receive an array and return an array.
    Expects channel axis in the last position.
    Expects the mask to be concatenated to the image data as the last sub-stack
    of the channel axis.

    feature_funct must be a top-level function that takes as input an image and a mask, and returns a feature map of the same spatial dimensions as the input image.
    """
    if kwargs is None:
        kwargs = {}

    depth = {
        0: window_size // 2,
        1: window_size // 2,
        2: 0,
    }  # depth for the spatial dimensions

    data_da = da.from_array(
        data, chunks=chunks
    )  # chunks in xy but not on the channel axis

    def _wrapper(data_block, **w_kwargs):
        # data_block is a numpy array corresponding to a chunk of the input data
        # it has the same structure as the input data (channel axis in the last position, mask concatenated to image data)

        img_block = data_block[
            ..., :-1
        ]  # all channels except the last one are image channels
        mask_block = data_block[..., -1]  # the last channel is the mask

        return feature_funct(image=img_block, mask=mask_block, **w_kwargs)

    fm = data_da.map_overlap(
        _wrapper, depth=depth, boundary=boundary, dtype=dtype, trim=True, **kwargs
    )  # trim the overlapping regions after computation to avoid double-counting

    return fm.compute()  # returns a numpy array


@dask.delayed
def measure_feature_map(feature_map, mask):
    # generates measurements from the feature map and the mask
    return  # returns a dataframe of measurements


def f_mbfm(
    filenames, window_size=11, chunks=(256, 256, -1), channel_axis=-1, kwargs=None
):

    if kwargs is None:
        kwargs = {}

    results = []

    for filename in filenames:

        image, mask = load(filename)

        data, mask = process_image_mask_mbfm(
            image, mask, input_channel_axis=channel_axis
        )

        fm = compute_mask_based_feature_map(
            data, window_size=window_size, chunks=chunks, **kwargs
        )

        measurements = measure_feature_map(fm, mask)

        results.append(measurements)

    return results


dask.compute(f_mbfm(filenames))
