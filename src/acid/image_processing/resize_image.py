import numpy as np
from skimage.transform import resize_local_mean


def downsample_local_mean(image:np.array,
                          factor:int|None=None,
                          channel_axis:int|None=None,
                          kwargs:dict|None=None)->np.array:
    # use default empty dict for kwargs if None is passed
    if kwargs is None:
        kwargs = {}
    
    # check that kwargs does not contain keys that are already passed as dedicated arguments
    assert 'channel_axis' not in kwargs, "channel_axis can't be passed to kwargs, please use the dedicated argument"
    assert 'output_shape' not in kwargs, "output_shape can't be passed to kwargs"

    # intialize a list to collect the new shape of the image after resizing
    new_shape = []
    
    # iterate through the image shape
    for p,s in enumerate(image.shape):
        
        # avoid including the channel axis in the new shape if channel_axis is specified
        if isinstance(channel_axis, int) and p==channel_axis:
            continue
        
        # downsize the axes and collect them to the axes to be downsized
        else:
            new_shape.append(s//factor)

    # resize image
    resized_img = resize_local_mean(image, new_shape, channel_axis=channel_axis, **kwargs)
    
    # return dtype_image
    return resized_img



def match_image_shape_2d(image:np.array,
                         target_image:np.array,
                         pad_mode:str='constant',
                         pad_value:int=0,
                         kwargs:dict={})-> np.array:
    """
    Match the size of a 2d image to the size of a second 2d target_image. The match is made either by cropping or
    padding image. In either cases, image is centered on target_image when doing the matching.

    Inputs:
    - image: numpy 2d array, the image to be cropped or padded.
    - target_image: numpy 2d array, the array to match the image shape.
    - pad_mode: str, the mode to use for padding if the mask is smaller than the image.
    - pad_value: int, the value to use for padding if the mask is smaller than the image.

    Outputs: np.array. The cropped or padded image with the same shape as the target_image.
    """
    assert 'pad_mode' not in kwargs, "pad_mode can't be passed to kwargs, please use the dedicated argument"
    assert 'pad_value' not in kwargs, "pad_value can't be passed to kwargs, please use the dedicated argument"

    # Copy the image to avoid modifying the original
    image = np.copy(image)

    # Get the shape of the target_image
    target_image_shape = target_image.shape

    # Get the shape of the image
    image_shape = image.shape

    # Calculate the difference in dimensions
    diff_x = image_shape[0] - target_image_shape[0]
    diff_y = image_shape[1] - target_image_shape[1]

    # If the imge is larger than the target_image, crop it
    if diff_x > 0 or diff_y > 0:
        # Calculate the cropping coordinates
        start_x = diff_x // 2 if diff_x > 0 else 0
        start_y = diff_y // 2 if diff_y > 0 else 0
        end_x = start_x + target_image_shape[0] if diff_x > 0 else target_image_shape[0]
        end_y = start_y + target_image_shape[1] if diff_y > 0 else target_image_shape[1]
        # Crop the image
        image = image[start_x:end_x, start_y:end_y]
    
    # If the image is smaller than the target_image, pad it
    elif diff_x < 0 or diff_y < 0:
        pad_x = abs(diff_x) // 2
        pad_y = abs(diff_y) // 2
        image = np.pad(image, ((pad_x, pad_x), (pad_y, pad_y)), mode=pad_mode, constant_values=pad_value, **kwargs)

    # Return the cropped or padded image
    return image