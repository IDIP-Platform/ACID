import numpy as np
from typing import Optional

def correct_background_nd(
    image:np.ndarray,
    background:np.ndarray,
    method:Optional[str]="subtraction",
    offset:Optional[float]=0,
    epsilon:Optional[float]=1e-8,
    working_dtype:Optional[type]=float,
    output_dtype:Optional[type]=None,
    rescale_background:Optional[str]=None,
    clip_corrected_image:Optional[bool]=False,
    min_clip_value:Optional[float]=0,
    max_clip_value:Optional[float]=None,
    offset_background:Optional[bool]=False
)->np.ndarray:
    """
    Apply background correction to a microscopy image.

    NOTE: This function does not apply any type of denoising to the input image or background. As noise can be
    amplified by background correction, it is recommended to input denoised background (and potentially
    denoised image) to this function for optimal results.   

    Parameters    ----------
    - image : np.ndarray. The input image to be corrected. The image can't contain NaN or Inf values.
    
    - background : np.ndarray. The background image to be subtracted or divided from the input image. Must be
    the same shape as the input image. The background can't contain NaN or Inf values.
    
    - method : str, optional. Default is 'subtraction'. The method of background correction to apply. Must be either
    'subtraction' or 'division'.
    
    - offset : int, optional. Default is 0. A constant value to SUBTRACT from the image before applying
    the correction.
    
    - epsilon : float, optional. Default is 1e-8. Must be positive. A small constant added to the denominator
    in division-based correction to prevent division by zero.
    
    - working_dtype : type, optional. The data type to use for intermediate calculations.
    Default is float (aka float64) which allows for decimal values and prevents overflow during subtraction
    or division. You can set this to a specific type like np.float32 or np.float64 if desired. However mind that
    float64 is HIGHLY RECOMMENDED for division-based background correction.
    
    - output_dtype : type, optional. Default is None. The data type to convert the corrected image to before returning. If None,
    the corrected image will be returned in the same dtype as the working_dtype.
    
    - rescale_background : str, optional. If specified, the background
    image will be rescaled before applying the correction. Possible options are:
        - None: No rescaling will be applied to the background. This is the default behavior.
        - 'max': Rescale the background so that its maximum value is 1. This is done by dividing
        the background by its maximum value. This is the recommended option for division-based background
        correction to ensure that the background values are in a suitable range for division. Note that if the
        correction method is 'division' and rescale_background is not set to 'max', a warning will be printed
        but the process will still proceed.
        - 'minmax': Rescale the background to the range [0, 1] by subtracting the minimum value and
        dividing by the range (max - min).
    
    - clip_corrected_image : bool, optional. Default is False. If True, any values in the corrected
    image will be clipped to the range [min_clip_value, max_clip_value]. This can be useful to prevent negative
    and extreme pixel values in the output image, which may not be meaningful in the context of microscopy images.
    
    - min_clip_value : float, optional. Default is 0. The minimum value to which pixel
    values in the corrected image will be clipped if clip_corrected_image is True.
    
    - max_clip_value : float, optional. Default is None. The maximum value to which pixel
    values in the corrected image will be clipped if clip_corrected_image is True.

    - offset_background : bool, optional. Default False. If True the offset is subtracted also to the background
    function. Note that if background is rescaled the application of the offset happens before the rescaling.

    Outputs    -------
    - corrected_image : np.ndarray. The background-corrected image, in the same shape as the input
    image and in the dtype specified by output_dtype (or working_dtype if output_dtype is None).  
    
    """
    # Check that image and background have the same shape
    if image.shape != background.shape:
        raise ValueError("Image and background must have the same shape.")

    # Ensure that image and background do not contain NaN or Inf values
    if not np.isfinite(image).all() or not np.isfinite(background).all():
        raise ValueError("Input contains NaN or Inf values.")

    # warn against the use of interger dtypes as outputs
    if np.issubdtype(output_dtype, np.integer):
        print("Warning: casting to integer may lose precision.")

    # Convert to working dtype
    image = image.astype(working_dtype)
    background = background.astype(working_dtype)

    # ensure that epsilon is positive
    if epsilon <= 0:
        raise ValueError("epsilon must be positive for numerical stability.")

    # warn against the risk of using unsigned integer working types when doing subtraction-based
    # background correction
    if method=="subtraction" and np.issubdtype(working_dtype, np.unsignedinteger):
            print("warning. Using an unsigned integer working type with subtraction method can lead to mathematical" \
            "instability because unsigned integers can't accomodate negative numbers")

    # warn against the risk of using integer output types when doing division-based
    # background correction
    if method=="division" and np.issubdtype(working_dtype, np.integer):
        print("warning. Using interger output with division method leads to a loss of numerical precision"
        "as integers can't accomodated decimals")

    # warn against the risk of using unsigned integer output types when doing subtraction-based
    # background correction
    if method=="subtraction" and np.issubdtype(output_dtype, np.unsignedinteger) and clip_corrected_image==False:
            print("warning. Using an unsigned integer output with subtraction method can lead to overflow because"
            "unsigned integers can't accomodate negative numbers. Ensure accurate clipping is done or change the"
            "output data type")

    # offset background if required
    if offset_background:
        
        # ensure that background offsetting does not lead to negative numbers
        if offset>0:
            min_val_background = np.min(background)
            if offset > min_val_background:
                raise ValueError(f"If offset_background is True, offset must be less than the minimum pixel value in the background ({min_val_background}) to avoid negative values.")

        background = background - offset

    # Optional background rescaling
    if rescale_background is None:
        pass

    elif rescale_background == "max":
        max_val = np.max(background)
        background = background / (max_val + epsilon)

    elif rescale_background == "minmax":
        min_val = np.min(background)
        max_val = np.max(background)

        # check that the image is not constant
        if max_val == min_val:
            raise ValueError("Cannot minmax scale a constant background.")
        
        background = (background - min_val) / (max_val - min_val + epsilon)

    else:
        raise ValueError("rescale_background must be None, 'max', or 'minmax'.")

    # ensure that when offset is applied before division, it does not lead to negative values in the
    # numerator which would cause issues in division-based correction
    if offset > 0 and method == "division":
        min_val = np.min(image)
        if offset >= min_val:
            raise ValueError(f"Offset must be less than the minimum pixel value in the image ({min_val}) to avoid negative values in the division.")

    # Apply correction
    if method == "subtraction":

        # warn against the risk of using unsigned integers as working types when subtraction method is used
        if np.issubdtype(working_dtype, np.unsignedinteger):
            print("warning. Using unsigned interger as working dtype with subtraction method can lead to" \
            "problematic output as unsigned integers can't accomodate negative numbers")

        corrected_image = image - background - offset

    elif method == "division":

        # warn against the risk of using integer data types when doing division-based correction
        if np.issubdtype(working_dtype, np.integer):
            print("warning. Using integer as working dtype when doing division-based correction leads to a loss of" \
            "precision")

        # If the method is division and the background is not rescaled to a maximum of 1, print a
        # warning that this may lead to suboptimal results due to potential issues with scaling and
        # numerical stability.
        if rescale_background != "max":
            print("Warning: division-based background correction typically requires the background to be rescaled to a maximum of 1. Consider setting rescale_background='max' for optimal results.")
        
        corrected_image = (image - offset) / (background + epsilon) 

    else:
        raise ValueError("Method must be 'subtraction' or 'division'.")

    # Optionally clip the corrected image
    if clip_corrected_image:
        corrected_image = np.clip(corrected_image, a_min=min_clip_value, a_max=max_clip_value)

    # Convert to desired output dtype if provided
    if output_dtype is not None:
        corrected_image = corrected_image.astype(output_dtype)

    return corrected_image


def correct_background(
    image:np.ndarray,
    background:np.ndarray,
    method:Optional[str]="subtraction",
    channel_axis:Optional[int]=None,
    offset:Optional[float]=0,
    epsilon:Optional[float]=1e-8,
    working_dtype:Optional[type]=float,
    output_dtype:Optional[type]=None,
    rescale_background:Optional[str]=None,
    clip_corrected_image:Optional[bool]=False,
    min_clip_value:Optional[float]=0,
    max_clip_value:Optional[float]=None,
    offset_background:Optional[bool]=False,
    zero_kwargs:Optional[dict]=None
)->np.ndarray:
    """
    Apply background correction to a microscopy image with the option of correcting each channel separately.

    Parameters    ----------
    - image : np.ndarray. The input image to be corrected.
    
    - background : np.ndarray. The background image to be subtracted or divided from the input image. Must be
    the same shape as the input image.
    
    - method : str, optional. Default is 'subtraction'. The method of background correction to apply. Must be
    either 'subtraction' or 'division'.
    
    - channel_axis : int or None, optional. Default is None. If specified, the background correction
    will be applied separately to each channel along the specified axis. For example, if channel_axis=0,
    the correction will be applied separately to each channel along the first dimension of the image.
    If None, the correction will be applied to the entire image without separating channels.
    
    - offset : int, optional. Default is 0. A constant value to SUBTRACT from the image before applying
    the correction.
    
    - epsilon : float, optional. Default is 1e-8. A small constant added to the denominator in division-based correction
    to prevent division by zero. Note that if epsilon is negative, it may lead to unexpected results in
    division-based correction. A warning is printed in that case but the process will still proceed.
    
    - working_dtype : type, optional. The data type to use for intermediate calculations.
    Default is float which allows for decimal values and prevents overflow during subtraction or division.
    You can set this to a specific type like np.float32 or np.float64 if desired.
    
    - output_dtype : type, optional. Default is None. The data type to convert the corrected image to before returning. If None,
    the corrected image will be returned in the same dtype as the working_dtype.
    
    - rescale_background : str, optional. If specified, the background
    image will be rescaled before applying the correction. Possible options are:
        - None: No rescaling will be applied to the background. This is the default behavior.
        - 'max': Rescale the background so that its maximum value is 1. This is done by dividing
        the background by its maximum value. This is the recommended option for division-based background
        correction to ensure that the background values are in a suitable range for division. Note that if the
        correction method is 'division' and rescale_background is not set to 'max', a warning will be printed
        but the process will still proceed.
        - 'minmax': Rescale the background to the range [0, 1] by subtracting the minimum value and
        dividing by the range (max - min).
    
    - clip_corrected_image : bool, optional. Default is False. If True, any values in the corrected
    image will be clipped to the range [min_clip_value, max_clip_value]. This can be useful to prevent negative
    and extreme pixel values in the output image, which may not be meaningful in the context of microscopy images.
    
    - min_clip_value : float, optional. Default is 0. The minimum value to which pixel
    values in the corrected image will be clipped if clip_corrected_image is True.
    
    - max_clip_value : float, optional. Default is None. The maximum value to which pixel
    values in the corrected image will be clipped if clip_corrected_image is True.

    - offset_background : bool, optional. Default False. If True the offset is subtracted also to the background
    function. Note that if background is rescaled the application of the offset happens before the rescaling.

    - zero_kwargs : dict, optional. Default is None. A dictionary of keyword arguments to pass
    to np.zeros when creating the output array for channel-wise correction. This allows you to
    specify additional parameters such as order when creating the zero array. Note that it is not possible
    to specify the dtype for the zero array in zero_kwargs since the dtype is determined by working_dtype.
    If zero_kwargs is None, an empty dictionary will be used and no additional parameters will
    be passed to np.zeros.

    Outputs    -------
    - corrected_image : np.ndarray. The background-corrected image, in the same shape as the input
    image and in the dtype specified by output_dtype (or working_dtype if output_dtype is None).
    If channel_axis is specified, the correction will be applied separately to each channel
    along the specified axis.

    """
    # set default zero_kwargs if not provided
    if zero_kwargs is None:
        zero_kwargs = {}
    else:
        # ensure that zero_kwargs does not contain a 'dtype' key since the dtype for the zero array is
        # determined by working_dtype
        assert 'dtype' not in zero_kwargs, "zero_kwargs should not contain a 'dtype' key as the dtype for the zero array is determined by working_dtype." 

    # Check that image and background have the same shape
    if image.shape != background.shape:
        raise ValueError("Image and background must have the same shape.")
    
    # Apply background correction to the entire image if channel_axis is None
    if channel_axis is None:
        corrected_image = correct_background_nd(
            image=image,
            background=background,
            method=method,
            offset=offset,
            epsilon=epsilon,
            working_dtype=working_dtype,
            output_dtype=output_dtype,
            rescale_background=rescale_background,
            clip_corrected_image=clip_corrected_image,
            min_clip_value=min_clip_value,
            max_clip_value=max_clip_value,
            offset_background=offset_background
        )

    # Apply background correction separately to each channel if channel_axis is specified
    elif isinstance(channel_axis, int):
        
        # normalize axis to the image dimensions
        channel_axis = np.core.numeric.normalize_axis_index(channel_axis, image.ndim)

        # move channel axis to the first position
        image_moved = np.moveaxis(image, channel_axis, 0)
        background_moved = np.moveaxis(background, channel_axis, 0)

        # istantiate a container array to store corrected sub-arrays along the channel axis
        corrected = np.zeros_like(image_moved, dtype=working_dtype, **zero_kwargs)

        # iterate over channels and correct for background
        for ch in range(image.shape[channel_axis]):
            corrected[ch] = correct_background_nd(
                image=image_moved[ch],
                background=background_moved[ch],
                method=method,
                offset=offset,
                epsilon=epsilon,
                working_dtype=working_dtype,
                output_dtype=working_dtype, # note that in this case the output data type is consistent with the working data type
                rescale_background=rescale_background,
                clip_corrected_image=clip_corrected_image,
                min_clip_value=min_clip_value,
                max_clip_value=max_clip_value,
                offset_background=offset_background
            )
        
        # move back the channel axis to the original position
        corrected_image = np.moveaxis(corrected, 0, channel_axis)

        # change the output dtype if specified - NOTE: this is required as corrected_image has working_dtype
        if output_dtype is not None:
            corrected_image = corrected_image.astype(output_dtype)

    else:
        raise ValueError("channel_axis must be None or an integer specifying the channel dimension.")

    return corrected_image

