import numpy as np

def correct_background_nd(
    image,
    background,
    method="subtraction",
    offset=0,
    epsilon=1e-8,
    working_dtype=float,
    output_dtype=None,
    rescale_background=None
):
    """
    Apply background correction to a microscopy image.
    """

    # Convert once to working dtype
    image = image.astype(working_dtype)
    background = background.astype(working_dtype)

    # Optional background rescaling
    if rescale_background is None:
        pass

    elif rescale_background == "max":
        max_val = np.max(background)
        background = background / (max_val + epsilon)

    elif rescale_background == "minmax":
        min_val = np.min(background)
        max_val = np.max(background)
        background = (background - min_val) / (max_val - min_val + epsilon)

    else:
        raise ValueError("rescale_background must be None, 'max', or 'minmax'.")

    if offset > 0 and method == "division":
        min_val = np.min(image)
        if offset >= min_val:
            raise ValueError(f"Offset must be less than the minimum pixel value in the image ({min_val}) to avoid negative values in the division.")

    # Apply correction
    if method == "subtraction":
        corrected_image = image - background - offset

    elif method == "division":
        if rescale_background != "max":
            print("Warning: division-based background correction typically requires the background to be rescaled to a maximum of 1. Consider setting rescale_background='max' for optimal results.")
        
        corrected_image = (image - offset) / (background + epsilon) 

    else:
        raise ValueError("Method must be 'subtraction' or 'division'.")

    # Convert to desired output dtype if provided
    if output_dtype is not None:
        corrected_image = corrected_image.astype(output_dtype)

    return corrected_image


def correct_background(
    image,
    background,
    method="subtraction",
    offset=0,
    channel_axis=None,
    epsilon=1e-8,
    working_dtype=float,
    output_dtype=None,
    rescale_background=None
):
    """
    Apply background correction to a microscopy image separately for each channel.
    """

    if channel_axis is None:
        corrected_image = correct_background_nd(
            image=image,
            background=background,
            method=method,
            offset=offset,
            epsilon=epsilon,
            working_dtype=working_dtype,
            output_dtype=output_dtype,
            rescale_background=rescale_background
        )

    elif isinstance(channel_axis, int):

        corrected_image = np.zeros_like(image, dtype=working_dtype)

        for ch in range(image.shape[channel_axis]):
            corrected_image[ch] = correct_background(
                image=image[ch],
                background=background[ch],
                method=method,
                offset=offset,
                epsilon=epsilon,
                working_dtype=working_dtype,
                output_dtype=output_dtype,
                rescale_background=rescale_background
            )

    else:
        raise ValueError("channel_axis must be None or an integer specifying the channel dimension.")

    return corrected_image

