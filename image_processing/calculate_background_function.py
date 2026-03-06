import os
import pandas as pd
import numpy as np
import tifffile

def import_fov(df: pd.DataFrame,
                fov_dir: str,
                fov_clm: str,
                fov_shape: tuple,
                verbose: bool = True,
                np_zero_kwargs: dict | None = None,
                sample_df: bool = False,
                sample_fraction: float = 0.5,
                sample_kwargs: dict | None = None)-> np.ndarray:

    # copy the dataframe to avoid modifying the original one
    df_copy = df.copy()

    # sample the dataframe if requested
    if sample_df:
        if sample_kwargs is None:
            sample_kwargs = {"random_state": 42}
        assert "frac" not in sample_kwargs, "sample_kwargs cannot contain 'frac' as it is passed separately via sample_fraction"
        df_copy = df_copy.sample(frac=sample_fraction, **sample_kwargs)

    # use default kwargs
    if np_zero_kwargs is None:
        np_zero_kwargs = {'dtype':np.float32}
    else:
        # use float32 as default data type for container array
        if 'dtype' not in np_zero_kwargs:
            np_zero_kwargs["dtype"] = np.float32

    # Initialize an array container to store the fields of view for background function calculation
    # NOTE: the axis along which the fields of view are stored is hard coded to be the last one (i.e. -1)
    container_arr_shape = tuple([a for a in fov_shape] + [df_copy.shape[0]]) # the shape of the container array is the shape of the fields of view + the number of fields of view (aka number of rows in the metadata dataframe)
    container_arr = np.zeros(container_arr_shape, **np_zero_kwargs)

    # initialize a positional counter
    pos_counter = 0

    # iterate over the scenes and store the fields of view in the container array
    for scene_idx in df_copy.index:
        
        # get the field of view name from the metadata dataframe
        fov_name = df_copy.loc[scene_idx, fov_clm]

        # open the field of view using tifffile
        fov_path = os.path.join(fov_dir, fov_name)
        fov_arr = tifffile.imread(fov_path)
        
        # store the field of view in the container array
        container_arr[..., pos_counter] = fov_arr
        
        # update the positional counter
        pos_counter += 1

    if verbose:
        print(f"container_arr shape: {container_arr.shape}")
    
    return container_arr

def calculate_background_function(container_arr: np.ndarray,
                                  method: str = "median",
                                  axis: int = -1,
                                  verbose: bool = True)-> np.ndarray:
    
    # copy the container array to avoid modifying the original one
    container_arr_copy = container_arr.copy()

    # average the pixel values along the last axis (the one corresponding to the
    # different fields of view) to calculate the background function
    # use the method indicated in background_function_method
    if method == "median":
        background_function = np.median(container_arr_copy, axis=axis) # NOTE: the axis along which the fields of view are stored was hard coded to be the last one (i.e. -1). See container_arr initialization and population above.
    elif method == "mean":
        background_function = np.mean(container_arr_copy, axis=axis) # NOTE: the axis along which the fields of view are stored was hard coded to be the last one (i.e. -1). See container_arr initialization and population above.
    else:
        raise ValueError(f"method {method} not recognized. Please use 'median' or 'mean'.")

    if verbose:
        print(f"background_function shape: {background_function.shape}")
    
    return background_function

def calculate_bg_funct_per_condition(df: pd.DataFrame,
                                     condition_clm:str,
                                     fov_dir:str,
                                     fov_clm: str,
                                     fov_shape:tuple,
                                     method:str='median',
                                     stack_axis:int=-1,
                                     verbose:bool=True,
                                     )->dict:

    # copy df to avoid modification
    df_copy = df.copy()

    # Get all unique conditions in the input dataframe for the condition target column
    unique_conditions= df_copy[condition_clm].unique()
    print(f"unique conditions: {unique_conditions}")
    print("--- --- ---")

    # Initialize a dictionary to store the background functions calculated per each unique condition
    background_functions_per_condition = {}

    # iterate over the unique conditions
    for uni_cond in unique_conditions:

        if verbose:
            print("--- --- ---", uni_cond)
        
        # filter the input dataframe to select only the rows corresponding to the current condition
        df_condition = df_copy[df_copy[condition_clm] == uni_cond]

        # initialize a container array for the current condition
        container_arr_condition = import_fov(df=df_condition,
                                             fov_dir=fov_dir,
                                             fov_clm=fov_clm,
                                             fov_shape=fov_shape)

        # calculate the background function for the current condition
        background_function_condition = calculate_background_function(container_arr=container_arr_condition,
                                                                      method=method,
                                                                      axis=stack_axis,
                                                                      verbose=verbose)

        # add the calculated background function to the dictionary
        background_functions_per_condition[uni_cond] = background_function_condition
    
    return background_functions_per_condition



def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    === === ===
    Source - https://stackoverflow.com/a/57923405
    Posted by Paddy Harrison, modified by community. See post 'Timeline' for change history
    Retrieved 2026-02-11, License - CC BY-SA 4.0
    === === ===

    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.ravel()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)


def get_polyfit_background_function(background_function: np.ndarray,
                                kx: int = 3,
                                ky: int = 3,
                                order: int = None,
                                verbose: bool = True)-> np.ndarray:
    """
    Fit a 2d polynomial to the background function and return the fitted polynomial as a 2d array.
    
    Parameters
    ----------
    background_function: np.ndarray
        The background function to fit the polynomial to.
    
    kx: int, default is 3
        The order of the polynomial in the x direction.
    
    ky: int, default is 3
        The order of the polynomial in the y direction.
    
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.
    
    verbose: bool, default is True
        If True, print the shape of the fitted polynomial background function.
    
    Returns
    -------
    np.ndarray
        The fitted polynomial background function as a 2d array.
     
    """
    # copy the background function to avoid modifying the original one
    background_function_copy = background_function.copy()

    # get the x and y coordinates of the background function
    x = np.arange(background_function_copy.shape[1])
    y = np.arange(background_function_copy.shape[0])

    # fit a 2d polynomial to the background function
    soln, residuals, rank, s = polyfit2d(x, y, background_function_copy, kx=kx, ky=ky, order=order)

    # reshape the solution to get the polynomial coefficients in a 2d array
    poly_coeffs = soln.reshape((kx+1, ky+1))

    # evaluate the fitted polynomial on the grid defined by x and y
    polyfit_background_function = np.polynomial.polynomial.polygrid2d(x, y, poly_coeffs)

    if verbose:
        print(f"polyfit_background_function shape: {polyfit_background_function.shape}")
    
    return polyfit_background_function

def get_polyfit_bg_funct_channel(background_function: np.ndarray,
                                    channel_axis: int,
                                    kx: int | tuple = 3,
                                    ky: int | tuple = 3,
                                    order: int | tuple | None = None,
                                    verbose: bool = True)-> np.ndarray:
    """
    Fit a 2d polynomial to the background function for each channel and return the fitted polynomial background functions as a 3d array.
    
    Parameters
    ----------
    background_function: np.ndarray
        The background function to fit the polynomial to. The background function is expected to have a channel axis along which the different channels are organized.
    
    channel_axis: int
        The axis along which the channels are organized in the background function array.
    
    kx: int or tuple of ints, default is 3
        The order of the polynomial in the x direction.
        If tuple of ints, the order of the polynomial in the x direction can be different for each channel.
        The length of the tuple should be equal to the number of channels.
        A tuple of ints must be passed also to ky and order (if not None) in this case.
    
    ky: int or tuple of ints, default is 3
        The order of the polynomial in the y direction.
        If tuple of ints, the order of the polynomial in the y direction can be different for each channel.
        The length of the tuple should be equal to the number of channels.
        A tuple of ints must be passed also to kx and order (if not None) in this case.
    
    order: int or tuple of ints or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.
        If tuple of ints, the maximum order of the polynomial coefficients to consider can be different for each channel.
        The length of the tuple should be equal to the number of channels.
        A tuple of ints must be passed also to kx and ky in this case.
    
    verbose: bool, default is True
        If True, print the shape of the fitted polynomial background function for each channel.
    
    Returns
    -------
    np.ndarray
        The fitted polynomial background functions for each channel as a 3d array.
        The channel axis is the same as the input background function.
    """
    
    # if kx, ky and order are ints, convert them to tuples of ints with length equal to the number of channels in the background function
    if isinstance(kx, int):
        kx = tuple([kx] * background_function.shape[channel_axis])
    
    if isinstance(ky, int):
        ky = tuple([ky] * background_function.shape[channel_axis])
    
    if isinstance(order, int) or order is None:
        order = tuple([order] * background_function.shape[channel_axis])
    
    # check if kx, ky and order are tuples and if their length is equal to the number of channels
    # in the background function
    if isinstance(kx, tuple) or isinstance(ky, tuple) or isinstance(order, tuple):
        if not (isinstance(kx, tuple) and isinstance(ky, tuple) and isinstance(order, tuple)):
            raise ValueError("If kx, ky or order is a tuple, all of them must be tuples.")
        if not (len(kx) == len(ky) == len(order) == background_function.shape[channel_axis]):
            raise ValueError("If kx, ky or order is a tuple, their length must be equal to the number of channels in the background function.")
    

    # copy the background function to avoid modifying the original one
    background_function_copy = background_function.copy()

    # move the channel axis to the last axis for easier processing
    background_function_copy = np.moveaxis(background_function_copy, channel_axis, -1)

    # create an empty array to store the fitted polynomial background functions for each channel
    polyfit_bg_funct_channel = np.zeros_like(background_function_copy)

    # unstack the background function along the channel axis
    unstacked_bg_funct = np.unstack(background_function_copy, axis=-1)

    # for each channel, fit a 2d polynomial to the background function and store the result
    for ch, bg_funct_ch in enumerate(unstacked_bg_funct):

        if verbose:
            print(f"Processing channel {ch}...")
        
        # fit a 2d polynomial to the background function for this channel
        polyfit_bg_funct_ch = get_polyfit_background_function(bg_funct_ch, kx=kx[ch], ky=ky[ch], order=order[ch], verbose=verbose)

        # store the fitted polynomial background function for this channel
        polyfit_bg_funct_channel[..., ch] = polyfit_bg_funct_ch
    
    # move the channel axis back to its original position
    polyfit_bg_funct_channel = np.moveaxis(polyfit_bg_funct_channel, -1, channel_axis)

    return polyfit_bg_funct_channel