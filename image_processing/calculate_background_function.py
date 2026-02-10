import os
import pandas as pd
import numpy as np
import tifffile

def import_fov(df: pd.DataFrame,
                fov_dir: str,
                fov_clm: str,
                fov_shape: tuple,
                verbose: bool = True)-> np.ndarray:

    # copy the dataframe to avoid modifying the original one
    df_copy = df.copy()

    # Initialize an array container to store the fields of view for background function calculation
    # NOTE: the axis along which the fields of view are stored is hard coded to be the last one (i.e. -1)
    container_arr_shape = tuple([a for a in fov_shape] + [df_copy.shape[0]]) # the shape of the container array is the shape of the fields of view + the number of fields of view (aka number of rows in the metadata dataframe)
    container_arr = np.zeros(container_arr_shape)

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