from typing import Union
import numpy as np
import pandas as pd
from utils.label_image_utils import permute_values_per_label


def measure_corr_coeff_per_label_single_pair(label_image:np.array,
                                             intensity_image_stack:np.array,
                                             axis:int=0,
                                             n_permutation:int=0,
                                             ignore_index:bool=True,
                                             return_permutations:bool=False,
                                             method:str|None=None,
                                             min_periods:int=1,
                                             numeric_only:bool=True,
                                             **kwargs) -> pd.DataFrame | tuple:
    """
    measures the correlation coefficients between the intensity values of two images per label in the label image.

    The function calculates the correlation coefficients between the intensity values of two images per label in the label image.
    If n_permutation is set to a value greater than 0, the function performs permutations of the pixel positions within each label and calculates
    the mean and standard deviation of the correlation coefficients obtained after the permutation.

    Inputs:
    - intensity_image_stack: np.array. Must have at least 2 dimensions. The stack of intensity images to compare. Each image must have the same shape.
    The images to compare are expected to be stacked along the specified axis. Images can have any number of dimensions.
    
    - label_image: np.array. The shape must correspond to that of the individual images to compare. The labels for each pixel in the intensity images.
    
    - axis: int. Optional. Default 0. The axis along which the intensity images are stacked.
    
    - n_permutation: int. Optional. Default 0. The number of permutations to perform for the intensity values per label.
    If 0, no permutations are performed.
    
    - ignore_index: bool. Optional. Default True. Whether or not to ignore indexes when concatenating permutations pd.Series. NOTE: this is meant to be
    True, the behaviour of the function is not guaranteed if set to False.
    
    - return_permutations: bool. Optional. Default False. If True, the individual permutations are returned as pd.DataFrame in addition
    to the main result.
    
    - method: str or None. Optional. Default 'spearman'. The method to use for calculating the correlation coefficients. Can be 'pearson', 'spearman', or 'kendall'.
    It is passed to pandas.DataFrame.corr() method.
    
    - min_periods: int. Optional. Default 1. Minimum number of observations required per label to have a valid result.
    This is passed to pandas.DataFrame.corr() method.
    If the number of observations is less than min_periods, NaN is returned.
    
    - numeric_only: bool. Optional. Default True. Whether to include only numeric columns in the correlation calculation.
    This is passed to pandas.DataFrame.corr() method.
    
    - **kwargs: Additional keyword arguments passed to permute_values_per_label function.

    Returns:
    - If n_permutation is 0, returns a pd.Series with the correlation coefficients for each label. Rows are labels, columns are the correlation coefficients.
    - If n_permutation > 0 and return_permutations is False, returns a pd.DataFrame with the observed correlation coefficients for each label and the
    the mean and standard deviation of the correlation coefficients obtained after the permutation of the pixel positions within each label.
    Rows are labels, columns are:
        - f'permuted_mean_{method}_coeff': the mean of the correlation coefficients obtained after the permutation of the pixel positions within each label.
        - f'permuted_std_{method}_coeff': the standard deviation of the correlation coefficients obtained after the permutation of the pixel positions within each label.
        - {method}_coeff: the observed correlation coefficient for each label.
    - If n_permutation > 0 and return_permutations is True, returns a tuple with:
        - position-0. pd.DataFrame with the observed correlation coefficients for each label and the
    the mean and standard deviation of the correlation coefficients obtained after the permutation of the pixel positions within each label (see above).
        - position-1. pd.DataFrame with the correlation coefficients obtained per each individual permutations. Each row corresponds to a label, and
        the columns are the correlation coefficients for each permutation.

    NOTE: all labels in label image are threated equally, including 0, irrespective of the
    the possibility that a value corresponds to the background.
    
    NOTE: this function calculates correlation coeffients using pandas.DataFrame.corr() method. When there is no variation in the data NaN is returned.

    NOTE: the function assumes that the intensity_image_stack has exactly 2 images to compare and are stacked along axis.
    If more than 2 images are provided, only the first two images are used for the correlation calculation.
    If less than 2 images are provided, an error is raised.

    NOTE: the function was not tested for images of 1 pixel.
    """
    # set default method
    if method is None:
        method='spearman'

    assert method not in kwargs, "method can't be passed to kwargs. Use the method argument instead."

    if return_permutations:
        assert n_permutation>0, "return_permutations can only be True if n_permutation > 0"

    # split the intensity_image_stack along the specified axis
    intensity_image_list = [np.squeeze(a, axis=axis) for a in np.split(intensity_image_stack, indices_or_sections=intensity_image_stack.shape[axis], axis=axis)]
    
    # flatten the intensity sub-arrays and the label image
    flat_intensity_0 = intensity_image_list[0].ravel()
    flat_intensity_1 = intensity_image_list[1].ravel()
    flat_labels = label_image.ravel()

    # Create a DataFrame to hold the values
    df = pd.DataFrame({'label': flat_labels, 'value_0': flat_intensity_0, 'value_1': flat_intensity_1})
        
    # Group by label, apply correlation, unstack the result and select the relevant correlation coefficients
    correlation_series = df.groupby('label')[['value_0','value_1']].corr(method=method, min_periods=min_periods, numeric_only=numeric_only).unstack().iloc[:,1]

    # If no permutations are requested, return the correlation series
    if n_permutation==0:
        return correlation_series
    
    # If permutations are requested, perform them
    else:
        # Check that the kwargs do not contain 'replace' or 'ignore_index'
        assert 'replace' not in kwargs, "replace can't be passed to kwargs and it is by default set to False in permute_values_per_label function"
        assert ignore_index not in kwargs, "ignore_index can't be passed to kwargs. Use the ignore_index argument instead."

        # intiialize a list to collect the results of the permutations
        collection_permutation = []

        # Perform the permutations n_permutation times
        for i in range(n_permutation):

            # Permute the values per label
            permutation_df = permute_values_per_label(label_image=label_image,
                                                       intensity_image_stack=intensity_image_stack,
                                                       axis=axis,
                                                       **kwargs)
            # Group by label, apply correlation, unstack the result and select the relevant correlation coefficients
            i_correlation_df = permutation_df.groupby('label')[['value_0','value_1']].corr(method=method, min_periods=min_periods, numeric_only=numeric_only).unstack().iloc[:,1]
            
            # collect the results of the permutation
            collection_permutation.append(i_correlation_df)

        # Concatenate the results of the permutations into a DataFrame
        collection_permutation_df = pd.concat(collection_permutation, axis=1, ignore_index=ignore_index)
        
        # Calculate the mean and standard deviation of the correlation coefficients for each label
        collection_permutation_df[f'permuted_mean_{method}_coeff'] = collection_permutation_df.mean(axis=1)
        collection_permutation_df[f'permuted_std_{method}_coeff'] = collection_permutation_df.std(axis=1)

        # Join the observed correlation coefficients with the permuted ones
        correlation_df = collection_permutation_df[[f'permuted_mean_{method}_coeff', f'permuted_std_{method}_coeff']].join(correlation_series.rename(f'{method}_coeff'), how='right')

        # If return_permutations is True, return the correlation_df and the collection_permutation_df
        if return_permutations:
            return correlation_df, collection_permutation_df
        
        # If return_permutations is False, return only the correlation_df
        else:
            return correlation_df


def measure_corr_coeff_per_label(label_image:np.array,
                                 intensity_image_stack:np.array,
                                 axis:int=0,
                                 n_permutation:int=0,
                                 ignore_index:bool=True,
                                 return_permutations:bool=False,
                                 method:str|None=None,
                                 min_periods:int=1,
                                 numeric_only:bool=True,
                                 sep:str|None=None,
                                 **kwargs) -> pd.DataFrame | tuple:
    """
    Given a stack of intensity images and a label image, for each label in the label image and for all the image-pairs in a stack, this function measures
    the correlation coefficients between the intensity values.
    If n_permutation is set to a value greater than 0, the function performs permutations of the pixel positions within each label and calculates
    the mean and standard deviation of the correlation coefficients obtained after the permutation.

    Inputs:
    - intensity_image_stack: np.array. Must have at least 2 dimensions. The stack of intensity images to compare. Each image must have the same shape.
    The images to compare are expected to be stacked along the specified axis. Images can have any number of dimensions.
    
    - label_image: np.array. The shape must correspond to that of the individual images to compare. The labels for each pixel in the intensity images.
    
    - axis: int. Optional. Default 0. The axis along which the intensity images are stacked.
    
    - n_permutation: int. Optional. Default 0. The number of permutations to perform for the intensity values per label.
    If 0, no permutations are performed.
    
    - ignore_index: bool. Optional. Default True. Whether or not to ignore indexes when concatenating permutations pd.Series. NOTE: this is meant to be
    True, the behaviour of the function is not guaranteed if set to False.
    
    - return_permutations: bool. Optional. Default False. If True, the individual permutations are returned as pd.DataFrame in addition
    to the main result.
    
    - method: str or None. Optional. Default 'spearman'. The method to use for calculating the correlation coefficients. Can be 'pearson', 'spearman', or 'kendall'.
    It is passed to pandas.DataFrame.corr() method.
    
    - min_periods: int. Optional. Default 1. Minimum number of observations required per label to have a valid result.
    This is passed to pandas.DataFrame.corr() method.
    If the number of observations is less than min_periods, NaN is returned.
    
    - numeric_only: bool. Optional. Default True. Whether to include only numeric columns in the correlation calculation.
    This is passed to pandas.DataFrame.corr() method.
    
    - sep. str or None. Optional. Default '_'. The string to use for separating the channels from the rest of the column name in the output dataframe.
    
    - **kwargs: Additional keyword arguments passed to permute_values_per_label function.

    Returns:
    - If n_permutation is 0, returns a pd.DataFrame with the correlation coefficients for each label and image pair.
    Rows are labels, columns have the following syntax: {method}_coeff{sep}{int0}_{int1} where method is the correlation method used and int0 and int1 are the
    indexes of the images used for the measurement along the specified axis.
    - If n_permutation > 0 and return_permutations is False, returns a pd.DataFrame with the observed correlation coefficients for each label and the
    the mean and standard deviation of the correlation coefficients obtained after the permutation of the pixel positions within each label is returned
    per each image pair.
    Rows are labels, columns are (per each image pair):
        - permuted_mean_{method}_coeff{sep}{int0}_{int1}: the mean of the correlation coefficients obtained after the permutation of the pixel positions within each label.
        - permuted_std_{method}_coeff{sep}{int0}_{int1}: the standard deviation of the correlation coefficients obtained after the permutation
        of the pixel positions within each label.
        - {method}_coeff{sep}{int0}_{int1}: the observed correlation coefficient for each pair of images.
    - If n_permutation > 0 and return_permutations is True, returns a tuple with:
        - position-0. pd.DataFrame with the observed correlation coefficients for each label and the
    the mean and standard deviation of the correlation coefficients obtained after the permutation of the pixel positions within each label (this is
    the exact same output of setting return_permutation False - see above -).
        - position-1. pd.DataFrame with the correlation coefficients obtained per each individual permutations. Each row corresponds to a label, and
        the columns are the correlation coefficients for each permutation and each image pair. The syntax of the columns is:
        {int0}{sep}{int1}_{int2} where int0 is the permutation number and int0 and int1 are the indexes of the images used for the measurement along
        the specified axis.

    NOTE: all labels in label image are threated equally, including 0, irrespective of the
    the possibility that a value corresponds to the background.
    
    NOTE: this function calculates correlation coeffients using pandas.DataFrame.corr() method. When there is no variation in the data NaN is returned.

    NOTE: the function assumes that the intensity_image_stack has at least 2 images to compare and are stacked along axis.
    If less than 2 images are provided, an error is raised.

    NOTE: the function was not tested for images of 1 pixel.
    """

    # set default method and sep
    if method is None:
        method='spearman'
    
    if sep is None:
        sep='_'
    
    assert method not in kwargs, "method can't be passed to kwargs. Use the method argument instead."

    if return_permutations:
        assert n_permutation>0, "return_permutations can only be True if n_permutation > 0"

    # split the intensity_image_stack along the specified axis - this is done to allow passing image stacks with images stacked along any axis
    intensity_image_list = [np.squeeze(a, axis=axis) for a in np.split(intensity_image_stack, indices_or_sections=intensity_image_stack.shape[axis], axis=axis)]
    
    # initialize a list to collect the results
    corr_coeff_collection = []

    # if return_permutations is True, initialize a list to collect the individual permutations
    if return_permutations:
        permutation_collection = []
    
    # iterate over the pairs of images in the intensity_image_list
    for c_i, i in enumerate(intensity_image_list):
        
        # initialize a positional counter for the second image in the pair
        c_j = c_i + 1

        # iterate over the second image in the pair
        for j in intensity_image_list[c_j:]:

            # stack the image pair on the first axis - this is done to allow passing image stacks with images stacked along any axis
            image_pair = np.stack((i, j), axis=0)

            # measure the correlation coefficients for the current pair of images
            i_correlation_df = measure_corr_coeff_per_label_single_pair(intensity_image_stack=image_pair,
                                                                         label_image=label_image,
                                                                         axis=0, # axis=0 because the images are stacked along the first axis
                                                                         n_permutation=n_permutation,
                                                                         ignore_index=ignore_index,
                                                                         return_permutations=return_permutations,
                                                                         method=method,
                                                                         min_periods=min_periods,
                                                                         numeric_only=numeric_only,
                                                                         **kwargs)

            # if return_permutations is True, collect the correlation coefficients and the individual permutations
            # for the current pair of images to the collection
            if return_permutations:

                # replace the column names of the individual permutations with the pair index
                i_correlation_df[0].columns = [f'{col}{sep}{c_i}_{c_j}' for col in i_correlation_df[0].columns]
                i_correlation_df[1].columns = [f'{col}{sep}{c_i}_{c_j}' for col in i_correlation_df[1].columns]

                corr_coeff_collection.append(i_correlation_df[0])
                permutation_collection.append(i_correlation_df[1])
            
            # if return_permutations is False, collect only the correlation coefficients for the current pair of images to the collection
            else:
                # replace the column names of the correlation coefficients with the pair index
                # i_correlation_df is a pd.DataFrame if n_permutation > 0, otherwise it is a pd.Series
                if n_permutation>0:
                    i_correlation_df.columns = [f'{col}{sep}{c_i}_{c_j}' for col in i_correlation_df.columns]
                else:
                    i_correlation_df.rename(f'{method}_coeff{sep}{c_i}_{c_j}', inplace=True)

                corr_coeff_collection.append(i_correlation_df)

            # update the second image index
            c_j += 1
    
    # concatenate the results of the correlation coefficients into a DataFrame
    corr_coeff_df = pd.concat(corr_coeff_collection, axis=1, ignore_index=False)

    # if return_permutations is True, concatenate the individual permutations into a DataFrame
    if return_permutations:
        permutation_df = pd.concat(permutation_collection, axis=1, ignore_index=False)
        return corr_coeff_df, permutation_df

    return corr_coeff_df
