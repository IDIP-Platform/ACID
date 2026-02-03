import numpy as np
import pandas as pd
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.morphology import disk, erosion
from skimage.measure import regionprops_table


class MeasureStructureTensor():
    """
    NOTE: this class will be changed in the future to avoid the repetition of the functions measure_obj_struct_tensor_eigenval_single_image_sigma,
    measure_object_anisotropy_single_sigma_image and (potentially) rename_columns.
    It could also change to make num_sigma, sigma_max and sigma_min names and behavior closer to MeasureHessianMatrix class.
    
    The two relevant methods are measure_obj_struct_tensor_eigenval and measure_object_anisotropy. These are the last 2 methods at the bottom of the class. All the rest of the methods are
    building blocks for these two methods and their results can be obtained using measure_obj_struct_tensor_eigenval and measure_object_anisotropy.

    ========= ========= =========
    Intuition:
    The structure tensors of an image are the raw gradient statistics in the image axes. AKA: how strong and
    how correlated is the intensity gradient along the axes of the image.
    
    The eigenvalues/vectors are the same information, but rotated into the best-fitting local
    coordinate frame where one axis aligns with the strongest gradient direction.
    
    For this reason, for a 2D image:
    - the structure tensors capture the local gradient orientation and strength.
    - their eigenvalues (aka the eigenvalues of their eigenvectors) tell:
        - if both are large -> there is a corner (strong gradient in all directions).
        - one large and one small -> there is an edge (strong gradient only in one direction).
        - both small -> there is a flat region.
    
    ========= ========= =========

    measure_obj_struct_tensor_eigenval: given an image (passed to __init__) and corresponding labelled objects (label_image) measures the statistics (by default mean, max, min and std) of the intensity distribution
    of the structure tensor eigenvalues for each labelled object in a label image. Precisely:

    1) the structure tensor eigenvalues (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.structure_tensor and
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.structure_tensor_eigenvalues) are calculated per each pixel of the image.
    The output of this step is an image stack (eigen_stack), where each channel corresponds to one of the eigenvalues.

    2) The labelled objects in label_image are eroded using the the parameters specified in erosion_kwargs. By default, a disk-shaped structuring element of radius 9 is used.

    3) The measurements are computed from the eigen_stack by applying the regionprops_table function (https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops_table)
    from skimage.measure to extract the relevant statistics for each eroded labelled object. NOTE: as the measurements are computed using regionprops_tables, any measurement compatible with regionprops_table can be
    extracted, including extra_properties. Any argument accepted by regionprops_table can be passed to the argument regionprops_kwargs of the method.

    4) The measurements are returned as a pandas dataframe. The dataframe structure and layout matches regionprops output: rows correspond to labelled objects, columns correspond to measurements. If image has
    multiple channels, the measurements are returned for each channel separately and the channel index is indicated as the last number of the column name. The method uses '-' as default separator for the
    channel index in the column name. The separator can be changed by passing a different value to regionprops_kwargs. NOTE: if '_' is used as separator the output might be wrong. The method does not
    return an error but raises a warning.


    The structure tensor eigenvalues are measurements of the intensity gradient (aka the steepness of the intensity change) in a neighbourhood of a pixel. The highest the eigenvalue, the strongest the intensity change.
    There are as many structure tensor eigenvalues as dimensions of the image. If a pixel has a prevalent eigenvalue it indicates that in its neighbourhood there is a prevalent orientation of the intensity change. In
    other words, there is an edge-like pattern.
    
    ========= ========= =========

    measure_object_anisotropy: given an image (passed to __init__) and corresponding labelled objects (label_image) measures the statistics (by default mean, max, min and std) of the intensity distribution
    of the ratio between each structure tensor eigenvalues for each labelled object in a label image. Such ration is an indication of the
    signal anisotropy for the object (https://scikit-image.org/docs/0.25.x/auto_examples/applications/plot_3d_structure_tensor.html). The intensity signal is anisotropic if there is a prevalent
    direction along which it changes, for example in the case of striped patterns. The method, precisely:

    1) the structure tensor eigenvalues (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.structure_tensor and
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.structure_tensor_eigenvalues) are calculated per each pixel of the image.
    The output of this step is an image stack (eigen_stack), where each channel corresponds to one of the eigenvalues.

    2) The image structure eigenvalues are divided with each others. The division always uses a higher eigenvalue as the numerator and a smaller eigenvalue as denominator, and it is progressively computed by
    iterating through the eigenvalues from the highest to the lowest and pairing them with the remaining, lower eigenvalues from the highest to the lowest. For example, with 4 eigenvalues (eig1, eig2, eig3, eig4) the
    following ratios are computed, in order: eig1/eig2, eig1/eig3, eig1/eig4, eig2/eig3, eig2/eig4 and eig3/eig4. The output of this step is an image stack (eigen_ratio_stack),
    where each channel corresponds to one of the eigenvalue ratio.

    3) The labelled objects in label_image are eroded using the the parameters specified in erosion_kwargs. By default, a disk-shaped structuring element of radius 9 is used.

    4) The measurements are computed from the eigen_ratio_stack by applying the regionprops_table function (https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops_table)
    from skimage.measure to extract the relevant statistics for each eroded labelled object. NOTE: as the measurements are computed using regionprops_tables, any measurement compatible with regionprops_table can be
    extracted, including extra_properties. Any argument accepted by regionprops_table can be passed to the argument regionprops_kwargs of the method. NOTE: as the measurements are computed on per-pixel eigenvalue
    ratio, the output of this method is different than what would be obtained by dividing the measurements of measure_obj_struct_tensor_eigenval.

    5) The measurements are returned as a pandas dataframe. The dataframe structure and layout matches regionprops output: rows correspond to labelled objects, columns correspond to measurements.
    If image has multiple channels, the measurements are returned for each channel separately and the channel index is indicated as the last number of the column name. The method uses '-' as default separator for the
    channel index in the column name. The separator can be changed by passing a different value to regionprops_kwargs. NOTE: if '_' is used as separator the output might be wrong. The method does not
    return an error but raises a warning.
    
    ========= ========= =========

    NOTE:
    
    - the measurements which are returned by measure_obj_struct_tensor_eigenval and measure_object_anisotropy are obtained by aggregating the pixels of each object in label_image.
    In addition, the default behaviour is to erode the object before computing the measurements and using a disk-shaped structuring element of radius 9 as footprint.
    This behaviour is done to counteract the influence of the object boundaries on the measurements for small sigma, as the a relatively strong
    intensity gradient is expected at the object boundaries (between object and background). However, this behaviour can cause the loss of small objects as well as
    errors if the footprint dimensions don't match the number of dimensions of label_image. This behaviour can be controlled using the parameter erosion_kwargs,
    which takes all the keyword arguments for the erosion function (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.erosion).

    - The size of the neighbourhood for which the intensity gradient is calculated can be indicated using a parameter called sigma. Both measure_obj_struct_tensor_eigenval and measure_object_anisotropy return measurements
    for a range of sigmas, if provided. Such a range can be specified using the parameters min_sigma, max_sigma and num_sigma to specify, respectively, the smallest, largest and number of equally spaced steps
    for the sigma range. This behaviour is similar to the behaviour of sigma in https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.structure_tensor and
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.structure_tensor_eigenvalues

    - If the image to analyze has multiple channels, it is possible to specify the position of the channel axis (parameter axis). In this case label_image must have the same shape of each individual channel.
    In this case, both for measure_obj_struct_tensor_eigenval and measure_object_anisotropy the results are returned per each sigma (if provided) and per each channel. The methods follow regionprops convention, thus
    the channel index is indicated as the last number of the column name. Also, the method uses '-' as default separator for the
    channel index in the column name. The separator can be changed by passing a different value to regionprops_kwargs. NOTE: if '_' is used as separator the output might be wrong. The method does not
    return an error but raises a warning. When multiple sigmas are provided and multiple channels are present, the output column names will include sigma index in the specified range as the second last number and the
    channel index as the last number. The channel index is further recognized as separated by the specified separator (regionprops_kwargs['separator']). The sigma index is returned instead of the actual sigma value as,
    as reported below, a number of formats are possible for sigma values including iterables and floats.

    - The input image can theoretically have n-dimensions. It has been tested on 2D images, 2D images with multiple channels, 3D images and 3D images with multiple channels. Due to the default setting of erosion
    (see erosion_kwargs below), the default setting will return an error when working with images with more than 2 spatial dimensions (this error does not involve 2D images with multiple channels),
    one should pass a structuring element for erosion which matches the number of dimensions of the input image (or image channels), or don't pass
    any structural element, in which case the default behaviour follows skimage.morphology.erosion (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.erosion).

    ========= ========= =========

    __init__ Inputs:
    - image: The input image to analyze, as a NumPy array.

    ========= ========= =========

    measure_obj_struct_tensor_eigenval Inputs:

    - label_image. The label image to analyze, as a NumPy array. If axis is None, label_image must have the same shape as image.
    If axis is not None, label_image must have the same shape as image channels.
    
    - min_sigma. int, float, tuple, list or None. Optional, default None. The minimum sigma value for the structure tensor. Ref to below for the options to be passed and default behaviour.
    
    - max_sigma. int, float, tuple, list or None. Optional, default None. The maximum sigma value for the structure tensor. Ref to below for the options to be passed and default behaviour.
    
    - num_sigma. int, float or None. Optional, default None. The number of sigma values to compute. Ref to below for the options to be passed and default behaviour.

    - struct_tens_kwargs. dict or None. Optional, default {} (empty dictionary). Keyword arguments for the structure tensor function. All parameters skimage.feature.structure_tensor's parameters (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.structure_tensor)
    EXCEPT FOR 'sigma', can be passed to this dictionary as key-value pairs. 'sigma' must be specified using min_sigma, max_sigma and num_sigma.

    - erosion_kwargs. dict or None. Optional, default {'footprint':disk(9)}. Keyword arguments for the erosion function. All parameters skimage.morphology.erosion's parameters (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.erosion)
    can be passed to this dictionary as key-value pairs. NOTE: the default setting is can only be used for eroding a 2D image. For eroding images with higher dimensions one should
    pass a structuring element which matches the number of dimensions of the input image (or image channels), or no structural element, in which case the default behaviour
    follows skimage.morphology.erosion (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.erosion).

    - regionprops_kwargs. dict or None. Optional, default {'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'], 'separator':'-'}. Keyword arguments for the regionprops function.
    All parameters skimage.measure.regionprops's parameters (https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops) can be passed to this dictionary as key-value pairs. These
    include properties to measure and extra_properties to measure.
    NOTE: if 'properties' is passed to regionprops_kwargs, it must include 'label'. NOTE: if 'separator' is passed to regionprops_kwargs and it is '_', the behavior of the function is not guaranteed (a warning is printed)

    - erosion_warning. bool. Optional, default True. If True, a warning will be issued if the erosion operation removes any labelled object in label_image.

    - renaming_kwargs. dict or None. Optional, default {'keep_column':('label'),'measurement_pos':0,'ch_pos':-1}. Keyword arguments used for renaming columns in the output DataFrame. NOTE: the list associated to 'keep_column'
    must contain 'label'.

    ========= ========= =========

    measure_object_anisotropy Inputs:
    
    NOTE: all but 'eigen_ratio_kwargs' are identical to measure_obj_struct_tensor_eigenval inputs. 'measure_obj_struct_tensor_eigenval' is roughly identical to 'struct_tens_kwargs'.

    - label_image. The label image to analyze, as a NumPy array. If axis is None, label_image must have the same shape as image.
    If axis is not None, label_image must have the same shape as image channels.
    
    - min_sigma. int, float, tuple, list or None. Optional, default None. The minimum sigma value for the structure tensor. Ref to below for the options to be passed and default behaviour.
    
    - max_sigma. int, float, tuple, list or None. Optional, default None. The maximum sigma value for the structure tensor. Ref to below for the options to be passed and default behaviour.
    
    - num_sigma. int, float or None. Optional, default None. The number of sigma values to compute. Ref to below for the options to be passed and default behaviour.

    - erosion_kwargs. dict or None. Optional, default {'footprint':disk(9)}. Keyword arguments for the erosion function. All parameters skimage.morphology.erosion's parameters (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.erosion)
    can be passed to this dictionary as key-value pairs. NOTE: the default setting is can only be used for eroding a 2D image. For eroding images with higher dimensions one should
    pass a structuring element which matches the number of dimensions of the input image (or image channels), or no structural element, in which case the default behaviour
    follows skimage.morphology.erosion (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.erosion).

    - regionprops_kwargs. dict or None. Optional, default {'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'], 'separator':'-'}. Keyword arguments for the regionprops function.
    All parameters skimage.measure.regionprops's parameters (https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops) can be passed to this dictionary as key-value pairs.
    These include properties to measure and extra_properties to measure. NOTE: if 'properties' is passed to regionprops_kwargs, it must include 'label'. NOTE: if 'separator' is passed to regionprops_kwargs and it is '_',
    the behavior of the function is not guaranteed (a warning is printed)

    - eigen_ratio_kwargs. dict or None. Optional, default {'eps':1.e-6, 'sep':'_'}. Keyword arguments for the structure_tensor_eigenval_ratio.
    NOTE: as the structure tensor and relative eigenvalues are calculated within the structure_tensor_eigenval_ratio function, their parameters must be specified using this function. Precisely,
    all skimage.feature.structure_tensor's parameters (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.structure_tensor)
    EXCEPT FOR 'sigma', can be passed to this dictionary as the following key-value pair: 'kwargs':{'structure_tensor_parameter_1':value, 'structure_tensor_parameter_2':value, ...}.
    ALSO NOTE: if 'eps' is passed to eigen_ratio_kwargs, it must be different than 0.

    - erosion_warning. bool. Optional, default True. If True, a warning will be issued if the erosion operation removes any labelled object in label_image.

    - renaming_kwargs. dict or None. Optional, default {'keep_column':('label'),'measurement_pos':0,'ch_pos':-1}. Keyword arguments used for renaming columns in the output DataFrame. NOTE: the list associated to 'keep_column'
    must contain 'label'.

    ========= ========= =========

    measure_obj_struct_tensor_eigenval Output:
    
    pd.DataFrame. The dataframe structure and layout matches regionprops output: rows correspond to labelled objects in label_image, columns correspond to measurements. Per each image channel, per each sigma, the
    properties indicated in regionprops_kwargs (using the 'properties' and 'extra_properties' parameters) are computed per each channel eigenvalue.
    
    If image has multiple channels, the channel index is indicated as the last number of the column name. It is separated from the rest of the column name by the string indicated in
    regionprops_kwarg['separator'].

    If multiple sigmas are measured, the sigma index in the specified range is the second last number in the column name and the channel index is the last number.
    The sigma index is returned instead of the actual sigma value as, as reported below, a number of formats are possible for sigma values including iterables and floats.

    ========= ========= =========

    measure_object_anisotropy Output:
    
    pd.DataFrame. The dataframe structure and layout matches regionprops output: rows correspond to labelled objects in label_image, columns correspond to measurements. Per each image channel, per each sigma, the
    properties indicated in regionprops_kwargs (using the 'properties' and 'extra_properties' parameters) are computed per each ratio of channel eigenvalue pair (see point 2 above for eigenvalue pairing).
    
    If image has multiple channels, the channel index is indicated as the last number of the column name. It is separated from the rest of the column name by the string indicated in
    regionprops_kwarg['separator'].

    If multiple sigmas are measured, the sigma index in the specified range is the second last number in the column name and the channel index is the last number.
    The sigma index is returned instead of the actual sigma value as, as reported below, a number of formats are possible for sigma values including iterables and floats.

    If the input image (or image channel) has more than 2 dimensions, the indeces of the eigenvalue pair for which the ratio is calculated are included in the column name. The numerator and denominator indeces are,
    respectively, the fourth and third to the last numbers in the column name.

    ========= ========= =========
    min_sigma, max_sigma and num_sigma behaviour (also ref to https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.structure_tensor):
    
    - num_sigma must be an int or float or None.

    - min_sigma and max_sigma can be:
        - both None. In in this case, a tuple of 1s per each dimension of the image is passed to skimage.feature.structure_tensor sigma. num_sigma is ignored.

        - one None and the second a single value (int or float). In this case a tuple with the indicated value (irrespective of whether it is min_sigma or max_sigma) per each dimension of the image is passed to
        skimage.feature.structure_tensor sigma. num_sigma is ignored.
         
          
        - one None and the second a sequence of values (list or tuple). In this case the sequence of values must match the number of dimensions of the image to process (or channel).
        The sequence is transformed to a tuple passed to skimage.feature.structure_tensor sigma. num_sigma is ignored.

        - both single values (int or float). In this case min_sigma must be smaller than max_sigma. The following behaviour depends on num_sigma:
            
            - if num_sigma is None. Measurements are calculated, independently for min_sigma and max_sigma. Each of the measurement follows the same rule of "one None and the second a single value (int or float)" (see above).
            The output dataframe will include both measurements. The sigma index in column names (see above) will be from min_sigma to max_sigma: 0 for min_sigma measurements and 1 for max_sigma measurements.

            - if num_sigma is not None. First the linspace between min_sigma and max_sigma with num_sigma number of steps is calculated (https://numpy.org/devdocs/reference/generated/numpy.linspace.html).
            Secondly, measurements are calculated, independently per each sigma in the linspace. Each of the measurement follows the same rule of "one None and the second a single value (int or float)" (see above).
            The output dataframe will include measurements for all sigmas in the linspace. The sigma index in column names (see above) will be from min_sigma to max_sigma: 0 is the index of min_sigma measurements
            and the highest index is the index for max_sigma measurements, with all progressively increasing indeces matching the progression in the linspace.

        - both sequences of values (list, tuple). In this case both min_sigma and max_sigma must have a number of values matching the number of dimensions of the image (or image channel). In addition,
        each i-th value in min_sigma must be smaller or equal to the corresponding i-th value in max_sigma. Finally, min_sigma and max_sigma must be different. The following behaviour depends on num_sigma:
        
            - if num_sigma is None. Measurements are calculated, independently for min_sigma and max_sigma. Each of the measurement follows the same rule of "one None and the second a sequence of values (list or tuple)"
            (see above). The output dataframe will include both measurements. The sigma index in column names (see above) will be from min_sigma to max_sigma: 0 for min_sigma measurements and 1 for max_sigma measurements.

            - if num_sigma is not None. Per each i-th pair in min_sigma and max_sigma, the linspace between i-th-min_sigma and i-th_max_sigma is calculated with num_sigma number as number of steps. This forms a matrix M
            with shape (num_sigma, image.ndim) where the first row is min_sigma and the last row is max_sigma. Measurements are calculated independently for each row R in M, with R being passed as sigma to
            skimage.feature.structure_tensor. The output dataframe will include measurements for all rows in M. The sigma index in column names (see above) will be from min_sigma to max_sigma: 0 is the index of min_sigma
            measurements and the highest index is the index for max_sigma measurements, with all progressively increasing indeces matching the progression in the rows of M from min_sigma to max_sigma.
    """

    def __init__(self, image:np.array):
        self.image = image

    # rename columns
    def rename_columns(self,
                       dataframe:pd.DataFrame,
                       prefix:str|None=None,
                       suffix:str|None=None,
                       keep_column:tuple|None=None,
                       pre_sep:str|None=None,
                       suf_sep:str|None=None)->pd.DataFrame:
        
        # use default pre_sep and suf_sep
        if pre_sep is None:
            pre_sep="_"
        if suf_sep is None:
            suf_sep="_"
        
        # initialize a dictionary to be used as mapper for column renaming
        mapper_col = {}
        
        # iterate through the columns of the dataframe
        for cl in dataframe.columns:

            # check if the column should be kept
            if keep_column!=None:
                if cl not in keep_column:

                    # rename the column
                    if prefix!=None and suffix!=None:
                        mapper_col[cl]=f"{prefix}{pre_sep}{cl}{suf_sep}{suffix}"
                    elif prefix!=None and suffix==None:
                        mapper_col[cl]=f"{prefix}{pre_sep}{cl}"
                    elif prefix==None and suffix!=None:
                        mapper_col[cl]=f"{cl}{suf_sep}{suffix}"
                    else:
                        mapper_col[cl]=cl
                
                # keep the column
                else:
                    mapper_col[cl]=cl
        
        # rename the columns of the dataframe
        renamed_dataframe = dataframe.rename(mapper_col,axis=1, copy=True)
        
        return renamed_dataframe


    def structure_tensor_eigenval_ratio(self,
                                        concat_axis:int=-1,
                                        eps:float=1.e-6,
                                        sep:str|None=None,
                                        return_eigen_pairs:bool=False,
                                        kwargs:dict|None=None)->np.array:
        """
        """

        # use default sep and kwargs
        if sep is None:
            sep='_'
        if kwargs is None:
            kwargs={}

        # ensure that no division by 0 is present
        assert eps>0, "eps must be different than 0"

        # calculate image structure tensors
        A_elems = structure_tensor(self.image, **kwargs)

        # calculate structure tensor eigenvalues
        eigen = structure_tensor_eigenvalues(A_elems)
        
        # initialize a collection list for the eigenvalues fractions
        eigen_ratio_list = []

        # initialize a collection list for tracking the pair of dimentions used for the ratio measurement
        eigen_ratio_dim_pair = []

        # initialize a positional counter
        d1 = 0

        # iterate through the eigenvalue orders
        for e1 in eigen:

            # initialize a second positional counter
            d2 = d1+1

            # iterate through the rest of the eigenvalue orders
            for e2 in eigen[d2:]:

                # avoid a 0 division
                e2 = e2 + eps

                # divide the eigenvalue
                e12_ratio = e1/e2

                # collect the result of eigenvalue division
                eigen_ratio_list.append(e12_ratio)

                # collect the pair of dimensions used for the ratio measurement
                eigen_ratio_dim_pair.append(f"{d1+1}{sep}{d2+1}")

                # update the second positional counter
                d2=d2+1
            
            # update the first positional counter
            d1=d1+1
        
        # if image has more than 2 dimensions, transform result in a concatenated array
        if len(eigen_ratio_list)>1:
            eigen_ratio = np.stack(eigen_ratio_list, axis=concat_axis)
        else:
            eigen_ratio = eigen_ratio_list[0]
        
        if return_eigen_pairs:
            return eigen_ratio, eigen_ratio_dim_pair
        
        else:
            return eigen_ratio


    def image_mean_anisotropy(self, eps:float=1.e-6, kwargs:dict|None=None) -> float:
        
        """
        """
        # use default kwargs
        if kwargs is None:
            kwargs={}
        
        # calculate eigenvalue ratio
        eigen_ratio = self.structure_tensor_eigenval_ratio(eps, **kwargs)

        return np.mean(eigen_ratio)


    def measure_obj_struct_tensor_eigenval_single_image_sigma(self,
                                                        label_image:np.array,
                                                        struct_tens_kwargs:dict|None=None,
                                                        erosion_kwargs:dict|None=None,
                                                        regionprops_kwargs:dict|None=None,
                                                        erosion_warning:bool=True)-> pd.DataFrame:
        """
        """

        # use default kwargs settings
        if struct_tens_kwargs is None:
            struct_tens_kwargs={}
        
        if erosion_kwargs is None:
            erosion_kwargs={'footprint':disk(9)}
        
        if regionprops_kwargs is None:
            regionprops_kwargs={'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std']}

        assert 'properties' in regionprops_kwargs, "'properties' must be passed to regionpros_kwargs"
        assert 'label' in regionprops_kwargs['properties'], "'label' must be a property passed to 'properties' in regionprops_kwargs"
        if 'sigma' in struct_tens_kwargs:
            assert len(struct_tens_kwargs['sigma'])==len(self.image.shape), "only one sigma value can be passed to the present function"

        # calculate image structure tensors
        A_elems = structure_tensor(self.image, **struct_tens_kwargs)

        # calculate structure tensor eigenvalues
        eigenvals = structure_tensor_eigenvalues(A_elems)

        # erode labels
        eroded_label_image = erosion(label_image, **erosion_kwargs)
        
        # check for cell loss due to erosion if erosion_warning is set to True
        if erosion_warning:
            
            # get the number of labelled objects before erosion
            labels_before = np.unique(label_image)

            # get the number of labelled objects after erosion
            labels_after = np.unique(eroded_label_image)

            # print a warning message if the number of labels before and after erosion don't correspond
            if labels_before.shape != labels_after.shape:
                print(f"WARNING: possible loss of labelled object due to erosion. Number of objects before erosion: {labels_before[0]-1}. Number of objects after erosion: {labels_after[0]-1}")
        
        # measure intensity of first structure tensor eigenvalue
        eigen_measurement_i = pd.DataFrame(regionprops_table(eroded_label_image,
                                                             intensity_image=eigenvals[0],
                                                             **regionprops_kwargs))
        
        # rename columns
        eigen_measurement = self.rename_columns(eigen_measurement_i,
                                                prefix=f"str_tens_eigv_1",
                                                keep_column=('label'))

        # copy the measurement dataframe - NOTE: this is a step which can potentially be cleaned
        glob_eigen_measurement = eigen_measurement.copy()
        
        # initialize a counter of eigenvalue orders
        o = 2

        # iterate through the eigenvalues (starting from the second as the first has been measured)
        for eigenval in eigenvals[1:]:

            # measure intensity of structure tensor eigenvalue
            eigen_measurement_iii = pd.DataFrame(regionprops_table(eroded_label_image,
                                                                intensity_image=eigenval,
                                                                **regionprops_kwargs))
        
            # rename columns
            eigen_measurement_ii = self.rename_columns(eigen_measurement_iii,
                                                       prefix=f"str_tens_eigv_{o}",
                                                       keep_column=('label'))
            
            # merge measurements for the o-th eigenvalue to the global measurements dataframe - store the merged dataframe in a new dataframe (NOTE: this is a step which can potentially be cleaned)
            eigen_measurement_i = glob_eigen_measurement.merge(eigen_measurement_ii,
                                                             on='label',
                                                             how='left',
                                                             copy=True)

            # update the global measurements dataframe - NOTE: this is a step which can potentially be cleaned
            glob_eigen_measurement = eigen_measurement_i

            # update the counter of eigenvalue orders 
            o = o +1
        
        return glob_eigen_measurement


    def measure_object_anisotropy_single_image_sigma(self,
                                               label_image:np.array,
                                               erosion_kwargs:dict|None=None,
                                               regionprops_kwargs:dict|None=None,
                                               eigen_ratio_kwargs:dict|None=None,
                                               erosion_warning:bool=True)-> pd.DataFrame:
        """
        """

        # use default kwargs
        if erosion_kwargs is None:
            erosion_kwargs={'footprint':disk(9)}
        
        if regionprops_kwargs is None:
            regionprops_kwargs={'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'],
                                'separator':'-'}
        
        if eigen_ratio_kwargs is None:
            eigen_ratio_kwargs={'eps':1.e-6, 'sep':'_'}

        assert 'properties' in regionprops_kwargs, "'properties' must be passed to regionprops_kwargs"
        assert 'label' in regionprops_kwargs['properties'], "'label' must be a property passed to 'properties' in regionprops_kwargs"
        assert 'concat_axis' not in eigen_ratio_kwargs, "'concat_axis' should not be passed to eigen_ratio_kwargs as it is hard coded in the last position for the regionprops-based measurement"
        assert 'return_eigen_pairs' not in eigen_ratio_kwargs, "'return_eigen_pairs' should not be passed to eigen_ratio_kwargs as it is hard coded to be True"
        if 'separator' in regionprops_kwargs:
           if regionprops_kwargs['separator']=='_':
               print("WARNING: using '_' as separator for regionprops can lead to ambiguous column names and potential error when renaming columns")
        else:
            regionprops_kwargs = regionprops_kwargs.copy() # to avoid modifying the input dictionary
            regionprops_kwargs['separator']='-'

        if 'sigma' in eigen_ratio_kwargs['kwargs']:
            assert len(eigen_ratio_kwargs['kwargs']['sigma'])==len(self.image.shape), "only one sigma value can be passed to the present function"

        # calculate eigenvalue ratio
        # NOTE: 1) in this case concat_axis is hard coded in the last position as it is required for the following regionprops-based measurement
        # 2) return_eigen_pairs is hard coded to be True as it is required for renaming the columns
        eigen_ratio, eigen_ratio_pair = self.structure_tensor_eigenval_ratio(concat_axis=-1, return_eigen_pairs=True, **eigen_ratio_kwargs)

        # erode labels
        eroded_label_image = erosion(label_image, **erosion_kwargs)
        
        # check for cell loss due to erosion if erosion_warning is set to True
        if erosion_warning:
            
            # get the number of labelled objects before erosion
            labels_before = np.unique(label_image)

            # get the number of labelled objects after erosion
            labels_after = np.unique(eroded_label_image)

            # print a warning message if the number of labels before and after erosion don't correspond
            if labels_before.shape != labels_after.shape:
                print(f"WARNING: possible loss of labelled object due to erosion. Number of objects before erosion: {labels_before[0]-1}. Number of objects after erosion: {labels_after[0]-1}")
        
        # measure intensity of structure tensor eigenvalue ratio
        eigen_ratio_measurement = pd.DataFrame(regionprops_table(eroded_label_image,
                                                                intensity_image=eigen_ratio,
                                                                **regionprops_kwargs))

        # add 'anisotropy' to measurement column names
        eigen_ratio_measurement = self.rename_columns(eigen_ratio_measurement,
                                                       prefix="anisotropy",
                                                       keep_column=('label'))
        
        # add the pair of eigenvalues used for ratio calculation to the column names

        # initialize a dictionary to be used as a column name mapper
        cl_name_mapper = {}

        # iterate through the columns of the measurement DataFrame
        for cl in eigen_ratio_measurement.columns:

            # check if column is not 'label'
            if cl != 'label':

                # split column name to separate the channel from the measurement
                cl_name_split = cl.split(regionprops_kwargs['separator'])

                # check if channel number is valid
                try:
                    channel_number = int(cl_name_split[-1])

                    # get the pair of eigenvalues used for ratio calculation
                    eigenval_pair = eigen_ratio_pair[channel_number]

                    # update column name mapper with new eigenvalue pair
                    cl_name_mapper[cl] = f"{cl_name_split[0]}{eigen_ratio_kwargs['sep']}{eigenval_pair}"
                except:
                    cl_name_mapper[cl] = cl
            else:
                cl_name_mapper[cl] = cl

        # rename columns in eigen_ratio_measurement
        eigen_ratio_measurement = eigen_ratio_measurement.rename(columns=cl_name_mapper)

        return eigen_ratio_measurement


    def measure_obj_struct_tensor_eigenval_single_sigma(self,
                                           label_image:np.array,
                                           axis:int|None=None,
                                           struct_tens_kwargs:dict|None=None,
                                           erosion_kwargs:dict|None=None,
                                           regionprops_kwargs:dict|None=None,
                                           erosion_warning:bool=True)->pd.DataFrame:
        """
        """

        # use default kwargs
        if struct_tens_kwargs is None:
            struct_tens_kwargs={}
        
        if erosion_kwargs is None:
            erosion_kwargs={'footprint':disk(9)}
        
        if regionprops_kwargs is None:
            regionprops_kwargs={'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'],
                                'separator':'-'}

        if 'separator' in regionprops_kwargs:
           if regionprops_kwargs['separator']=='_':
               print("WARNING: using '_' as separator for regionprops can lead to ambiguous column names and potential error when renaming columns")
        else:
            regionprops_kwargs = regionprops_kwargs.copy() # to avoid modifying the input dictionary
            regionprops_kwargs['separator']='-'

        if axis==None:
            if 'sigma' in struct_tens_kwargs:
                assert len(struct_tens_kwargs['sigma'])==len(self.image.shape), "only one sigma value can be passed to the present function"

            # measure structure tensor eigenvalues
            eigen_measurement = self.measure_obj_struct_tensor_eigenval_single_image_sigma(label_image,
                                                                                     struct_tens_kwargs=struct_tens_kwargs,
                                                                                     erosion_kwargs=erosion_kwargs,
                                                                                     regionprops_kwargs=regionprops_kwargs,
                                                                                     erosion_warning=erosion_warning)
            return eigen_measurement

        else:
            
            # redefine measure_obj_struct_tensor_eigenval_single_image_sigma
            # this is the same function as above, it accepts an image different than self.image
            def measure__obj__struct__tensor__eigenval__single__image__sigma(imag_e:np.array,
                                                                      label_imag_e:np.array,
                                                                      suffix:int|None=None,
                                                                      suf_sep:str|None=None)-> pd.DataFrame:
                """
                """

                # use default suf_sep
                if suf_sep is None:
                    suf_sep="_"
                
                assert 'properties' in regionprops_kwargs, "'properties' must be passed to regionpros_kwargs"
                assert 'label' in regionprops_kwargs['properties'], "'label' must be a property passed to 'properties' in regionprops_kwargs"

                # calculate image structure tensors
                A_elems = structure_tensor(imag_e, **struct_tens_kwargs)

                # calculate structure tensor eigenvalues
                eigenvals = structure_tensor_eigenvalues(A_elems)
                
                # measure intensity of first structure tensor eigenvalue
                eigen_measurement_i = pd.DataFrame(regionprops_table(label_imag_e,
                                                                    intensity_image=eigenvals[0],
                                                                    **regionprops_kwargs))
                
                # rename columns
                eigen_measurement = self.rename_columns(eigen_measurement_i,
                                                        prefix=f"str_tens_eigv_1",
                                                        suffix=f"{suffix}",
                                                        keep_column=('label'),
                                                        suf_sep=suf_sep)

                # copy the measurement dataframe - NOTE: this is a step which can potentially be cleaned
                glob_eigen_measurement = eigen_measurement.copy()
                
                # initialize a counter of eigenvalue orders
                o = 2

                # iterate through the eigenvalues (starting from the second as the first has been measured)
                for eigenval in eigenvals[1:]:

                    # measure intensity of structure tensor eigenvalue
                    eigen_measurement_iii = pd.DataFrame(regionprops_table(eroded_label_image,
                                                                        intensity_image=eigenval,
                                                                        **regionprops_kwargs))
                
                    # rename columns
                    eigen_measurement_ii = self.rename_columns(eigen_measurement_iii,
                                                            prefix=f"str_tens_eigv_{o}",
                                                            suffix=f"{suffix}",
                                                            keep_column=('label'),
                                                            suf_sep=suf_sep)
                    
                    # merge measurements for the o-th eigenvalue to the global measurements dataframe - store the merged dataframe in a new dataframe (NOTE: this is a step which can potentially be cleaned)
                    eigen_measurement_i = glob_eigen_measurement.merge(eigen_measurement_ii,
                                                                    on='label',
                                                                    how='left',
                                                                    copy=True)

                    # update the global measurements dataframe - NOTE: this is a step which can potentially be cleaned
                    glob_eigen_measurement = eigen_measurement_i

                    # update the counter of eigenvalue orders 
                    o = o +1
                
                return glob_eigen_measurement

            # erode labels
            eroded_label_image = erosion(label_image, **erosion_kwargs)
            
            # check for cell loss due to erosion if erosion_warning is set to True
            if erosion_warning:
                
                # get the number of labelled objects before erosion
                labels_before = np.unique(label_image)

                # get the number of labelled objects after erosion
                labels_after = np.unique(eroded_label_image)

                # print a warning message if the number of labels before and after erosion don't correspond
                if labels_before.shape != labels_after.shape:
                    print(f"WARNING: possible loss of labelled object due to erosion. Number of objects before erosion: {labels_before[0]-1}. Number of objects after erosion: {labels_after[0]-1}")
            
            # unstack self.image on the indicated axis
            unstacked_image = [np.moveaxis(a,axis,-1)[...,0] for a in np.split(self.image, indices_or_sections=self.image.shape[axis], axis=axis)]
            
            if 'sigma' in struct_tens_kwargs:
                assert len(struct_tens_kwargs['sigma'])==len(unstacked_image[0].shape), "only one sigma value can be passed to the present function"

            # measure the intensity of the structure tensor eigenvalues for the first image
            eigen_measurement = measure__obj__struct__tensor__eigenval__single__image__sigma(unstacked_image[0],
                                                                                      eroded_label_image,
                                                                                      suffix=0,
                                                                                      suf_sep=regionprops_kwargs['separator'])

            # copy the measurement dataframe - NOTE: this is a step which can potentially be cleaned
            glob_eigen_measurement = eigen_measurement.copy()

            # iterate through unstacked images
            for i, img in enumerate(unstacked_image[1:]):
                
                # measure the intensity of the structure tensor eigenvalues for the first image
                eigen_measurement_ii = measure__obj__struct__tensor__eigenval__single__image__sigma(img,
                                                                                             eroded_label_image,
                                                                                             suffix=i+1,
                                                                                             suf_sep=regionprops_kwargs['separator'])

                # merge measurements of the i-th image with the global measurements dataframe - store the merged dataframe in a new dataframe (NOTE: this is a step which can potentially be cleaned)
                eigen_measurement_i = glob_eigen_measurement.merge(eigen_measurement_ii,on='label',
                                                                   how='left',
                                                                   copy=True)
                
                # update the global measurements dataframe - NOTE: this is a step which can potentially be cleaned
                glob_eigen_measurement = eigen_measurement_i
                
            return glob_eigen_measurement
    
    def measure_object_anisotropy_single_sigma(self,
                                  label_image:np.array,
                                  axis:int|None=None,
                                  erosion_kwargs:dict|None=None,
                                  regionprops_kwargs:dict|None=None,
                                  eigen_ratio_kwargs:dict|None=None,
                                  erosion_warning:bool=True)-> pd.DataFrame:
        """
        """

        # set default kwargs
        if erosion_kwargs is None:
            erosion_kwargs={'footprint':disk(9)}
        
        if regionprops_kwargs is None:
            regionprops_kwargs={'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'], 'separator':'-'}
        
        if eigen_ratio_kwargs is None:
            eigen_ratio_kwargs:dict={'eps':1.e-6, 'sep':'_'}

        if axis==None:
            if 'sigma' in eigen_ratio_kwargs['kwargs']:
                assert len(eigen_ratio_kwargs['kwargs']['sigma'])==len(self.image.shape), "only one sigma value can be passed to the present function"

            # measure eigenvalues ratio without iterating over the axis
            eigen_ratio_measurement = self.measure_object_anisotropy_single_image_sigma(label_image=label_image,
                                                                                  erosion_kwargs=erosion_kwargs,
                                                                                  regionprops_kwargs=regionprops_kwargs,
                                                                                  eigen_ratio_kwargs=eigen_ratio_kwargs,
                                                                                  erosion_warning=erosion_warning)
            return eigen_ratio_measurement
        
        else:
            def structure__tensor__eigenval__ratio(imag_e:np.array,
                                                   concat_axis:int=-1,
                                                   eps:float=1.e-6,
                                                   sep:str|None=None,
                                                   return_eigen_pairs:bool=False,
                                                   kwargs:dict|None=None)->np.array:
                """
                """

                # set default sep and kwargs
                if sep is None:
                    sep = "_"
                
                if kwargs is None:
                    kwargs={}

                # calculate image structure tensors
                A_elems = structure_tensor(imag_e, **kwargs)

                # calculate structure tensor eigenvalues
                eigen = structure_tensor_eigenvalues(A_elems)

                # initialize a collection list for the eigenvalues fractions
                eigen_ratio_list = []

                # initialize a collection list for tracking the pair of dimentions used for the ratio measurement
                eigen_ratio_dim_pair = []

                # initialize a positional counter
                d1 = 0

                # iterate through the eigenvalue orders
                for e1 in eigen:

                    # initialize a second positional counter
                    d2 = d1+1

                    # iterate through the rest of the eigenvalue orders
                    for e2 in eigen[d2:]:

                        # avoid a 0 division
                        e2 = e2 + eps

                        # divide the eigenvalue
                        e12_ratio = e1/e2

                        # collect the result of eigenvalue division
                        eigen_ratio_list.append(e12_ratio)

                        # collect the pair of dimensions used for the ratio measurement
                        eigen_ratio_dim_pair.append(f"{d1+1}{sep}{d2+1}")

                        # update the second positional counter
                        d2=d2+1
                    
                    # update the first positional counter
                    d1=d1+1

                # if image has more than 2 dimensions, transform result in a concatenated array
                if len(eigen_ratio_list)>1:
                    eigen_ratio = np.stack(eigen_ratio_list, axis=concat_axis)
                else:
                    eigen_ratio = eigen_ratio_list[0]
                
                if return_eigen_pairs:
                    return eigen_ratio, eigen_ratio_dim_pair
                
                else:
                    return eigen_ratio


            def measure__object__anisotropy__single__image__sigma(imag_e:np.array,
                                                           label_imag_e:np.array,
                                                           suffix:int|None=None)-> pd.DataFrame:
                """
                """

                # calculate eigenvalue ratio
                # NOTE: 1) in this case concat_axis is hard coded in the last position as it is required for the following regionprops-based measurement
                # 2) return_eigen_pairs is hard coded to be True as it is required for renaming the columns
                eigen_ratio, eigen_ratio_pair = structure__tensor__eigenval__ratio(imag_e,
                                                                                   concat_axis=-1,
                                                                                   return_eigen_pairs=True,
                                                                                   **eigen_ratio_kwargs)
                
                # measure intensity of structure tensor eigenvalue ratio
                eigen_ratio_measurement = pd.DataFrame(regionprops_table(label_imag_e,
                                                                        intensity_image=eigen_ratio,
                                                                        **regionprops_kwargs))

                # add 'anisotropy' to measurement column names
                eigen_ratio_measurement = self.rename_columns(eigen_ratio_measurement,
                                                            prefix="anisotropy",
                                                            keep_column=('label'))
                
                # add the pair of eigenvalues used for ratio calculation to the column names

                # initialize a dictionary to be used as a column name mapper
                cl_name_mapper = {}

                # iterate through the columns of the measurement DataFrame
                for cl in eigen_ratio_measurement.columns:

                    # check if column is not 'label'
                    if cl != 'label':

                        # split column name to separate the channel from the measurement
                        cl_name_split = cl.split(regionprops_kwargs['separator'])

                        # check if channel number is valid
                        try:
                            channel_number = int(cl_name_split[-1])

                            # get the pair of eigenvalues used for ratio calculation
                            eigenval_pair = eigen_ratio_pair[channel_number]

                            # update column name mapper with new eigenvalue pair
                            # NOTE the difference with measure_object_anisotropy_single_image_sigma:
                            # 1) the separator for the measurement and the eigenval_pair is now "_" fixed
                            # 2) regionprops_kwargs['separator'] used to separate the suffix
                            cl_name_mapper[cl] = f"{cl_name_split[0]}_{eigenval_pair}{regionprops_kwargs['separator']}{suffix}"
                        except:
                            # NOTE the difference with measure_object_anisotropy_single_image_sigma:
                            # 1) regionprops_kwargs['separator'] used to separate the suffix
                            cl_name_mapper[cl] = f"{cl}{regionprops_kwargs['separator']}{suffix}"
                    else:
                        # NOTE the difference with measure_object_anisotropy_single_image_sigma:
                        # 1) regionprops_kwargs['separator'] used to separate the suffix
                        cl_name_mapper[cl] = cl

                # rename columns in eigen_ratio_measurement
                eigen_ratio_measurement = eigen_ratio_measurement.rename(columns=cl_name_mapper)

                return eigen_ratio_measurement

            # erode labels
            eroded_label_image = erosion(label_image, **erosion_kwargs)
            
            # check for cell loss due to erosion if erosion_warning is set to True
            if erosion_warning:
                
                # get the number of labelled objects before erosion
                labels_before = np.unique(label_image)

                # get the number of labelled objects after erosion
                labels_after = np.unique(eroded_label_image)

                # print a warning message if the number of labels before and after erosion don't correspond
                if labels_before.shape != labels_after.shape:
                    print(f"WARNING: possible loss of labelled object due to erosion. Number of objects before erosion: {labels_before[0]-1}. Number of objects after erosion: {labels_after[0]-1}")
            
            # unstack self.image on the indicated axis
            unstacked_image = [np.moveaxis(a,axis,-1)[...,0] for a in np.split(self.image, indices_or_sections=self.image.shape[axis], axis=axis)]

            if 'sigma' in eigen_ratio_kwargs['kwargs']:
                assert len(eigen_ratio_kwargs['kwargs']['sigma'])==len(unstacked_image[0].shape), "only one sigma value can be passed to the present function"


            # measure the intensity of the structure tensor eigenvalues for the first image
            eigen_ratio_measurement = measure__object__anisotropy__single__image__sigma(unstacked_image[0],
                                                                                 eroded_label_image,
                                                                                 suffix=0)
            
            # copy the measurement dataframe - NOTE: this is a step which can potentially be cleaned
            glob_eigen_ratio_measurement = eigen_ratio_measurement.copy()

            # iterate through unstacked images
            for i, img in enumerate(unstacked_image[1:]):
                
                # measure the intensity of the structure tensor eigenvalues for the first image
                eigen_ratio_measurement_ii = measure__object__anisotropy__single__image__sigma(img,
                                                                                        eroded_label_image,
                                                                                        suffix=i+1)

                # merge measurements of the i-th image with the global measurements dataframe - store the merged dataframe in a new dataframe (NOTE: this is a step which can potentially be cleaned)
                eigen_ratio_measurement_i = glob_eigen_ratio_measurement.merge(eigen_ratio_measurement_ii,
                                                                               on='label',
                                                                               how='left',
                                                                               copy=True)
                
                # update the global measurements dataframe - NOTE: this is a step which can potentially be cleaned
                glob_eigen_ratio_measurement = eigen_ratio_measurement_i
                
            return glob_eigen_ratio_measurement
    
    def generate_sigma_collection(self,
                                  image:np.array,
                                  min_sigma:int|float|tuple|list|None=None,
                                  max_sigma:int|float|tuple|list|None=None,
                                  num_sigma:int|float|None=None)->list:
        """
        sigma can't be passed to struct_tens_kwargs.

        num_sigma must be an int or float or None.

        min_sigma and max_sigma can be:
        - both None. In which case, a kernel of size 1 per each dimension of the image is used. num_sigma is ignored.

        - one None and the second a single value (int or float) or a sequence of values (list or tuple) with length equal
        to the number of dimensions of the image. In which case the non-None value is used as the sigma value and
        num_sigma is ignored. If single value, the value is used on all the dimensions of the image.

        - both single values (int or float). In this case...
            - if num_sigma is None, the function is calculated for the two sigmas indicated. Precisely first
            for a kernel of size min_sigma in all the dimensions of the image, secondly for a kernel of size max_sigma in
            all the dimensions of the image.

            - if num_sigma is not None, the function is calculated for the linspace between min_sigma and max_sigma, divided by
            num_sigma. Meaning that a kernel of size S per each image dimension is used for each S-value in the linspace.

        - both sequences of values (list, tuple), both with the same length as the number of dimensions of the image. In this case...
            - the i-th position in min_sigma and max_sigma indicates the i-th dimension in the kernel used.    
        
            - if num_sigma is None, the function is calculated for the two sigmas indicated. First for a kernel of shape min_sigma.
            Secondly for a kernel of shape max_sigma.

            - if num_sigma is not None, N kernels are calculated with N corresponding to num_sigma. Per each i value in
            min_sigma and corresponding i value in max_sigma, a N long linspace is created in the i-min_sigma - i-max_sigma range.
            Each of N-th kernel is generated by taking the N-th value per each of i-th linspace.
        """

        # if no indication is provided for min_sigma and max_sigma
        if min_sigma ==None and max_sigma==None:
            
            # use a tuple the same length of image shape and size 1 per each element, if no axis is provided
            output_sigmas=[tuple([1 for d in image.shape])]

        # if only min_sigma is provided
        elif min_sigma!=None and max_sigma==None:

            # if a single value is provided
            if isinstance(min_sigma,int) or isinstance(min_sigma,float):

                # use a tuple the same length of image shape and size min_sigma per each element, if no axis is provided
                output_sigmas=[tuple([min_sigma for d in image.shape])]
            
            # if an iterable is provided, use it as it is
            elif (isinstance(min_sigma, list) or isinstance(min_sigma, tuple)):
                
                # make sure that size provided by min_sigma matches the number of sizes of image
                assert len(min_sigma)==len(image.shape), "if the only parameter, min_sigma must have the same number of dimension of the image to process"
                output_sigmas=[tuple(min_sigma)]
            
            else:
                print("min_sigma must be an int, a float, a tuple or a list")
                raise TypeError

        # if only max_sigma is provided
        elif min_sigma==None and max_sigma!=None:

            # if a single value is provided
            if (isinstance(max_sigma,int) or isinstance(max_sigma, float)):

                # use a tuple the same length of image shape and size min_sigma per each element, if no axis is provided
                output_sigmas=[tuple([max_sigma for d in image.shape])]
            
            # if an iterable is provided, use it as it is
            elif (isinstance(max_sigma, list) or isinstance(max_sigma, tuple)):

                # make sure that size provided by max_sigma matches the number of sizes of image
                assert len(max_sigma)==len(image.shape), "if the only parameter, max_sigma must have the same number of dimension of the image to process"
                
                output_sigmas=[tuple(max_sigma)]
            else:
                print("max_sigma must be an int, a float, a tuple or a list")
                raise TypeError
        
        # if both min_sigma ad max_sigma are provided
        else:

            # make sure that min_sigma and max_sigma are both single values or both iterables
            assert (((isinstance(min_sigma,int) or isinstance(min_sigma,float)) and (isinstance(max_sigma,int) or isinstance(max_sigma,float))) or ((isinstance(min_sigma, list) or isinstance(min_sigma, tuple)) and (isinstance(max_sigma, list) or isinstance(max_sigma, tuple)))), "min_sigma and max_sigma must either be both single numbers (int or float) or both iterables (tuple or list), but not mixed"

            # if no num_sigma is provided
            if num_sigma==None:
                
                # if min_sigma and max_sigma are single numbers
                if ((isinstance(min_sigma,int) or isinstance(min_sigma,float)) and (isinstance(max_sigma,int) or isinstance(max_sigma,float))):
                    assert min_sigma<max_sigma, "min_sigma must be smaller than max_sigma"
                    
                    # use a tuple with the same length of image shape and size min_sigma per each element
                    # and a tuple with the same length of image shape and size max_sigma per each element
                    min_sigma_shape = tuple([min_sigma for d in image.shape])
                    max_sigma_shape = tuple([max_sigma for d in image.shape]) 
                    output_sigmas=[min_sigma_shape, max_sigma_shape]
                
                # if they are iterables
                else:
                    assert min_sigma!=max_sigma, "min_sigma must be different than max_sigma"

                    assert len(min_sigma)==len(image.shape), "if iterables, min_sigma and max_sigma must have the same number of dimensions as the image to process"
                    assert len(max_sigma)==len(image.shape), "if iterables, min_sigma and max_sigma must have the same number of dimensions as the image to process"

                    # ensure that per each dimension, min_sigma is smaller or equal to max_sigma
                    for j,min_s in enumerate(min_sigma):
                        
                        # ensure that per each dimension, min_sigma is smaller or equal to max_sigma
                        assert min_s<=max_sigma[j], f"min_sigma must be equal or smaller than max_sigma at position {j}"
                    
                    output_sigmas=[tuple(min_sigma), tuple(max_sigma)]
                
            else:

                # if min_sigma and max_sigma are single numbers
                if (((isinstance(min_sigma,int) or isinstance(min_sigma,float)) and (isinstance(max_sigma,int) or isinstance(max_sigma,float)))):
                    
                    assert min_sigma<max_sigma, "min_sigma must be smaller than max_sigma"
                    
                    # create a linspace of sigmas
                    sigma_linspace = np.linspace(min_sigma, max_sigma, num_sigma)

                    # initialize the list containing the sigma values
                    output_sigmas=[]

                    # iterate through the sigma linspace
                    for s in sigma_linspace:

                        # use a tuple with the same length of image shape and size s per each element
                        output_sigmas.append(tuple([s for i in image.shape]))
                        
                else:
                    assert len(min_sigma)==len(max_sigma), "if iterables, min_sigma and max_sigma mus't have the same number of dimensions"
                    assert len(min_sigma)==len(image.shape), "if iterables, min_sigma and max_sigma must have the same number of dimensions as the image to process"
                    assert min_sigma!=max_sigma, "min_sigma must be different than max_sigma"

                    # create a list to collect sigma' linspaces
                    sigma_collection_i = []

                    # iterate through the dimensions of the image
                    for i, d in enumerate(image.shape):
                        
                        # ensure that per each dimension, min_sigma is smaller or equal to max_sigma
                        assert min_sigma[i]<=max_sigma[i], f"min_sigma must be equal or smaller than max_sigma at position {i}"

                        # create a linspace for the i-th dimension using the corresponding min and max sigma
                        sigma_collection_i.append(np.linspace(min_sigma[i], max_sigma[i], num_sigma))
                    
                    # transpose the collection of sigma linspaces and transform it into a list of lists
                    sigma_collection = list(map(list, zip(*sigma_collection_i)))
                    output_sigmas=sigma_collection

        return output_sigmas


    def measure_obj_struct_tensor_eigenval(self,
                                           label_image:np.array,
                                           min_sigma:int|float|tuple|list|None=None,
                                           max_sigma:int|float|tuple|list|None=None,
                                           num_sigma:int|float|None=None,
                                           axis:int|None=None,
                                           struct_tens_kwargs:dict|None=None,
                                           erosion_kwargs:dict|None=None,
                                           regionprops_kwargs:dict|None=None,
                                           erosion_warning:bool=True,
                                           renaming_kwargs:dict|None=None) -> pd.DataFrame:
        """
        sigma can't be passed to struct_tens_kwargs.

        num_sigma must be an int or float or None.

        min_sigma and max_sigma can be:
        - both None. In which case, a kernel of size 1 per each dimension of the image is used. num_sigma is ignored.

        - one None and the second a single value (int or float) or a sequence of values (list or tuple) with length equal
        to the number of dimensions of the image. In which case the non-None value is used as the sigma value and
        num_sigma is ignored. If single value, the value is used on all the dimensions of the image.

        - both single values (int or float). In this case...
            - if num_sigma is None, the function is calculated for the two sigmas indicated. Precisely first
            for a kernel of size min_sigma in all the dimensions of the image, secondly for a kernel of size max_sigma in
            all the dimensions of the image.

            - if num_sigma is not None, the function is calculated for the linspace between min_sigma and max_sigma, divided by
            num_sigma. Meaning that a kernel of size S per each image dimension is used for each S-value in the linspace.

        - both sequences of values (list, tuple), both with the same length as the number of dimensions of the image. In this case...
            - the i-th position in min_sigma and max_sigma indicates the i-th dimension in the kernel used.    
            
            - if num_sigma is None, the function is calculated for the two sigmas indicated. First for a kernel of shape min_sigma.
            Secondly for a kernel of shape max_sigma.

            - if num_sigma is not None, N kernels are calculated with N corresponding to num_sigma. Per each i value in
            min_sigma and corresponding i value in max_sigma, a N long linspace is created in the i-min_sigma - i-max_sigma range.
            Each of N-th kernel is generated by taking the N-th value per each of i-th linspace.
        """

        # set default kwargs
        if struct_tens_kwargs is None:
            struct_tens_kwargs={}
        
        if erosion_kwargs is None:
            erosion_kwargs={'footprint':disk(9)}
        
        if regionprops_kwargs is None:
            regionprops_kwargs={'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'], 'separator':'-'}
        
        if renaming_kwargs is None:
            renaming_kwargs={'keep_column':('label'),'measurement_pos':0,'ch_pos':-1}

        # ensure that 'sigma' is not passed to struct_tens_kwargs
        assert 'sigma' not in struct_tens_kwargs, "sigma can't be passed to struct_tens_kwargs. Use min_sigma, max_sigma and num_sigma instead"
        
        # ensure that 'label is in rename_kwargs['keep_column']
        assert 'label' in renaming_kwargs['keep_column'], "'label' must be in renaming_kwargs['keep_column']"

        if 'separator' in regionprops_kwargs:
           if regionprops_kwargs['separator']=='_':
               print("WARNING: using '_' as separator for regionprops can lead to ambiguous column names and potential error when renaming columns")
        else:
            regionprops_kwargs = regionprops_kwargs.copy() # to avoid modifying the input dictionary
            regionprops_kwargs['separator']='-'

        # check that 'label' is in regionprops_kwargs['properties']
        if 'properties' in regionprops_kwargs:
            assert 'label' in regionprops_kwargs['properties'], "'label' must be in regionprops_kwargs['properties']"

        # unstack image if axis is provided
        if axis!=None:
            unstacked_image = [np.moveaxis(a,axis,-1)[...,0] for a in np.split(self.image, indices_or_sections=self.image.shape[axis], axis=axis)]
            
            assert len(unstacked_image[0].shape)==len(label_image.shape), "if axis is provided, the sub-image and label_image must have the same number of dimensions"

            # generate a collection of sigmas
            sigma_collection =  self.generate_sigma_collection(image=unstacked_image[0],
                                                               min_sigma=min_sigma,
                                                               max_sigma=max_sigma,
                                                               num_sigma=num_sigma)
        
        else:
            assert len(self.image.shape)==len(label_image.shape), "if no axis is provided, image and label_image must have the same number of dimensions"

            sigma_collection =  self.generate_sigma_collection(image=self.image,
                                                               min_sigma=min_sigma,
                                                               max_sigma=max_sigma,
                                                               num_sigma=num_sigma)

        # initialize a collection list
        glob_eigen_measurement_list = []

        # iterate through the sigma collection
        for sigma in sigma_collection:

            # add sigma to struct_tens_kwargs
            struct_tens_kwargs['sigma'] = sigma
            sigma_eigen_measurement = self.measure_obj_struct_tensor_eigenval_single_sigma(label_image=label_image,
                                                                                           axis=axis,
                                                                                           struct_tens_kwargs=struct_tens_kwargs,
                                                                                           erosion_kwargs=erosion_kwargs,
                                                                                           regionprops_kwargs=regionprops_kwargs,
                                                                                           erosion_warning=erosion_warning)

            glob_eigen_measurement_list.append(sigma_eigen_measurement)
        
        # if only a single sigma was measured, return the measurements for the sigma
        if len(glob_eigen_measurement_list)==1:
            
            return glob_eigen_measurement_list[0]
        
        # add sigmas to column names if more than a measurement was performed
        elif len(glob_eigen_measurement_list)>1:

            # rename columns
            def rename_sigma_columns(measurement_df:pd.DataFrame,
                                     ch_sep:str|None=None,
                                     sigma_sep:str|None=None,
                                     sigma_str:str|None=None,
                                     keep_column:tuple|None=('label'),
                                     measurement_pos:int=0,
                                     ch_pos:int=-1)->pd.DataFrame:
                
                # set default renaming variables
                if ch_sep is None:
                    ch_sep="-"
                
                if sigma_sep is None:
                    sigma_sep="_"
                
                if sigma_str is None:
                    sigma_str="s0"

                # intialize a mapper
                column_mapper = {}

                # iterate through the columns
                for col in measurement_df.columns:

                    # avoid renaming of certain columns
                    if keep_column!=None and col not in keep_column:

                        # split column name - NOTE: this should lead to no split in case no axis is passed
                        col_split = col.split(ch_sep)

                        # reform column with channel if a channel was present, otherwise without channel
                        if len(col_split)>1:
                            new_col = f"{col_split[measurement_pos]}{sigma_sep}{sigma_str}{ch_sep}{col_split[ch_pos]}"
                        else:
                            new_col = f"{col_split[measurement_pos]}{sigma_sep}{sigma_str}"

                        column_mapper[col] = new_col
                    
                    else:
                        column_mapper[col] = col
                
                # rename columns
                measurement_df = measurement_df.rename(columns=column_mapper)
                return measurement_df
            
            # select the first measurement and rename it's columns - keep 'label' column name intact
            sigma_eigen_measurement_0 = rename_sigma_columns(glob_eigen_measurement_list[0],
                                                                 sigma_str="0",
                                                                 **renaming_kwargs)

            # copy the first measurement to the global list
            glob_eigen_measurement = sigma_eigen_measurement_0.copy()

            # rename columns for all subsequent measurements - merge dataframes
            for i, measurement in enumerate(glob_eigen_measurement_list[1:], start=1):

                sigma_eigen_measurement_i = rename_sigma_columns(measurement,
                                                                 sigma_str=f"{i}",
                                                                 **renaming_kwargs)
            
                sigma_eigen_measurement = glob_eigen_measurement.merge(sigma_eigen_measurement_i,
                                                                       on='label',
                                                                       how='left',
                                                                       copy=True)
                
                glob_eigen_measurement = sigma_eigen_measurement.copy()

            return glob_eigen_measurement

        else:
            raise ValueError("no measurement was performed. Check min_sigma, max_sigma and num_sigma values")

    def measure_object_anisotropy(self,
                                  label_image:np.array,
                                  min_sigma:int|float|tuple|list|None=None,
                                  max_sigma:int|float|tuple|list|None=None,
                                  num_sigma:int|float|None=None,
                                  axis:int|None=None,
                                  erosion_kwargs:dict|None=None,
                                  regionprops_kwargs:dict|None=None,
                                  eigen_ratio_kwargs:dict|None=None,
                                  erosion_warning:bool=True,
                                  renaming_kwargs:dict|None=None) -> pd.DataFrame:
        
        """
        sigma can't be passed to eigen_ratio_kwargs['kwargs'].

        num_sigma must be an int or float or None.

        min_sigma and max_sigma can be:
        - both None. In which case, a kernel of size 1 per each dimension of the image is used. num_sigma is ignored.

        - one None and the second a single value (int or float) or a sequence of values (list or tuple) with length equal
        to the number of dimensions of the image. In which case the non-None value is used as the sigma value and
        num_sigma is ignored. If single value, the value is used on all the dimensions of the image.

        - both single values (int or float). In this case...
            - if num_sigma is None, the function is calculated for the two sigmas indicated. Precisely first
            for a kernel of size min_sigma in all the dimensions of the image, secondly for a kernel of size max_sigma in
            all the dimensions of the image.

            - if num_sigma is not None, the function is calculated for the linspace between min_sigma and max_sigma, divided by
            num_sigma. Meaning that a kernel of size S per each image dimension is used for each S-value in the linspace.

        - both sequences of values (list, tuple), both with the same length as the number of dimensions of the image. In this case...
            - the i-th position in min_sigma and max_sigma indicates the i-th dimension in the kernel used.    
            
            - if num_sigma is None, the function is calculated for the two sigmas indicated. First for a kernel of shape min_sigma.
            Secondly for a kernel of shape max_sigma.

            - if num_sigma is not None, N kernels are calculated with N corresponding to num_sigma. Per each i value in
            min_sigma and corresponding i value in max_sigma, a N long linspace is created in the i-min_sigma - i-max_sigma range.
            Each of N-th kernel is generated by taking the N-th value per each of i-th linspace.
        """

        # set default kwargs
        if erosion_kwargs is None:
            erosion_kwargs={'footprint':disk(9)}
        
        if regionprops_kwargs is None:
            regionprops_kwargs={'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'],
                                'separator':'-'}
        
        if eigen_ratio_kwargs is None:
            eigen_ratio_kwargs={'eps':1.e-6, 'sep':'_'}
        
        if renaming_kwargs is None:
            renaming_kwargs:dict={'keep_column':('label'),'measurement_pos':0,'ch_pos':-1}
        

        eigen_ratio_kwargs = eigen_ratio_kwargs.copy() # to avoid modifying the input dictionary
        
        # ensure that 'sigma' is not passed to eigen_ratio_kwargs['kwargs']
        if 'kwargs' in eigen_ratio_kwargs:
            assert 'sigma' not in eigen_ratio_kwargs['kwargs'], "sigma can't be passed to eigen_ratio_kwargs['kwargs']. Use min_sigma, max_sigma and num_sigma instead"
        else:
            eigen_ratio_kwargs['kwargs'] = {}

        # ensure that 'label is in rename_kwargs['keep_column']
        assert 'label' in renaming_kwargs['keep_column'], "'label' must be in renaming_kwargs['keep_column']"

        if 'separator' in regionprops_kwargs:
           if regionprops_kwargs['separator']=='_':
               print("WARNING: using '_' as separator for regionprops can lead to ambiguous column names and potential error when renaming columns")
        else:
            regionprops_kwargs = regionprops_kwargs.copy() # to avoid modifying the input dictionary
            regionprops_kwargs['separator']='-'
        
        # check that 'label' is in regionprops_kwargs['properties']
        if 'properties' in regionprops_kwargs:
            assert 'label' in regionprops_kwargs['properties'], "'label' must be in regionprops_kwargs['properties']"
        
        # ensure that no division by 0 is present
        if 'eps' in eigen_ratio_kwargs:
            assert eigen_ratio_kwargs['eps']!=0, "eps must be different than 0"
        else:
            eigen_ratio_kwargs['eps'] = 1.e-6

        # unstack image if axis is provided
        if axis!=None:
            unstacked_image = [np.moveaxis(a,axis,-1)[...,0] for a in np.split(self.image, indices_or_sections=self.image.shape[axis], axis=axis)]
            
            assert len(unstacked_image[0].shape)==len(label_image.shape), "if axis is provided, the sub-image and label_image must have the same number of dimensions"

            # generate a collection of sigmas
            sigma_collection =  self.generate_sigma_collection(image=unstacked_image[0],
                                                               min_sigma=min_sigma,
                                                               max_sigma=max_sigma,
                                                               num_sigma=num_sigma)
        
        else:
            assert len(self.image.shape)==len(label_image.shape), "if no axis is provided, image and label_image must have the same number of dimensions"

            # generate a collection of sigmas
            sigma_collection =  self.generate_sigma_collection(image=self.image,
                                                               min_sigma=min_sigma,
                                                               max_sigma=max_sigma,
                                                               num_sigma=num_sigma)

        # initialize a collection list
        glob_eigen_ratio_list = []

        # iterate through the sigma collection
        for sigma in sigma_collection:

            # add sigma to eigen_ratio_kwargs['kwargs']
            eigen_ratio_kwargs['kwargs']['sigma'] = sigma

            # measure the anisotropy for the current sigma
            sigma_eigen_ratio = self.measure_object_anisotropy_single_sigma(label_image=label_image,
                                                                            axis=axis,
                                                                            erosion_kwargs=erosion_kwargs,
                                                                            regionprops_kwargs=regionprops_kwargs,
                                                                            eigen_ratio_kwargs=eigen_ratio_kwargs,
                                                                            erosion_warning=erosion_warning)

            # collect results
            glob_eigen_ratio_list.append(sigma_eigen_ratio)
        
        
        # if only a single sigma was measured, return the measurements for the sigma
        if len(glob_eigen_ratio_list)==1:
            
            return glob_eigen_ratio_list[0]
        
        # add sigmas to column names if more than a measurement was performed
        elif len(glob_eigen_ratio_list)>1:

            # rename columns
            def rename_sigma_columns(measurement_df:pd.DataFrame,
                                     ch_sep:str|None=None,
                                     sigma_sep:str|None=None,
                                     sigma_str:str|None=None,
                                     keep_column:tuple|None=('label'),
                                     measurement_pos:int=0,
                                     ch_pos:int=-1)->pd.DataFrame:
                
                # set default renaming variables
                if ch_sep is None:
                    ch_sep="-"
                
                if sigma_sep is None:
                    sigma_sep="_"
                
                if sigma_str is None:
                    sigma_str="s0"

                # intialize a mapper
                column_mapper = {}

                # iterate through the columns
                for col in measurement_df.columns:

                    # avoid renaming of certain columns
                    if keep_column!=None and col not in keep_column:

                        # split column name - NOTE: this should lead to no split in case no axis is passed
                        col_split = col.split(ch_sep)

                        # reform column with channel if a channel was present, otherwise without channel
                        if len(col_split)>1:
                            new_col = f"{col_split[measurement_pos]}{sigma_sep}{sigma_str}{ch_sep}{col_split[ch_pos]}"
                        else:
                            new_col = f"{col_split[measurement_pos]}{sigma_sep}{sigma_str}"

                        column_mapper[col] = new_col
                    else:
                        column_mapper[col] = col
                
                # rename columns
                measurement_df = measurement_df.rename(columns=column_mapper)
                return measurement_df
            
            # select the first measurement and rename it's columns - keep 'label' column name intact
            sigma_eigen_ratio_measurement_0 = rename_sigma_columns(glob_eigen_ratio_list[0],
                                                                   sigma_str="0",
                                                                   **renaming_kwargs)

            # copy the first measurement to the global list
            glob_eigen_ratio_measurement = sigma_eigen_ratio_measurement_0.copy()

            # rename columns for all subsequent measurements - merge dataframes
            for i, ratio_measurement in enumerate(glob_eigen_ratio_list[1:], start=1):

                sigma_eigen_ratio_measurement_i = rename_sigma_columns(ratio_measurement,
                                                                       sigma_str=f"{i}",
                                                                       **renaming_kwargs)
            
                sigma_eigen_ratio_measurement = glob_eigen_ratio_measurement.merge(sigma_eigen_ratio_measurement_i,
                                                                                   on='label',
                                                                                   how='left',
                                                                                   copy=True)
                
                glob_eigen_ratio_measurement = sigma_eigen_ratio_measurement.copy()

            return glob_eigen_ratio_measurement

        else:
            raise ValueError("no measurement was performed. Check min_sigma, max_sigma and num_sigma values")