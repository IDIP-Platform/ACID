import math
import numpy as np
import pandas as pd
from skimage.feature import multiscale_basic_features
from skimage.morphology import erosion, disk
from skimage.measure import regionprops_table

class MeasureHessianMatrix():
    """
    Given an image (passed to __init__) and corresponding labelled objects (label_image) measures the statistics
    (by default mean, max, min and std) of the intensity distribution of the hessian matrix eigenvalues for each
    labelled object in a label image.

    The hessian matrix eigenvalues are measurements of the intensity changes in the image in a neighbourhood of a pixel.
    Precisely, they are the second derivative of the intensity function on the image, thus they capture how steeply
    the intensity change changes. They are positive when the instensity change increases, negative when the intensity
    change decreases and they are 0 if the intensity change does not change.

    Thus, relative to the actual intensity of the image, they capture the CURVATURE of the intensity
    (aka how BENT the intensity surface is):
    - large absolute values indicate a strong curvature.
    - If eigenvalues have comparable absolute values, then there is not a prevalent directionality of the curvature. In
    a 2D image, this would correspond to a bright/dark blob-like structure or a minima/maxima. Conversely, if
    an eigenvalue has a stronger absolute value than the others, the curvature is stronger in a direction.

    Hessian eigenvalues have necessarily to be interpreted together:
    - If both are positive they indicate that there is a "valley" in the image (e.g. a dark blob)
    - If both are negative they indicate that there is a local maximum in the image (e.g. a bright blob)
    - If they have opposite side they indicate a saddle point

    It is important to NOTE that hessian eigenvalues are sensitive to noise. It is recommended to use a smoothed image as input.

    NOTE: a possible way of combining the hessian eigenvalues in a single index is by using
    the shape_index (https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_shape_index.html ;
    https://www.sciencedirect.com/science/article/pii/026288569290076F?via%3Dihub)

    ========= ========= =========
    The Hessian matrix of an image summarizes the pure second derivatives of the image along
    all its axis.

    The eigenvalues/vectors are the same information, but rotated into the best-fitting local
    coordinate frame where one axis aligns with the strongest curvature direction.

    The eigenvalues indicate the amount of curvature in the reoriented axis (aka, the
    principal curvature directions).

    The eigenvectors indicate the directions of the reoriented axis (aka, the principal
    curvature directions).

    ========= ========= =========

    The function, precisely:

    1) the hessian matrix eigenvalues are calculated per each pixel of the image by setting the 'texture' argument
    as True in skimage.feature.multiscale_basic_features
    (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.multiscale_basic_features).
    The output of this step is an image stack (eigen_stack), where each sub-stack on the axis h_eigenval_position
    (forced to be to -1) corresponds to one of the eigenvalues of the hessian matrix of the input image.

    2) The labelled objects in label_image are eroded using the the parameters specified in erosion_kwargs.
    By default, a disk-shaped structuring element of radius 9 is used.

    3) The measurements are computed from the eigen_stack by applying the regionprops_table function
    (https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops_table)
    from skimage.measure and using h_eigenval_position as a channel_axis and the eroded label image as a mask.
    This extracts the statistics per each eroded labelled object and each hessian eigenvalue. NOTE: as the measurements
    are computed using regionprops_tables, any measurement compatible with regionprops_table can be extracted,
    including extra_properties. Any argument accepted by regionprops_table can be passed to the argument
    regionprops_kwargs of the method.

    4) The measurements are returned as a pandas dataframe. The dataframe structure and layout matches regionprops
    output: rows correspond to labelled objects, columns correspond to measurements. If image has multiple channels,
    it is possible to extract and return the measurements per each channel separately. In the output dataframe the
    channel index is indicated as the last number of the column name. The method uses '-' as default separator for the
    channel index in the column name. The separator can be changed by passing a different value to regionprops_kwargs.
    NOTE: if '_' is used as separator the output might be wrong. The method does not return an error but raises a warning.

    NOTE:
    - the measurements which are returned per each object in label_image are obtained by aggregating the object pixels.
    In addition, the default behaviour is to erode the object before computing the measurements and using a
    disk-shaped structuring element of radius 9 as footprint. This behaviour is done to counteract the influence
    of the object boundaries on the measurements for small sigma, as a relatively strong intensity gradient change is
    expected at the object boundaries (between object and background). However, this behaviour can cause the loss of
    small objects as well as errors if the footprint dimensions don't match the number of dimensions of label_image.
    This behaviour can be controlled using the parameter erosion_kwargs, which takes all the keyword arguments
    for the erosion function (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.erosion).

    - The size of the neighbourhood for which the intensity gradient is calculated can be indicated using a parameter
    called sigma. The method accepts a range of sigmas, if provided. Such a range can be specified using the parameters
    sigma_min, sigma_max and num_sigma to specify, respectively, the smallest, largest and number of equally spaced
    steps for the sigma range. This behaviour is identical to the behaviour of sigma in
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.multiscale_basic_features

    - If the image to analyze has multiple channels, it is possible to specify the position of the channel axis
    (parameter axis). In this case label_image must have the same shape of each individual channel.
    In this case the results are returned per each sigma (if provided) and per each channel. The methods follow
    regionprops convention, thus the channel index is indicated as the last number of the column name. Also, the method
    uses '-' as default separator for the channel index in the column name. The separator can be changed by passing a
    different value to regionprops_kwargs. NOTE: if '_' is used as separator the output might be wrong. The method
    does not return an error but raises a warning. When multiple sigmas are provided and multiple channels are present,
    the output column names will include sigma index in the specified range as the second last number and the
    channel index as the last number. The channel index is further recognized as separated by the specified separator
    (regionprops_kwargs['separator']). The sigma index is returned instead of the actual sigma value as sigma can be
    a number with decimals.

    - The input image can theoretically have n-dimensions. It has been tested on 2D images, 2D images with multiple
    channels, 3D images and 3D images with multiple channels. Due to the default setting of erosion
    (see erosion_kwargs below), the default setting will return an error when working with images with more than 2
    spatial dimensions (this error does not involve 2D images with multiple channels), one should pass a structuring
    element for erosion which matches the number of dimensions of the input image (or image channels), or don't pass
    any structural element, in which case the default behaviour follows skimage.morphology.erosion
    (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.erosion).

    ========= ========= =========

    __init__ Inputs:
    - image: The input image to analyze, as a NumPy array.

    ========= ========= =========

    measure_obj_hessian_matrix_eigenval Inputs:

    - label_image. The label image to analyze, as a NumPy array. If axis is None, label_image must have the same shape
    as image. If axis is not None, label_image must have the same shape as image channels.

    - sigma_min. float. Optional, default 1. The minimum sigma value for the hessian matrix. The parameter is passed to,
    and therefore behaves identically to what reported in skimage.feature.measure_basic_features sigma_min
    (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.multiscale_basic_features).

    - sigma_max. float. Optional, default 1. The maximum sigma value for the hessian matrix. The parameter is passed to,
    and therefore behaves identically to what reported in skimage.feature.measure_basic_features sigma_max
    (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.multiscale_basic_features).

    - num_sigma. int or None. Optional, default None. The number of sigma values to compute in the range sigma_min
    (included) and sigma_max (included). The parameter is passed to, and therefore behaves identically to what reported
    in skimage.feature.measure_basic_features num_sigma (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.multiscale_basic_features).
    If None, the number of sigma is calculated as:
    num_sigma= 1+ math.floor(math.log2(sigma_max/sigma_min)), unless sigma_max is smaller or equal to 2*sigma_min, in
    which case 1 sigma is used.

    - axis. int or None. Optional, default None. If int, the measurements will be returned per each input image
    sub-stack along the specified axis.

    - mbf_kwargs. dict. Optional, default {} (empty dictionary). Keyword arguments for the skimage.feature.multiscale_basic_features function.
    (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.multiscale_basic_features). NOTE: intensity, edges and texture
    can't be passed to mbf_kwargs and they are forced to be, respectively, False, False and True, for proper behaviour
    of the function. sigma_min, sigma_max and num_sigma can't be passed to mbf_kwargs as they must be specified using
    the corresponding arguments in the method. channel_axis can't be passed to mbf_kwargs as it must be specified
    using axis argument in the method. image can't be passed to mbf_kwargs as the input image to the method is used.
    Thus, this leaves only num_workers, and extra parameters to be passed to mbf_kwargs.

    - erosion_kwargs. dict. Optional, default {'footprint':disk(9)}. Keyword arguments for the erosion function. All
    parameters of skimage.morphology.erosion (https://scikit-image.org/docs/0.25.x/api/skimage.morphology.html#skimage.morphology.erosion)
    can be passed to this dictionary as key-value pairs. NOTE: the default setting can only be used for eroding a 2D
    image. For eroding images with higher dimensions one should pass a structuring element which matches the number of
    dimensions of the input image (or image channels), or no structural element, in which case the default behaviour
    follows skimage.morphology.erosion.

    - regionprops_kwargs. dict. Optional, default {'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'], 'separator':'-'}.
    Keyword arguments for the regionprops function. All parameters of skimage.measure.regionprops (https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops)
    can be passed to this dictionary as key-value pairs. These include properties to measure and extra_properties
    to measure. NOTE: if 'separator' is passed to regionprops_kwargs and it is '_', the behavior of the function is
    not guaranteed (a warning is printed). In addition, such separator must be different than the string passed to sep argument.

    - erosion_warning. bool. Optional, default True. If True, a warning will be issued if the erosion operation
    removes any labelled object in label_image.

    - h_eigenval_position. int and forced to be -1. The index of the axis in the output of skimage.feature.multiscale_basic_features
    where eigenvalues are expected. NOTE: this parameter is useless, it is kept to avoid hard coded parameters directly
    in the function and keep them in the arguments.

    - sep. str. Optional, default '_'. The string to use to separate information bits in the column names of the output dataframe.
    It must be different than regionprops_kwargs['separator']. As a consequence, if the default separator is used for
    regionprops_kwargs, it can't be '-'.

    ========= ========= =========

    measure_obj_hessian_matrix_eigenval Output:

    pd.DataFrame. The dataframe structure and layout matches regionprops output: rows correspond to labelled objects
    in label_image, columns correspond to measurements. Per each image channel, per each sigma, the properties
    indicated in regionprops_kwargs (using the 'properties' and 'extra_properties' parameters) are computed per
    each eigenvalue.

    If image has multiple channels, the channel index is indicated as the last number of the column name. It is
    separated from the rest of the column name by the string indicated in regionprops_kwarg['separator'].

    If multiple sigmas are measured, the sigma index in the specified range is the second last number in the column
    name and the channel index is the last number. The sigma index is returned instead of the actual sigma value as sigma
    can be values with decimal points.
    """

    def __init__(self, image:np.array):
        self.image = image

    # def default_sigma_num(x:float,y:float,factor:float) -> int:
    #     """
    #     Calculates the number of hessian matrix - transformed images which are returned by skimage.feature.multiscale_basic_features
    #     (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.multiscale_basic_features) when None is passed
    #     to num_sigma.

    #     Inputs:
    #     - x. float. The value passed to sigma_min in skimage.feature.multiscale_basic_features
    #     - y. float. The value passed to sigma_max in skimage.feature.multiscale_basic_features
    #     - factor. float. The number of dimensions of the image passed to skimage.feature.multiscale_basic_features

    #     Ouptut: int. The number of hessian-matrix-eigenvalues calculated by skimage.feature.multiscale_basic_features for the input
    #     image, when num_sigma=None. In other words, the number of 'texture' features which are calculated by
    #     skimage.feature.multiscale_basic_features.
    #     """
    #     val= 1+ math.floor(math.log2(y/x))
    #     if y<=2*x:
    #         return 1*factor
    #     return val*factor

    def measure_obj_hessian_matrix_eigenval(self,
                                            label_image:np.array,
                                            sigma_min:float=1,
                                            sigma_max:float=1,
                                            num_sigma:int|None=None,
                                            axis:int|None=None,
                                            mbf_kwargs:dict|None=None,
                                            erosion_kwargs:dict|None=None,
                                            regionprops_kwargs:dict|None=None,
                                            erosion_warning:bool=True,
                                            h_eigenval_position:int=-1,
                                            sep:str|None=None)->pd.DataFrame:
        # use defaults
        if mbf_kwargs is None:
            mbf_kwargs={}

        if erosion_kwargs is None:
            erosion_kwargs={'footprint':disk(9)}

        if regionprops_kwargs is None:
            regionprops_kwargs={'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'],
                                'separator':'-'}

        if sep is None:
            sep='_'

        assert 'image' not in mbf_kwargs, "image can't be passed to mbf_kwargs, as the image input to __init__ is used for processing"
        assert 'sigma_min' not in mbf_kwargs, "sigma_min can't be passed to mbf_kwargs, use sigma_min argument instead"
        assert 'sigma_max' not in mbf_kwargs, "sigma_max can't be passed to mbf_kwargs, use sigma_max argument instead"
        assert 'num_sigma' not in mbf_kwargs, "num_sigma can't be passed to mbf_kwargs, use num_sigma argument instead"
        assert 'channel_axis' not in mbf_kwargs, "channel_axis can't be passed to mbf_kwargs, use axis argument instead"
        assert 'intensity' not in mbf_kwargs, "intensity can't be passed to mbf_kwargs as it is set to be False. Hessian eigenvalues can't be properly extracted if True"
        assert 'edges' not in mbf_kwargs, "edges can't be passed to mbf_kwargs as it is set to be False. Hessian eigenvalues can't be properly extracted if True"
        assert 'texture' not in mbf_kwargs, "texture can't be passed to mbf_kwargs as it is set to be True. Hessian eigenvalues can't be extracted otherwise"
        assert h_eigenval_position==-1, "for the best it could be checked, the eigenvalues of hessian matrix are added in the last dimension by skimage.feature.multiscale_basic_features"
        if 'separator' not in regionprops_kwargs:
            assert sep!='-', "using '-' as sep can only be done together with passing a 'separator' different than '-' to regionprops_kwargs"
            regionprops_kwargs = regionprops_kwargs.copy() # to avoid modifying the input dictionary
            regionprops_kwargs['separator']='-'
        else:
            assert regionprops_kwargs['separator']!=sep, "sep and regionprops's separator must be different"
            if regionprops_kwargs['separator']=='_':
                print("WARNING: using '_' as regionprops separator can lead to wrong column names")

        # if axis is provided
        if axis!=None:
            # move channel axis in the last position, if it is provided
            img = np.moveaxis(self.image, axis, -1)

            # set ax to -1 (will be passed to multiscale_basic_feature's channel axis)
            ax = -1

            # get number of channels
            ch_number = img.shape[-1]

            # get the number of eigenvalues
            eigenv_number = len(img.shape)-1

        else:
            # set img to be a copy of the input image (will be passed to multiscale_basic_feature's image)
            img = self.image.copy()

            # set ax to None (will be passed to multiscale_basic_feature's channel axis)
            ax=None

            # set the number of channels as 1
            ch_number = 1

            # get the number of eigenvalues
            eigenv_number = len(img.shape)


        # calculate hessian eigenvalues
        h_eigenvals_i = multiscale_basic_features(img,
                                                  sigma_min=sigma_min,
                                                  sigma_max=sigma_max,
                                                  num_sigma=num_sigma,
                                                  channel_axis=ax,
                                                  intensity=False,
                                                  edges=False,
                                                  texture=True,
                                                  **mbf_kwargs)

        # calculate the number of sigmas which have been computed
        if num_sigma!=None:
            computed_sigma = num_sigma

            # assertion statement: the number of sigma, times number of channels times number of eigenvalue should match
            # the hessian eigenvalues calculated by skimage.feature.multiscale_basic_features
            tot_number = computed_sigma*ch_number*eigenv_number
            assert tot_number==h_eigenvals_i.shape[h_eigenval_position]

        else:
            computed_sigma=int((h_eigenvals_i.shape[h_eigenval_position]/eigenv_number)/ch_number)

            # assertion statement: the number of sigma, times number of channels times number of eigenvalue should match
            # the hessian eigenvalues calculated by skimage.feature.multiscale_basic_features
            assert h_eigenvals_i.shape[h_eigenval_position]%eigenv_number==0
            assert (h_eigenvals_i.shape[h_eigenval_position]/eigenv_number)%ch_number==0

        # form a dictionary mapping the calculated hessian eigenvalues to the corresponding eigenvalue order,
        # (potential) sigma and (potential) channel
        mapper={}
        i=0
        for ch in range(ch_number):
            for si_gma in range(computed_sigma):
                for eigv in range(eigenv_number):
                    # mapper[i]=f"hessian{eigv}eigen{eigv}{sep}{si_gma}{regionprops_kwargs['separator']}{ch}"
                    mapper[str(i)]=(eigv+1, si_gma, ch)
                    i=i+1

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

        # ensure eigenvalues are in the last position of h_eigenvals
        h_eigenvals = np.moveaxis(h_eigenvals_i, h_eigenval_position,-1)

        # measure intensities of hessian eigenvalues
        h_eigen_measurement_i = pd.DataFrame(regionprops_table(eroded_label_image,
                                                               intensity_image=h_eigenvals,
                                                               **regionprops_kwargs))

        # rename columns
        # initialize a dictionary to be used as a mapper for column renaming
        column_mapper = {}

        # iterate through the columns
        for clm in h_eigen_measurement_i.columns:

            # split column name to separate the regionprops measurement and the h_eigenvals index
            clm_split = clm.split(sep=regionprops_kwargs['separator'])

            # don't modify the column name if no h_eigenvals index position is present
            # these are non-intensity measurements as 'area', 'perimenter' etc... in addition, they are the 'label' column and the centroid
            if len(clm_split)==1:
                new_clm=clm

            else:
                # get eigenvalue order and sigma info
                clm_eigenv, clm_sigma, clm_ch = mapper[clm_split[-1]]

                # it the input image has no channels
                if ch_number==1:

                    # if only 1 sigma was calculated
                    if computed_sigma==1:
                        # only add eigenval order info to the column
                        new_clm = f"hessian{sep}eigv{sep}{clm_eigenv}{sep}{clm_split[0]}"

                    # if more than 1 sigma was calculated
                    else:
                        # add eigenval order and sigma info to the column
                        new_clm = f"hessian{sep}eigv{sep}{clm_eigenv}{sep}{clm_split[0]}{sep}{clm_sigma}"

                # if the input channel has more than a channel
                else:

                    # if only 1 sigma was calculated
                    if computed_sigma==1:
                        # add eigenval order and channel to the column
                        new_clm = f"hessian{sep}eigv{sep}{clm_eigenv}{sep}{clm_split[0]}{regionprops_kwargs['separator']}{clm_ch}"

                    # if more than 1 sigma was calculated
                    else:
                        # add eigenval order, sigma and channel to the column
                        new_clm = f"hessian{sep}eigv{sep}{clm_eigenv}{sep}{clm_split[0]}{sep}{clm_sigma}{regionprops_kwargs['separator']}{clm_ch}"

            # link old and new column names in the column mapper dictionary
            column_mapper[clm]=new_clm

        # rename columns of h_eigen_measurement_i
        h_eigen_measurement = h_eigen_measurement_i.rename(column_mapper,axis=1, copy=True)

        return h_eigen_measurement
