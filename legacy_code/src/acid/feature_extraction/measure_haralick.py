from collections.abc import Sequence
import numpy as np
import dask.bag as db
import pandas as pd
from .graycoprops_compiled import graycoprops
from skimage.feature import graycomatrix
from skimage.morphology import disk, erosion
from skimage.measure import regionprops, regionprops_table
from skimage.util.shape import view_as_windows
# from skimage.exposure import rescale_intensity
from acid.utils.miscellaneous_utils import _normalize_to_list
from acid.image_processing.rescale_intensity import quantize_image


"""
SUMMARY OF FUNCTIONS:

- measure_haralick_image: measures haralick features on an entire image.

- glcm_feature_map_ch: computes a Haralick feature map for one object (mask)
and a one image channel.
If a mask is passed, the feature map is only computed inside the mask.
If no mask is passed, the feature map is computed on the entire image.

- glcm_feature_map: computes a Haralick feature map for one object (mask, ref to glcm_feature_map_ch)
and a multi-channel image, by iterating glcm_feature_map_ch per each channel and concatenating the results.

- parallel_glcm_feature_map: computes a Haralick feature map for all the label objects in a label_image.
If the input image has multiple channels, the feature maps can be computed per each channel independently.
Takes advantage of Dask to parallelize the computation of feature maps per each object and (eventually)
per each channel.

- measure_haralick_features: uses parallel_glcm_feature_map to compute the haralick feature maps of all the label objects
in a label_image, then measures the regionprops for all the label objects in a label_image based on the computed
haralick feature maps.


=== === ===
HARALICK FEATURES:
Haralick features are texture features based on the gray-level co-occurrence matrix (GLCM) of an image.
The basic idea is to quantify how often different combinations of pixel brightness values (gray levels) occur in an image,
by comparing pairs of pixels separated by a certain distance and angle.

- Contrast: Local intensity variation — how much neighboring pixels differ.
High when: The image has strong edges or sharp transitions (high texture contrast).
Low when: The image is smooth or uniform, with little gray-level variation.

- Dissimilarity: Average gray-level difference between neighboring pixels (like contrast, but linear rather than squared).
High when: Adjacent pixels have large intensity differences.
Low when: Neighboring pixels have similar intensities.

- Homogeneity (also called Inverse Difference Moment): Closeness of GLCM elements to the diagonal — i.e., similarity of neighboring pixels.
High when: Texture is smooth, with little contrast and gradual transitions.
Low when: Texture is rough or has strong contrasts (values far from diagonal).

- Energy: Square root of ASM — same interpretation, just rescaled.
High when: Texture is consistent or periodic.
Low when: Texture is noisy or heterogeneous.

- Correlation: Linear relationship between neighboring gray levels — the degree to which a pixel’s intensity can be predicted from its neighbor.
High when: There’s structured, repetitive texture or gradients (predictable intensity relationships).
Low when: Texture is random, or gray levels vary without relation.

- ASM (Angular Second Moment): Uniformity / orderliness of texture; based on squared GLCM values.
High when: Texture is highly uniform (repeating pattern, single tone).
Low when: Texture is random or varied (many co-occurring gray levels).

- Mean: Average value of the GLCM.

- Variance: Spread of the gray-level distribution around the mean; indicates overall intensity dispersion.

- Std: Standard deviation of the GLCM.

- Entropy: Texture randomness or complexity.
High when: The image has irregular, heterogeneous patterns and many gray-level combinations (noisy, complex texture).
Low when: The texture is regular or uniform, with few dominant gray-level relationships.


FOR DEAILS ON HARALICK FEATURES AND THEIR COMPUTATION, SEE:
https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycomatrix
https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycoprops
https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_glcm.html).


=== === ===
IMPORTANT NOTE:
As for CellProfiler (https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureTexture.html)
by default the gray intensity levels are min-max normalized per object bounding box to
8 levels (0-7) before calculating the GLCM and the haralick features.
This behavior falls back to the function glcm_feature_map_ch. The behaviour can be
changed by passing the 'levels' argument in graycomtx_kwargs.
Ref to the documentation of the functions glcm_feature_map_ch and
parallel_glcm_feature_map for more details.
"""



def name_column(measurement:str,
                 prefix:str|None=None,
                 suffix:str|None=None,
                 prefix_sep:str|None=None,
                 suffix_sep:str|None=None)->str:

    if prefix_sep is None:
        prefix_sep='_'

    if suffix_sep is None:
        suffix_sep:str='_'

    if prefix!=None and suffix==None:
        if prefix_sep!=None:
            new_col = f"{prefix}{prefix_sep}{measurement}"
        else:
            new_col = f"{prefix}_{measurement}"
    if prefix==None and suffix!=None:
        if suffix_sep!=None:
            new_col = f"{measurement}{suffix_sep}{suffix}"
        else:
            new_col = f"{measurement}_{suffix}"
    elif prefix!=None and suffix!=None:
        if prefix_sep!=None and suffix_sep==None:
            new_col = f"{prefix}{prefix_sep}{measurement}_{suffix}"
        elif prefix_sep==None and suffix_sep!=None:
            new_col = f"{prefix}_{measurement}{suffix_sep}{suffix}"
        elif prefix_sep!=None and suffix_sep!=None:
            new_col = f"{prefix}{prefix_sep}{measurement}{suffix_sep}{suffix}"
        else:
            new_col = f"{prefix}_{measurement}_{suffix}"
    else:
        new_col=measurement

    return new_col

def measure_haralick_image(image:np.array,
                           prop:str|Sequence|None=None,
                           distances:int|float|tuple|list|Sequence|None=None,
                           angles:int|float|tuple|list|Sequence|None=None,
                           graycomtx_kwargs:dict|None=None,
                           sep:str|None=None) -> pd.Series:
    """
    Measure haralick features on an entire image.

    Measurements are based on skimage.feature.graycomatrix and skimage.feature.graycoprops (see their documentation for details
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycomatrix
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycoprops
    https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_glcm.html).

    Image must be 2D (as a corollary, the image is single channel). If 'levels' is
    passed to graycomtx_kwargs, only integer typed input images are supported and only
    positive valued images are supported.

    Returns a pandas Series with haralick measurements.

    NOTE: by default, the image is rescaled to 8 intensity levels (0-7) before calculating the GLCM. To change this behaviour,
    set pass the 'levels' argument in graycomtx_kwargs. For example, if an image
    has pixel values in the range 0-255, setting 'levels' to 256 in graycomtx_kwargs
    will avoid rescaling and use the original pixel values for GLCM calculation.
    Note that 0 is always assumed to be the lowest intensity value and that no
    rescaling is performed if 'levels' is specified. 'levels' is simply the number of
    intensity levels used for GLCM calculation, starting from 0.
    The "levels" argument is passed to graycomatrix, ref to the documentation
    for additional details (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycomatrix).

    Inputs:
    - image: 2D numpy array. The input image for which the haralick features have to be computed.
    If 'levels' is passed to graycomtx_kwargs, only integer typed input images are supported and only
    positive valued images are supported.

    - prop: str, list of str or None. Single haralick feature (if str) or list of haralick features (if list) to measure. Optional.
    Default: ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'mean', 'variance', 'std', 'entropy'])

    - distances: int, float, tuple, list, sequence or None. The distance (if int or float) or list of distances to use for haralick
    features measurement. Optional. Default: [1].

    - angles: int, float, tuple, list, sequence or None. The angle (if int or float) or list of angles to use for haralick
    features measurement. Optional. Default: [0].

    - graycomtx_kwargs: dict or None. Additional arguments to pass to graycomatrix. Optional. Default: {'levels':8,'symmetric':True,'normed':True})
    NOTE: 'distances' and 'angles' can't be passed here, use the dedicated arguments instead.

    - sep: str or None. The separator to use for column names. Optional. Default: '_'.

    Outputs:
    - haralick_measurements: pandas Series. Series with haralick measurements as values and measurement names as index. Measurements
    per each distance and angle are included. The name of the measurements follow the pattern: {property}{sep}{distance_index}{sep}{angle_index}.
    As a consequence, neither distance nor angle values are included in the measurement names, only their index in the list of distances/angles.
    This is done to avoid problems with special characters or float numbers in column names. Note that when only one distance or angle is
    used, their index is still indicated, as 0, in the measurement name.

    """

    # set default properties
    if prop is None:
        prop=['contrast', 'dissimilarity', 'homogeneity', 'energy',
              'correlation', 'ASM', 'mean', 'variance', 'std', 'entropy']
    else:
        # if prop is str, store it in a list, for compatibility with following iteration
        if isinstance(prop,str):
            prop=[prop]

    # set default distances
    if distances is None:
        distances=[1]
    else:
        # if distances are int or float
        # store distances in a list, for compatibility with skimage.feature.graycomatrix
        if isinstance(distances, int) or isinstance(distances,float):
            distances=[distances]

    # set default angles
    if angles is None:
        angles=[0]

    else:
        # if angles are int or float
        # store angles in a list, for compatibility with skimage.feature.graycomatrix
        if isinstance(angles,int) or isinstance(angles,float):
            angles=[angles]

    # set default graycomtx_kwargs
    if graycomtx_kwargs is None:
        graycomtx_kwargs={'symmetric':True,'normed':True}

    # Rescale image in a 8 steps intensity level, if nothing is indicated in graycomtx_kwargs
    # NOTE: this is the default behaviour of CellProfiler
    if 'levels' not in graycomtx_kwargs:
        print("Default: Rescaling image to 8 intensity levels - indicate levels in graycomtx_kwargs to avoid this")
        image = quantize_image(image,
                                levels=8,
                                pmin=1,
                                pmax=99,
                                out_bottom=0,
                                out_dtype=np.uint8,
                                percentile_kwargs=None,
                                clip_kwargs=None,
                                rescale_kwargs=None,
                                floor_kwargs=None)

        # image = rescale_intensity(image,out_range=(0,7)).astype(np.uint8)
        graycomtx_kwargs = graycomtx_kwargs.copy()  # to avoid modifying the input dictionary
        graycomtx_kwargs['levels']=8

    if sep is None:
        sep='_'

    assert 'distances' not in graycomtx_kwargs, "distances can't be passed to graycomtx_kwargs, use the dedicated argument instead"
    assert 'angles' not in graycomtx_kwargs, "angles can't be passed to graycomtx_kwargs, use the dedicated argument instead"

    # Compute GLCM at different angles/distances
    glcm = graycomatrix(image,
                        distances=distances,
                        angles=angles,
                        **graycomtx_kwargs)

    # initialize a collection list
    haralick_measurements_list = []

    # iterate through the properties to measure
    for p in prop:

        # Extract Haralick feature
        haralick_feature = graycoprops(glcm, p)

        # Link measurements to their names in a dictionary
        # haralick_feature_df_val = []
        haralick_feature_df_val = haralick_feature.flatten()
        haralick_feature_df_col = []
        i=0
        for d in distances:
            j=0
            for a in angles:

                # Add distance and angle only if more than one is present
                if len(distances)==1 and len(angles)==1:
                    full_measurement_name = p
                elif len(distances)>1 and len(angles)==1:
                    da_name = f"{i}"
                    full_measurement_name = name_column(measurement=p,
                                                        suffix=da_name,
                                                        suffix_sep=sep)
                elif len(distances)==1 and len(angles)==1:
                    da_name = f"{j}"
                    full_measurement_name = name_column(measurement=p,
                                                        suffix=da_name,
                                                        suffix_sep=sep)
                else:
                    da_name = f"{i}{sep}{j}"
                    full_measurement_name = name_column(measurement=p,
                                                        suffix=da_name,
                                                        suffix_sep=sep)

                # haralick_feature_df_val.append(haralick_feature[i][j])
                haralick_feature_df_col.append(full_measurement_name)


                j=j+1

            i=i+1

        # Form a dataframe for the measurement
        haralick_feature_df = pd.Series(data=haralick_feature_df_val, index=haralick_feature_df_col)

        # Collect the dataframe in the collection list
        haralick_measurements_list.append(haralick_feature_df)

    # Concatenate all measurments
    haralick_measurements = pd.concat(haralick_measurements_list,axis=0)

    return haralick_measurements


def glcm_feature_map_ch(image:np.typing.ArrayLike,
                        props:str|Sequence|None=None,
                        distances:int|float|tuple|list|Sequence|None=None,
                        angles:int|float|tuple|list|Sequence|None=None,
                        mask:np.typing.ArrayLike|None=None,
                        window_shape:int|tuple=11,
                        graycomtx_kwargs:dict|None=None,
                        pad_kwargs:dict|None=None,
                        windows_kwargs:dict|None=None,
                        zeros_kwargs:dict|None=None,
                        concat_kwargs:dict|None=None)->np.array:

    """
    Compute a Haralick feature map for one object.

    Image must be 2D (as a corollary, the image is single channel). If 'levels' is
    passed to graycomtx_kwargs, only integer typed input images are
    supported and only positive valued images are supported.

    Mask is assumed to have 0 as background. Everything which is non-zero in the mask is considered foreground and part of a
    single object/region.

    Per each pixel in the object, a patch of the input image centered on the pixel and of size window_shape is extracted and
    the gray-level co-occurrence matrix (GLCM) is computed for that patch. Then, the haralick features indicated in props
    are measured from the GLCM and stored in the output feature map at the pixel position.

    Haralick features are based on skimage.feature.graycomatrix and skimage.feature.graycoprops (see their documentation for details
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycomatrix
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycoprops
    https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_glcm.html).

    The image is padded before extracting patches, to allow feature map computation on borders. By default, the padding width
    is half of the window size in all dimensions. The padding behaviour can be changed by passing arguments to the pad_kwargs.

    In the output array feature maps are stacked on the last dimension (position -1). Their order is:
    - measured property
        - distances
            - angles.

    In other words, when iterating:
    c = 0
    for p in props:
        for d in distances:
            for a in angles:
                output_array[..., c]
                c=c+1
    One gets the feature maps correctly matching the angle, distance and property.

    NOTE: by default, the image is rescaled to 8 intensity levels (0-7) before calculating the GLCM. To change this behaviour,
    set pass the 'levels' argument in graycomtx_kwargs. For example, if an image
    has pixel values in the range 0-255, setting 'levels' to 256 in graycomtx_kwargs
    will avoid rescaling and use the original pixel values for GLCM calculation.
    Note that 0 is always assumed to be the lowest intensity value and that no
    rescaling is performed if 'levels' is specified. 'levels' is simply the number of
    intensity levels used for GLCM calculation, starting from 0.
    The "levels" argument is passed to graycomatrix, ref to the documentation
    for additional details (https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycomatrix).

    Inputs:
    - image: 2D numpy array. The input image for which the haralick feature map has to be computed.
    If 'levels' is passed to graycomtx_kwargs, only integer typed input images are supported and only
    positive valued images are supported.

    - props: str, sequence or None. Single haralick feature (if str) or list of haralick features (if list) to measure. Optional.
    Default: 'contrast'.

    - distances: int, float, tuple, list, sequence or None. The distance (if int or float) or list of distances to use for haralick
    features measurement. Optional. Default: [1].

    - angles: int, float, tuple, list, sequence or None. The angle (if int or float) or list of angles to use for haralick
    features measurement. Optional. Default: [0].

    - mask: 2D numpy array or None. The mask indicating the object/region for which the haralick feature map has to be computed.
    If None, the entire image is considered as the object/region. Optional. Default: None.

    - window_shape: int or tuple. The size of the patch to extract around each pixel for GLCM computation. If int, the same
    size is used for all dimensions. Optional. Default: 11. NOTE: the window size must allow to calculate haralick
    features for the input distances and angles. Thus, it must be at least equal to the max distance used (in the desired
    angle). For this reason, a warning is printed if any of the window dimensions is smaller than the max distance. Nevertheless,
    the function will still try run, but could lead to errors.

    - graycomtx_kwargs: dict or None. Additional arguments to pass to graycomatrix. Optional. Default: {'symmetric':True,'normed':True}.
    NOTE: 'distances' and 'angles' can't be passed here, use the dedicated arguments instead.

    - pad_kwargs: dict or None. Additional arguments to pass to np.pad for image padding before patch extraction. Optional.
    Default: {'mode':'reflect'}. NOTE: by default, the padding width is set to half of the window size in all dimensions.

    - windows_kwargs: dict or None. Additional arguments to pass to skimage.util.shape.view_as_windows for patch extraction.
    Optional. Default: {}.

    - zeros_kwargs: dict or None. Additional arguments to pass to np.zeros for feature map array initialization. Optional.
    Default: {'dtype':float}. NOTE: 'a' can't be passed here, as the shape of the array is hard coded.

    - concat_kwargs: dict or None. Additional arguments to pass to np.concatenate for concatenating multiple haralick measurements
    per pixel. Optional. Default: {}. NOTE: 'axis' can't be passed here, as it is hard coded to be in position 0.

    Outputs:
    - feature_map: numpy array. The output haralick feature map. Its shape is the same as the input image, plus an extra
    dimension of size: number of distances calculated * number of angles calculated * number of properties measured.
    This extra dimension, which is always in position -1, is where the feature maps per each distance, angle and property
    are stacked.

    """

    # use default properties
    if props is None:
        props=['contrast']
    else:
        # if prop is str, store it in a list, for compatibility with following iteration
        if isinstance(props,str):
            props=[props]

    # use default distances
    if distances is None:
        distances=[1]
    else:
        # if distances are int or float
        # store distances in a list, for compatibility with skimage.feature.graycomatrix
        if isinstance(distances, int) or isinstance(distances,float):
            distances=[distances]

    # use default angles
    if angles is None:
        angles=[0]
    else:
        # if angles are int or float
        # store angles in a list, for compatibility with skimage.feature.graycomatrix
        if isinstance(angles,int) or isinstance(angles,float):
            angles=[angles]

    # if mask is None, get the entire image as a mask
    if hasattr(mask, "__len__"):
        mask=mask.copy()
    else:
        mask=np.ones(image.shape)

    # if windows_shape is an integer, form a tuple with as many integers as the dimensions of image
    if isinstance(window_shape,int):
        window_shape = tuple(window_shape for d in image.shape)

    # print a warning if the any of the window dimension is smaller than distance, as this could lead to the
    # impossibility of actually measuring the value
    if min(window_shape)<max(distances):
        print(f"WARNING: using a window with at least one dimension of size smaller than the max distance to caluculate could lead to errors. Min window dim: {min(window_shape)}. Max distance: {max(distances)}")

    # use defaults
    if graycomtx_kwargs is None:
        graycomtx_kwargs={'symmetric':True,'normed':True}

    if pad_kwargs is None:
        pad_kwargs={'mode':'reflect'}

    if windows_kwargs is None:
        windows_kwargs={}

    if zeros_kwargs is None:
        zeros_kwargs={'dtype':float}

    if concat_kwargs is None:
        concat_kwargs={}

    # Rescale image in a 8 steps intensity level, if nothing is indicated in graycomtx_kwargs
    # NOTE: this is the default behaviour of CellProfiler
    if 'levels' not in graycomtx_kwargs:
        print("Default: Rescaling image to 8 intensity levels - indicate levels in graycomtx_kwargs to avoid this")
        image = quantize_image(image,
                                levels=8,
                                pmin=1,
                                pmax=99,
                                out_bottom=0,
                                out_dtype=np.uint8,
                                percentile_kwargs=None,
                                clip_kwargs=None,
                                rescale_kwargs=None,
                                floor_kwargs=None)

        # image = rescale_intensity(image,out_range=(0,7)).astype(np.uint8)
        graycomtx_kwargs = graycomtx_kwargs.copy()  # to avoid modifying the input dictionary
        graycomtx_kwargs['levels']=8

    # Pad image to allow feature map computation on boarders - NOTE: the default, hard coded behavior of using half
    # of the window size (in all dimensions) as pad
    if 'pad_width' not in pad_kwargs:
        pad_kwargs = pad_kwargs.copy()  # to avoid modifying the input dictionary
        pad_kwargs['pad_width']=tuple([(ws//2, ws//2) for ws in window_shape])


    assert 'window_shape' not in windows_kwargs, "window_shape can't be passed to windows_kwargs, use window_shape argument instead"
    assert 'a' not in zeros_kwargs, "image is the input of numpy zeros array"
    assert 'distances' not in graycomtx_kwargs, "distances can't be passed to graycomtx_kwargs, use the dedicated argument instead"
    assert 'angles' not in graycomtx_kwargs, "angles can't be passed to graycomtx_kwargs, use the dedicated argument instead"
    assert 'axis' not in concat_kwargs, "axis can't be passed to concat kwargs as it is hard coded to be in position 0"


    padded = np.pad(image, **pad_kwargs)

    # get indexed patches of the input image
    windows = view_as_windows(padded, window_shape, **windows_kwargs)

    # Initialize a zero array to be updated for storing the feature map
    # NOTE: the feature array has the same shape of the input image, plus an extra dimension of
    # size: number of distances calculated * number of angles calculated. This extra dimension, which is always
    # in position -1, is where the feature maps per each distance and angle are stacked
    feature_map_shape = list(image.shape)
    feature_map_shape.append(len(distances)*len(angles)*len(props))
    feature_map = np.zeros(feature_map_shape, **zeros_kwargs)

    # === === ===
    # === THIS IS WERE THE FUNCTION STARTS TO ONLY WORK IN 2D ===
    # === THIS CAN'T BE AVOIDED AS graycomatrix ONLY ACCEPTS 2D IMAGES ===
    # === === ===

    # Get the coordinates of the mask (NOTE: if a mask is not passed, the mask becomes the entire image)
    rows, cols = np.nonzero(mask)  # only compute inside object
    # Iterate through the coordinates of the mask's pixels
    for i, j in zip(rows, cols):

        # Get a patch of the input image centered on the pixel coordinate and of size window_shape
        patch = windows[i, j]

        # Compute the gray-level co-occurrence matrix of the patch
        glcm = graycomatrix(patch, distances=distances, angles=angles,
                            **graycomtx_kwargs)

        # Initialize a collection list
        val_l = []

        # Iterate through the properties to measure:
        for p in props:
            # Get the haralick measurement for all distances and angles
            prop_i = graycoprops(glcm, p)

            # Flatten the value
            prop = prop_i.flatten()

            # Collect flatten property in collection list
            val_l.append(prop)

        # Concatenate measurements for multiple features - NOTE: When a single feature is measured, np.concatenate has
        # no effect
        val = np.concatenate(val_l,axis=0,**concat_kwargs) # it is known that axis is 0 since prop measurements have been flatten

        # Store haralick measurement in the feature map array
        feature_map[i, j,...] = val

    return feature_map



def glcm_feature_map(image: np.typing.ArrayLike,
                     props: str | Sequence | None = None,
                     distances: int | float | tuple | list | Sequence | None = None,
                     angles: int | float | tuple | list | Sequence | None = None,
                     mask: np.typing.ArrayLike | None = None,
                     channel_axis: int | None = None,
                     window_shape: int | tuple = 11,
                     graycomtx_kwargs: dict | None = None,
                     pad_kwargs: dict | None = None,
                     windows_kwargs: dict | None = None,
                     zeros_kwargs: dict | None = None,
                     glcm_concat_kwargs: dict | None = None,
                     stack_channels: bool = True,
                     stack_axis: int = -1,
                     stack_kwargs: dict | None = None,
                     feature_concat_axis: int = -1,
                     feature_concat_kwargs: dict | None = None) -> np.array:

    """
    Wrapper around glcm_feature_map_ch to support multi-channel images.

    If channel_axis is None:
        - behaves exactly like glcm_feature_map_ch

    If channel_axis is provided:
        - iterates over sub-arrays along channel_axis
        - computes glcm_feature_map_ch for each channel independently

    Channel handling:
    - If stack_channels=True (default):
        output shape: (..., F, C)
        where:
            F = number of features per channel
            C = number of channels
        By default the channel dimension is in the last position, but it can be moved to a different position using stack_axis.

    - If stack_channels=False:
        feature maps are concatenated along feature_concat_axis
        (default: last axis), resulting in flattened features across channels

    Inputs:
    - image: numpy array. Input image. Can be 2D or multi-channel.
    If 'levels' is passed to graycomtx_kwargs, only integer typed input images are supported and only
    positive valued images are supported.

    - props, distances, angles, mask, graycomtx_kwargs,
      pad_kwargs, windows_kwargs, zeros_kwargs:
      same as glcm_feature_map_ch.

    - window_shape.
        If channel_axis is None: same as glcm_feature_map_ch.
        Else: int or tuple.
            If int: same as glcm_feature_map_ch.
            Else: must match the number of dimensions of individual channels (aka, the number of
            dimensions of the image, minus 1). Then it behaves as glcm_feature_map_ch.

    - glcm_concat_kwargs: same as concat_kwargs in glcm_feature_map_ch.

    - channel_axis: int or None. Axis corresponding to channels.
      If None, the image is treated as single-channel. Default: None.

    - stack_channels: bool. If True, keeps channel dimension separate
      (recommended). If False, flattens channel features into one axis.
      Default: True.

    - stack_axis: int. Axis along which channel dimension is placed when stack_channels=True. Default: -1 (last axis).

    - feature_concat_axis: int. Axis along which feature maps are concatenated
      when stack_channels=False. Default: -1.

    - stack_kwargs: dict or None. Additional arguments to pass to np.stack
    when stack_channels=True. Optional. Default: {}. NOTE: 'axis'
    can't be passed here, as it is set by stack_axis argument.

    - feature_concat_kwargs: dict or None. Additional arguments to pass
    to np.concatenate when stack_channels=False. Optional. Default: {}. NOTE: 'axis'
    can't be passed here, as it is set by feature_concat_axis argument.

    Outputs:
    - feature_map: numpy array containing Haralick feature maps.

    NOTE: by default, None is passed to graycomtx_kwargs. This leads to the following
    behavior when rescaling the image intensity:
        - If no channel axis is specified, the entire image is rescaled to 8
        intensity levels (0-7) before calculating the GLCM.
        - If a channel axis is specified, each channel is independently
        rescaled to 8 intensity levels (0-7) before calculating the GLCM for that channel.
    Ref to the documentation of glcm_feature_map_ch for more details.
    """

    # Single-channel case → delegate directly
    if channel_axis is None:
        return glcm_feature_map_ch(
            image=image,
            props=props,
            distances=distances,
            angles=angles,
            mask=mask,
            window_shape=window_shape,
            graycomtx_kwargs=graycomtx_kwargs,
            pad_kwargs=pad_kwargs,
            windows_kwargs=windows_kwargs,
            zeros_kwargs=zeros_kwargs,
            concat_kwargs=glcm_concat_kwargs
        )

    # Set defaults
    if stack_kwargs is None:
        stack_kwargs:dict={}
    else:
        assert "axis" not in stack_kwargs, "axis can't be passed to stack_kwargs. Use stack_axis argument instead."

    if feature_concat_kwargs is None:
        feature_concat_kwargs:dict={}
    else:
        assert "axis" not in feature_concat_kwargs, "axis can't be passed to feature_concat_kwargs. Use feature_concat_axis argument instead."

    # ensure that window_shape is specific for channels, if channel axis is passed and
    # window shape is a tuple
    if not isinstance(window_shape, int):
        assert len(window_shape) == image.ndim - 1, \
        "If channel axis is passed and window_shape is a tuple, window_shape must match spatial dims (excluding channel axis)"

    # Prepare image
    image = np.asarray(image)
    image_moved = np.moveaxis(image, channel_axis, 0)  # (C, ...)

    # Handle mask (shared across channels)
    if hasattr(mask, "__len__"):
        mask = mask.copy()

    feature_maps = []

    # Compute per-channel feature maps
    for ch in image_moved:
        fm = glcm_feature_map_ch(
            image=ch,
            props=props,
            distances=distances,
            angles=angles,
            mask=mask,
            window_shape=window_shape,
            graycomtx_kwargs=graycomtx_kwargs,
            pad_kwargs=pad_kwargs,
            windows_kwargs=windows_kwargs,
            zeros_kwargs=zeros_kwargs,
            concat_kwargs=glcm_concat_kwargs
        )
        feature_maps.append(fm)

    # Stack channels → shape (..., F, C)
    feature_map = np.stack(feature_maps, axis=-1)

    # return stacked channels by default
    if stack_channels:

        # If stack_axis is not last, move axis
        if stack_axis != -1:
            feature_map = np.moveaxis(feature_map, -1, stack_axis)
        return feature_map

    # Flatten channels into feature axis
    # Move channel axis next to feature axis, then reshape
    *spatial_dims, F, C = feature_map.shape

    # reshape to (..., F*C)
    flattened = feature_map.reshape(*spatial_dims, F * C)

    # If feature_concat_axis is not last, move axis
    if feature_concat_axis != -1:
        flattened = np.moveaxis(flattened, -1, feature_concat_axis)

    return flattened



def glcm_object(region,
                channel_index: int = -1,
                props:str|Sequence|None=None,
                distances:int|float|tuple|list|Sequence|None=None,
                angles:int|float|tuple|list|Sequence|None=None,
                window_shape:int|tuple=11,
                graycomtx_kwargs:dict|None=None,
                pad_kwargs:dict|None=None,
                windows_kwargs:dict|None=None,
                zeros_kwargs:dict|None=None,
                glcm_concat_kwargs:dict|None=None)->tuple:
    """
    Wrapper for Dask: computes feature map for a region/segmented object in a given channel.

    See glcm_feature_map for details.

    IMPORTANT NOTES:
    - the channel axis is expected in position -1.
    - a channel axis is always expected. If a region has a single channel, there should anyway be a channel axis, in
    position -1, of size 1.
    - by default, None is passed to graycomtx_kwargs. As a consequence, the
    image/object is rescaled to 8 intensity levels (0-7) before calculating the GLCM.
    Ref to the documentation of glcm_feature_map_ch for more details.
    """

    # Get the region/object bounding box for the intensity image
    # (aka - the image for which haralick feature map has to be computed cropped to contain the region/object)
    sub_image = region.intensity_image[..., channel_index]  # pick channel

    # Get the region/object for the segmentation mask
    # (aka - the labelled image with object segmentation cropped to contain the region/object)
    mask = region.image

    # Compute glcm_feature_map for the region/segmented object
    fmap_local = glcm_feature_map(image=sub_image,
                                  mask=mask,
                                  props=props,
                                  distances=distances,
                                  angles=angles,
                                  channel_axis=None,  # channel axis is already handled by picking the channel in sub_image
                                  window_shape=window_shape,
                                  graycomtx_kwargs=graycomtx_kwargs,
                                  pad_kwargs=pad_kwargs,
                                  windows_kwargs=windows_kwargs,
                                  zeros_kwargs=zeros_kwargs,
                                  glcm_concat_kwargs=glcm_concat_kwargs)

    return (region.label, region.bbox, channel_index, fmap_local)


def parallel_glcm_feature_map(image:np.array,
                              label_image:np.array,
                              props:str|Sequence|None=None,
                              distances:int|float|tuple|list|Sequence|None=None,
                              angles:int|float|tuple|list|Sequence|None=None,
                              channel_axis:int|None=-1,
                              window_shape:int|tuple=11,
                              regionprops_kwargs:dict|None=None,
                              daskbag_kwargs:dict|None=None,
                              graycomtx_kwargs:dict|None=None,
                              pad_kwargs:dict|None=None,
                              windows_kwargs:dict|None=None,
                              zeros_kwargs:dict|None=None,
                              glcm_concat_kwargs:dict|None=None)->np.array:
    """
    Compute a Haralick feature map for all the label objects in a label_image. If the input image has multiple channels,
    the feature maps can be computed per each channel independently by passing the axis to channel_axis.

    ===
    === IMPORTANT NOTE ===
    The function takes advantage of Dask to parallelize the computation of feature maps per each object and (eventually) per
    each channel.
    ===

    Image must be 2D or 3D (multi-channel).
    If 'levels' is passed to graycomtx_kwargs, only integer typed input images are supported and only
    positive valued images are supported.

    Label_image is assumed to have 0 as background. Label objects are assumed to be positive integers.

    ===
    === IMPORTANT NOTE ===
    The feature maps are only computed inside the label objects. Pixels not belonging to any object (i.e. background pixels)
    will have value 0 in the output feature map.
    Per each pixel in the object, a patch of the input image centered on the pixel and of size window_shape is extracted and
    the gray-level co-occurrence matrix (GLCM) is computed for that patch. Then, the haralick features indicated in props
    are measured from the GLCM and stored in the output feature map at the pixel position.
    ===

    Haralick features are based on skimage.feature.graycomatrix and skimage.feature.graycoprops (see their documentation for details
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycomatrix
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycoprops
    https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_glcm.html).


    In the output array feature maps are stacked on the last dimension (position -1). Their order is:
    - measured property
        - distances
            - angles.

    In other words, when iterating:
    c = 0
    for p in props:
        for d in distances:
            for a in angles:
                output_array[..., c]
                c=c+1
    One gets the feature maps correctly matching the angle, distance and property.

    In the output array, a channel axis is always added in position -2 (i.e. before the feature maps axis).
    Thus, the output array will have shape: (Y, X, C, FM) where FM is the size of the feature maps axis
    (i.e. number of distances calculated * number of angles calculated * number of properties measured),
    Y and X are the height and width of the input image and C is the number of channels.

    The image is padded before extracting patches, to allow feature map computation on borders. By default, the padding width
    is half of the window size in all dimensions. The padding behaviour can be changed by passing arguments to the pad_kwargs.

    ===
    === IMPORTANT NOTE ===
    By default, None is passed to graycomtx_kwargs to the downstram glcm_feature_map_ch function.
    This leads to the following behavior when rescaling the image intensity:
        - If no channel axis is specified, each individual object bounding box
        is rescaled to 8 intensity levels (0-7) before calculating the GLCM.
        - If a channel axis is specified, per each channel, each individual
        object bounding box independently rescaled to 8 intensity levels (0-7)
        before calculating the GLCM for that channel.
    To avoid per-object rescaling, the 'levels' argument needs to be set to the
    desired value in graycomtx_kwargs. Note that setting 'levels' to any value will
    completely avoid any intensity rescaling, while also specifying the number of
    intensity levels to use for calculating the GLCM matrix.
    For example, if an image is uint8 data type, setting 'levels' to 256 in
    graycomtx_kwargs will avoid rescaling and indicate that per each object gray
    levels from 0 to 255 should be used for GLCM calculation. While setting 'levels'
    to 200 will avoid rescaling and indicate that per each object gray levels from 0 to 199
    should be used for GLCM calculation.
    Ref to the documentation of glcm_feature_map_ch for more details.
    ===

    The function has been tested on:
    - single channel images (2D numpy arrays) with no label objects (i.e. label_image is all zeros), with one label object
    and with multiple label objects.
    - multi-channel images (3D numpy arrays) with no label objects (i.e. label_image is all zeros), with one label object
    and with multiple label objects.

    === === ===
    Inputs:
    - image: 2D or 3D numpy array. The input image for which the haralick feature map has to be computed.
    If 'levels' is passed to graycomtx_kwargs, only integer typed input images are supported and only
    positive valued images are supported.

    - label_image: 2D numpy array. The labelled image indicating the objects/regions for which the haralick feature map
    has to be computed. Pixels with value 0 are considered background. Positive integers indicate label objects.
    label_image must have the same shape of the input image, minus the channel axis (if present).

    - props: str, sequence or None. Single haralick feature (if str) or list of haralick features (if list) to measure. Optional.
    Default: 'contrast'.

    - distances: int, float, tuple, list, sequence or None. The distance (if int or float) or list of distances to use for haralick
    features measurement. Optional. Default: [1].

    - angles: int, float, tuple, list, sequence or None. The angle (if int or float) or list of angles to use for haralick
    features measurement. Optional. Default: [0].

    - channel_axis: int or None. The axis of the input image corresponding to channels. If None, the input image is assumed
    to have no channel axis (i.e. it is single channel). Optional. Default: -1.

    - window_shape: int or tuple. The size of the patch to extract around each pixel for GLCM computation. If int, the same
    size is used for all dimensions. Optional. Default: 11. NOTE: the window size must allow to calculate haralick
    features for the input distances and angles. Thus, it must be at least equal to the max distance used (in the desired
    angle). For this reason, a warning is printed if any of the window dimensions is smaller than the max distance. Nevertheless,
    the function will still try run, but could lead to errors.

    - regionprops_kwargs: dict or None. Additional arguments to pass to skimage.measure.regionprops for region properties extraction.
    Optional. Default: {}.

    - daskbag_kwargs: dict or None. Additional arguments to pass to dask.bag.from_sequence for Dask bag creation. Optional. Default: {'npartitions':8}.
    NOTE: 'npartitions' can be adjusted depending on the number of workers and number of tasks (i.e. number of objects * number of channels).
    A good rule of thumb is to have about 4× the number of workers as npartitions, and about 10 tasks per partition. Thus, a good value for
    npartitions can be calculated as follows:

    npartitions = min(
                      max(n_tasks // 10, n_workers * 4),  # at least 4× workers, about 10 tasks per partition
                      n_tasks                             # can't exceed total tasks
                        )

    - graycomtx_kwargs: dict or None. Additional arguments to pass to graycomatrix. Optional. Default: {'symmetric':True,'normed':True}.
    NOTE: 'distances' and 'angles' can't be passed here, use the dedicated arguments instead.

    - pad_kwargs: dict or None. Additional arguments to pass to np.pad for image padding before patch extraction. Optional.
    Default: {'mode':'reflect'}. NOTE: by default, the padding width is set to half of the window size in all dimensions.

    - windows_kwargs: dict or None. Additional arguments to pass to skimage.util.shape.view_as_windows for patch extraction.
    Optional. Default: {}.

    - zeros_kwargs: dict or None. Additional arguments to pass to np.zeros for feature map array initialization. Optional.
    Default: {'dtype':float}. NOTE: 'a' can't be passed here, as the shape of the array is hard coded.

    - glcm_concat_kwargs: dict or None. Additional arguments to pass to np.concatenate for concatenating multiple haralick measurements
    per pixel. Optional. Default: {}. NOTE: 'axis' can't be passed here, as it is hard coded to be in position 0.



    === === ===
    Outputs:
    - feature_map: numpy array. The output haralick feature map. Its shape is (Y, X, C, FM) where Y and X are the height
    and width of the input image, C is the number of channels and FM is the size of the feature maps axis
    (i.e. number of distances calculated * number of angles calculated * number of properties measured).
    This extra dimension, which is always in position -1, is where the feature maps per each distance, angle and property
    are stacked.

    """
    assert (isinstance(channel_axis, int) or channel_axis==None), "channel_axis must be either int or None"

    # Copy image and label
    image = image.copy()
    label_image = label_image.copy()

    # Set defaults
    if regionprops_kwargs is None:
        regionprops_kwargs={}

    if daskbag_kwargs is None:
        daskbag_kwargs={'npartitions': 8}


    # move channel axis to the last position, if present - or add an axis of size 1 in the last position of image if no
    # channel axis is present
    if isinstance(channel_axis, int):
        image_with_ch_last = np.moveaxis(image, channel_axis, -1)

    else:
        image_with_ch_last = np.expand_dims(image,axis=-1)

    # # Rescale image in a 8 steps intensity level, if nothing is indicated in graycomtx_kwargs
    # # NOTE: this is the default behaviour of CellProfiler
    # # NOTE: this rescaling is done per each channel individually if channel_axis is not None!!!
    # if 'levels' not in graycomtx_kwargs:

    #     # print a warning
    #     print("Default: Rescaling image to 8 intensity levels - indicate levels in graycomtx_kwargs to avoid this")

    #     # unstack channels and rescale them individually if a channel axis is present
    #     if isinstance(channel_axis, int):

    #         # unstack channels
    #         unstacked_channels = [image_with_ch_last[..., ch] for ch in range(image_with_ch_last.shape[-1])]
    #         # rescale each channel individually
    #         rescaled_channels = [rescale_intensity(unstacked_channels[ch], out_range=(0,7)).astype(np.uint8) for ch in range(image_with_ch_last.shape[-1])]
    #         # restack channels
    #         image_with_ch_last = np.stack(rescaled_channels, axis=-1)

    #     # else, rescale the single channel image
    #     else:
    #         image_with_ch_last = rescale_intensity(image_with_ch_last,out_range=(0,7)).astype(np.uint8)

    #     graycomtx_kwargs = graycomtx_kwargs.copy()  # to avoid modifying the input dictionary
    #     # set the levels parameter in graycomtx_kwargs
    #     graycomtx_kwargs['levels']=8


    # get the number of channels
    n_channels = image_with_ch_last.shape[-1]

    # get the properties of the individual objects in label_image
    regions = regionprops(label_image, intensity_image=image_with_ch_last, **regionprops_kwargs)

    # build (region, channel) pairs and include them in a list of tasks
    tasks = [(region, ch) for region in regions for ch in range(n_channels)]

    # include tasks in a dask.bag
    bag = db.from_sequence(tasks, **daskbag_kwargs)

    # compute tasks in the dask.bag in parallel
    results = bag.map(lambda rc: glcm_object(rc[0],
                                             channel_index=rc[1],
                                             props=props,
                                             distances=distances,
                                             angles=angles,
                                             window_shape=window_shape,
                                             graycomtx_kwargs=graycomtx_kwargs,
                                             pad_kwargs=pad_kwargs,
                                             windows_kwargs=windows_kwargs,
                                             zeros_kwargs=zeros_kwargs,
                                             glcm_concat_kwargs=glcm_concat_kwargs)).compute()

    # Assemble full-size feature map

    # Initialize a zero array to be updated for storing the feature map
    # NOTE: the feature array has the same shape of the input image, plus an extra dimension of
    # size: number of distances calculated * number of angles calculated. This extra dimension, which is always
    # in position -1, is where the feature maps per each distance and angle are stacked
    fmap_shape = list(image_with_ch_last.shape) + [len(distances) * len(angles) * len(props)]
    fmap = np.zeros(fmap_shape, **zeros_kwargs)

    for label, bbox, channel_axis, fmap_local in results:
        minr, minc, maxr, maxc = bbox
        fmap[minr:maxr, minc:maxc, channel_axis, :] += fmap_local

    return fmap

def haralick_compute_feature_count(kwargs,
                                   n_channels):
    """
    helper to get the number of feature functions computed by glcm_feature_map_ch,
    glcm_feature_map and parallel_glcm_feature_map.
    This is only used in combination with dask.map_overlap.
    """
    props = _normalize_to_list(kwargs.get('props'), ['contrast'])
    distances = _normalize_to_list(kwargs.get('distances'), [1])
    angles = _normalize_to_list(kwargs.get('angles'), [0])

    F_per_channel = len(props) * len(distances) * len(angles)

    if kwargs.get('channel_axis')==None:
        return F_per_channel, None
    else:
        if kwargs.get('stack_channels', True):
            return F_per_channel, True
        else:
            return F_per_channel * n_channels, False


def haralick_prop_ch_map(props:Sequence,
                            distances:Sequence,
                            angles:Sequence,
                            sep:str="_")->dict:
    """
    Builds a dictionary mapping the haralick feature map channel index to the corresponding property, distance and angle.
    This is useful for renaming the columns of regionprops_table output when measuring haralick feature maps.
    """

    # Initialize the output dictionary, to be updated
    output_dict = {}

    # Initialize a positional counter
    pos_counter = 0

    # Iterate throught properties
    for p in props:
        # Iterate through distances
        for d_pos, d in enumerate(distances):
            # Iterate through angles - NOTE: the actual angle could be a decimal number, thus the position will be
            # reported in the name
            for a_pos, a in enumerate(angles):
                # Link positional counter to property, distance and angle
                output_dict[pos_counter]=f"{p}{sep}{d_pos}{sep}{a_pos}"

                # Udate the positional counter
                pos_counter=pos_counter+1

    return output_dict

def haralick_regionprops_channel(image:np.array,
                                 label_image:np.array,
                                 props:Sequence,
                                 distances:Sequence,
                                 angles:Sequence,
                                 sep:str|None=None,
                                 regionprops_kwargs:dict|None=None,
                                 suffix:str|None=None)->pd.DataFrame:
    """
    Computes regionprops_table for a a stack of haralick feature maps and renames the columns of the
    output table appropriately.

    Feature maps are assumed to be stacked on the last dimension (position -1) of image, in the following order:
    - measured property
        - distances
            - angles.

    This order is the same as the one produced by glcm_feature_map_ch.

    Image must be 2D grayscale image, with and extra dimension in position -1 containing the haralick feature maps.
    If 'levels' is passed to graycomtx_kwargs, only integer typed input images are supported and only
    positive valued images are supported.

    Label_image must have the same shape of image excluding the last dimension (i.e. the haralick feature maps dimension).

    regionprops_kwargs must contain 'separator'.

    regionprops_kwargs must contain 'properties':'label'

    suffix: if provided, is added at the end of each column name using regionprops_kwargs 'separator' as separator. This is meant
    to be used for adding the channel index as suffix when measuring haralick features on multi-channel images.

    === === ===
    This function is meant to be the building block for measuring haralick features on multi-channel images,
    by measuring haralick features per each channel independently and then merging the results.
    See measure_haralick_features (below) for details.
    === === ===
    """

    # use defaults
    if sep is None:
        sep="_"

    if regionprops_kwargs is None:
        regionprops_kwargs={'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'],
                            'separator':'-'}

    # ensure that 'separator' is in regionprops_kwargs and that it is different than sep
    # NOTE: by default regionprops_kwargs separator is set to '-'
    # NOTE: print a warning if regionprops separator is set to '_', as this can lead to wrong column names
    if 'separator' not in regionprops_kwargs:
        assert sep!='-', "using '-' as sep can only be done together with passing a 'separator' different than '-' to regionprops_kwargs"
        regionprops_kwargs = regionprops_kwargs.copy() # to avoid modifying the input dictionary
        regionprops_kwargs['separator']='-'
    else:
        assert regionprops_kwargs['separator']!=sep, "sep and regionprops's separator must be different"
        if regionprops_kwargs['separator']=='_':
            print("WARNING: using '_' as regionprops separator can lead to wrong column names")


    # measure intensities of hessian eigenvalues
    haralick_measurement_i = pd.DataFrame(regionprops_table(label_image,
                                                            intensity_image=image,
                                                            **regionprops_kwargs))

    # get a dictionary mapping the haralick feature maps to their position on the channel axis
    haralick_prop_ch_map_dict = haralick_prop_ch_map(props=props,
                                                     distances=distances,
                                                     angles=angles,
                                                     sep=sep)

    # initialize a column mapper
    column_mapper = {}

    # iterate through haralick_measurement columns
    for c in haralick_measurement_i.columns:

        # split column name to separate the regionprops measurement and the glcm feature map index
        c_split = c.split(sep=regionprops_kwargs['separator'])

        # don't modify the column name if no glcm feature map index position is present
        # these are non-intensity measurements as 'area', 'perimenter' etc... in addition, they are the 'label' column and the centroid
        if len(c_split)==1:
            new_c=c

        else:
            # get the new column name by mapping the glcm feature map index to the corresponding property, distance and angle

            # add suffix if provided
            if suffix!=None:
                new_c = f"{c_split[0]}{sep}{haralick_prop_ch_map_dict[int(c_split[-1])]}{regionprops_kwargs['separator']}{suffix}"
            else:
                new_c = f"{c_split[0]}{sep}{haralick_prop_ch_map_dict[int(c_split[-1])]}"

        column_mapper[c]=new_c

    # rename columns of haralick_measurement_i
    haralick_measurement = haralick_measurement_i.rename(column_mapper,axis=1, copy=True)

    return haralick_measurement



def measure_haralick_features(image:np.array,
                              label_image:np.array,
                              props:str|Sequence|None=None,
                              distances:int|float|tuple|list|Sequence|None=None,
                              angles:int|float|tuple|list|Sequence|None=None,
                              channel_axis:int|None=-1,
                              window_shape:int|tuple=11,
                              regionprops_kwargs:dict|None=None,
                              erosion_kwargs:dict|None=None,
                              merge_kwargs:dict|None=None,
                              glcm_regionprops_kwargs:dict|None=None,
                              glcm_daskbag_kwargs:dict|None=None,
                              glcm_graycomtx_kwargs:dict|None=None,
                              glcm_pad_kwargs:dict|None=None,
                              glcm_windows_kwargs:dict|None=None,
                              glcm_zeros_kwargs:dict|None=None,
                              glcm_concat_kwargs:dict|None=None,
                              erosion_warning:bool=True,
                              sep:str|None=None)->pd.DataFrame:
    """
    Given an image (2D or 3D multi-channel) and a labelled image (2D, same shape as image excluding channel axis if present),
    the function:
    1) computes haralick feature maps per each label object of the labelled image and per each channel of the input image. This
    is done by calling parallel_glcm_feature_map. Thus, per each channel, a feature map is computed per each
    haralick property, distance and angle indicated in the input.
    2) erodes the labelled image to avoid border effects in haralick feature maps.
    3) measures region properties on the haralick feature maps within the eroded labelled image, per each channel independently.
    4) merges the region properties measurements per each channel into a single dataframe.

    The function returns a pandas DataFrame. In this dataframe, the rows are individual label objects in label_image.
    The columns are individual region properties measurements on each haralick feature map. As one feature map is computed
    per each channel, and per each property, distance and angle indicated in the input, the number of columns is:
    number_of_channel * number_of_property * number_of_distance * number_of_angle. An extra column 'label' is also present,
    indicating the label of each object.
    Column names are built as follows:
    <regionprops measurement><sep><haralick property><sep><distance index><sep><angle index><regionprops separator><channel index>
    where:
    - <regionprops measurement> is the region property measured on the haralick feature map (e.g. intensity_mean, intensity_max, etc...)
    - <sep> is the separator indicated in the input (default: '_')
    - <haralick property> is the haralick property measured (e.g. contrast, dissimilarity, etc...)
    - <distance index> is the index of the distance used for haralick measurement (e.g. 0 for the first distance in the input distances)
    - <angle index> is the index of the angle used for haralick measurement (e.g. 0 for the first angle in the input angles)
    - <regionprops separator> is the separator indicated in regionprops_kwargs (default: '-'). This is
    only present if image has multiple channels.
    - <channel index> is the index of the channel on which the haralick feature map has been computed (e.g. 0 for the
    first channel). This is only present if image has multiple channels.

    Example column name: intensity_mean_contrast_0_0-1 (indicates intensity mean measured on the haralick contrast
    feature map computed with distance 0 and angle 0, on channel 1 of the input image).

    NOTE: It is possible to calculate regionprops properties non-related to intensity (ref to skimage.measure.regionprops
    https://scikit-image.org/docs/0.25.x/api/skimage.measure.html#skimage.measure.regionprops). These measurements (e.g. area, perimeter)
    will be returned on multiple copies, one per each channel. This behaviour will be fixed in the future,
    yet, for now, it is ignored as these measurements shouldn't be used in the context of this function.

    ===
    === IMPORTANT NOTE ===

    - The function calls parallel_glcm_feature_map to compute GLCM feature maps in parallel on the label objects and channels.
    Thus, Dask is used for parallelization. See parallel_glcm_feature_map for more details.

    - The function calls parallel_glcm_feature_map to compute GLCM feature maps. Therefore, GLCM feature maps are only computed
    inside the label objects. Pixels not belonging to any object (i.e. background pixels) will have value 0 in the output
    feature map. Ref to parallel_glcm_feature_map for more details.

    - Haralick features are based on skimage.feature.graycomatrix and skimage.feature.graycoprops (see their documentation for details
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycomatrix
    https://scikit-image.org/docs/0.25.x/api/skimage.feature.html#skimage.feature.graycoprops
    https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_glcm.html).

    - By default, the individual object-bounding boxes are rescaled to 8
    intensity levels (0-7) before calculating the GLCM.
    This is the default behaviour of CellProfiler.
    Ref to the documentation of parallel_glcm_feature_map and glcm_feature_map_ch for more details.

    - As label objects are eroded before measuring region properties on haralick feature maps, small objects could
    disappear after erosion. In this case, no region properties will be measured for these objects and they will be
    missing in the output dataframe. A warning is printed if any object disappears after erosion, unless erosion_warning
    is set to False.

    - If 'levels' is passed to graycomtx_kwargs, only integer typed input images are supported and only
    positive valued images are supported.
    ===


    === === ===
    Inputs:
    - image: 2D (grayscale) or 3D (multi-channel) numpy array. The input image for which the haralick feature maps have to be computed.
    If 'levels' is passed to graycomtx_kwargs, only integer typed input images are supported and only
    positive valued images are supported.

    - label_image: 2D numpy array. The labelled image indicating the objects/regions for which the haralick feature maps
    have to be computed. Pixels with value 0 are considered background. Positive integers indicate label objects. Label_image
    must be 2D and have the same shape of image if image is 2D, or the same shape of image excluding the channel axis if image
    is 3D.

    - props: str, sequence or None. Single haralick feature (if str) or list of haralick features (if list) to measure. Optional.
    Default: 'contrast'.

    - distances: int, float, tuple, list, sequence or None. The distance (if int or float) or list of distances to use for haralick
    features measurement. Optional. Default: [1].

    - angles: int, float, tuple, list, sequence or None. The angle (if int or float) or list of angles to use for haralick
    features measurement. Optional. Default: [0].

    - channel_axis: position of the channel axis in image. If image is 2D, channel_axis must be None. Optional. Default: -1.

    - window_shape: int or tuple. The size of the patch to extract around each pixel for GLCM computation. If int, the same
    size is used for all dimensions. Optional. Default: 11.

    - regionprops_kwargs: dict or None. Additional arguments to pass to skimage.measure.regionprops_table for region properties extraction.
    Optional. Default: {'properties':['label', 'intensity_mean', 'intensity_max', 'intensity_min', 'intensity_std'],'separator':'-'}.
    NOTE: 1) regionprops_kwargs must contain 'properties':'label'. 2) if 'separator' is passed, it must be different from sep.
    3) If 'separator' is passed and it is set to '_', a warning is printed as this can lead to wrong column names. Yet, the function will
    still try to run.

    - erosion_kwargs: dict or None. Additional arguments to pass to skimage.morphology.erosion for label image erosion.
    Optional. Default: {'footprint':disk(9)}.

    - merge_kwargs: dict or None. Additional arguments to pass to pd.merge for merging the region properties measurements
    per each channel into a single dataframe. Optional. Default: {}.

    - glcm_regionprops_kwargs: dict or None. Additional arguments to pass to the regionprops_kwargs argument of parallel_glcm_feature_map.
    Optional. Default: {}.

    - glcm_daskbag_kwargs: dict or None. Additional arguments to pass daskbag_kwargs argument of parallel_glcm_feature_map.
    These properties are passed to dask.bag.from_sequence for Dask bag creation. Optional. Default: {'npartitions':8}.
    NOTE: 'npartitions' can be adjusted depending on the number of workers and number of tasks (i.e. number of objects * number of channels).
    A good rule of thumb is to have about 4× the number of workers as npartitions, and about 10 tasks per partition. Thus, a good value for
    npartitions can be calculated as follows:

    npartitions = min(
                      max(n_tasks // 10, n_workers * 4),  # at least 4× workers, about 10 tasks per partition
                      n_tasks                             # can't exceed total tasks
                        )

    - glcm_graycomtx_kwargs: dict or None. Additional arguments to pass to graycomtx_kwargs argument of parallel_glcm_feature_map.
    These properties are passed to graycomatrix. Optional. Default: {'symmetric':True,'normed':True}.
    NOTE: 'distances' and 'angles' can't be passed here, use the dedicated arguments instead.

    - glcm_pad_kwargs: dict or None. Additional arguments to pass to pad_kwargs argument of parallel_glcm_feature_map.
    These properties are passed to np.pad for image padding before patch extraction. Optional.
    Default: {'mode':'reflect'}. NOTE: by default, the padding width is set to half of the window size in all dimensions.

    - glcm_windows_kwargs: dict or None. Additional arguments to pass to window_shape argument of parallel_glcm_feature_map. These
    properties are to window_shape passed skimage.util.shape.view_as_windows for patch extraction.
    Optional. Default: {}.

    - glcm_zeros_kwargs: dict or None. Additional arguments to pass zero_kwargs argument of parallel_glcm_feature_map.
    These properties are passed to np.zeros for feature map array initialization. Optional.
    Default: {'dtype':float}. NOTE: 'a' can't be passed here, as the shape of the array is hard coded.

    - glcm_concat_kwargs: dict or None. Additional arguments to pass to concat_kwargs of of parallel_glcm_feature_map.
    These properties are passed to np.concatenate for concatenating multiple haralick measurements
    per pixel. Optional. Default: {}. NOTE: 'axis' can't be passed here, as it is hard coded to be in position 0.

    - erosion_warning: bool. If True, a warning is printed if any label object disappears after erosion.
    Optional. Default: True.

    - sep: str or None. The separator to use when building the column names of the output dataframe. Optional. Default: '_'.


    === === ===
    Outputs:
    - haralick_measurements: pandas DataFrame. The output dataframe containing the region properties measurements
    on the haralick feature maps per each channel. Rows are individual label objects in label_image.
    Columns are individual region properties measurements on each haralick feature map. As one feature map is computed
    per each channel, and per each property, distance and angle indicated in the input, the number of columns is:
    number_of_channel * number_of_property * number_of_distance * number_of_angle. An extra column 'label' is also present,
    indicating the label of each object.


    === === ===
    The function has been tested on:
    - single channel images (2D numpy arrays) with no label objects (i.e. label_image is all zeros), with one label object
    and with multiple label objects.
    - multi-channel images (3D numpy arrays) with no label objects (i.e. label_image is all zeros), with one label object
    and with multiple label objects.

    """

    # Copy image and label
    original_image = image.copy()
    original_label_image = label_image.copy()

    # move channel axis to the last position, if present, and change channel_axis accordingly
    if isinstance(channel_axis, int):
        original_image = np.moveaxis(original_image, channel_axis, -1)
        ch_axis = -1
    else:
        ch_axis = channel_axis  # None

    if erosion_kwargs is None:
        erosion_kwargs={'footprint':disk(9)}

    if merge_kwargs is None:
        merge_kwargs={}


    # # Rescale image in a 8 steps intensity level, if nothing is indicated in graycomtx_kwargs
    # # NOTE: this is the default behaviour of CellProfiler
    # # NOTE: this rescaling is done per each channel individually if channel_axis is not None!!!
    # if 'levels' not in glcm_graycomtx_kwargs:

    #     # print a warning
    #     print("Default: Rescaling image to 8 intensity levels - indicate levels in glcm_graycomtx_kwargs to avoid this")

    #     # unstack channels and rescale them individually if a channel axis is present
    #     if isinstance(channel_axis, int):

    #         # unstack channels
    #         unstacked_channels = [original_image[..., ch] for ch in range(original_image.shape[-1])] # the channel axis is now in the last position

    #         # rescale each channel individually
    #         rescaled_channels = [rescale_intensity(unstacked_channels[ch], out_range=(0,7)).astype(np.uint8) for ch in range(original_image.shape[-1])]

    #         # restack channels
    #         original_image = np.stack(rescaled_channels, axis=-1)

    #     # else, rescale the single channel image
    #     else:
    #         original_image = rescale_intensity(original_image, out_range=(0,7)).astype(np.uint8)

    #     glcm_graycomtx_kwargs = glcm_graycomtx_kwargs.copy()  # to avoid modifying the input dictionary
    #     # set the levels parameter in glcm_graycomtx_kwargs
    #     glcm_graycomtx_kwargs['levels']=8

    assert (isinstance(channel_axis, int) or channel_axis==None), "channel_axis must be either int or None"

    # ensure that regionprops_kwargs contains 'properties' and 'label'
    if regionprops_kwargs is not None:
        assert 'properties' in regionprops_kwargs, "properties must be in regionprops_kwargs"
        assert 'label' in regionprops_kwargs['properties'], "label must be in regionprops_kwargs['properties']"


    # calculate haralick feature maps
    # NOTE: the output haralick_feature_map has the same shape of original_image, plus 2 extra dimensions
    # in the second last position there is the channel axis (if no channel_axis is present in original_image,
    # this dimenstion is of size 1) and in the last position there is the haralick feature maps axis
    haralick_feature_map = parallel_glcm_feature_map(image=original_image,
                                                     label_image=original_label_image,
                                                     props=props,
                                                     distances=distances,
                                                     angles=angles,
                                                     channel_axis=ch_axis,
                                                     window_shape=window_shape,
                                                     regionprops_kwargs=glcm_regionprops_kwargs,
                                                     daskbag_kwargs=glcm_daskbag_kwargs,
                                                     graycomtx_kwargs=glcm_graycomtx_kwargs,
                                                     pad_kwargs=glcm_pad_kwargs,
                                                     windows_kwargs=glcm_windows_kwargs,
                                                     zeros_kwargs=glcm_zeros_kwargs,
                                                     glcm_concat_kwargs=glcm_concat_kwargs)

    # erode labels
    eroded_label_image = erosion(original_label_image, **erosion_kwargs)

    # check for cell loss due to erosion if erosion_warning is set to True
    if erosion_warning:

        # get the number of labelled objects before erosion
        labels_before = np.unique(original_label_image)

        # get the number of labelled objects after erosion
        labels_after = np.unique(eroded_label_image)

        # print a warning message if the number of labels before and after erosion don't correspond
        if labels_before.shape != labels_after.shape:
            print(f"WARNING: possible loss of labelled object due to erosion. Number of objects before erosion: {labels_before[0]-1}. Number of objects after erosion: {labels_after[0]-1}")


    # Get the feature maps for the first channel - NOTE: channels are expected in the second last axis, also,
    # feature maps are expected in the last position - this is the output of parallel_glcm_feature_map
    first_channel_feature_map = haralick_feature_map[...,0,:]

    # if only one channel is present, calculate haralick measurements only on it
    if haralick_feature_map.shape[-2]==1:

        # Calculate region properties on eroded labels, return them as a table with appropriate names
        # NOTE: no channel suffix is added to the column names since there is only one channel
        haralick_measurements = haralick_regionprops_channel(image=first_channel_feature_map,
                                                                    label_image=eroded_label_image,
                                                                    props=props,
                                                                    distances=distances,
                                                                    angles=angles,
                                                                    sep=sep,
                                                                    regionprops_kwargs=regionprops_kwargs)

    else:

        # Calculate regionproperties on eroded labels, return them as a table with appropriate names
        # NOTE: a channel suffix is added to the column names
        ch0_haralick_measurements = haralick_regionprops_channel(image=first_channel_feature_map,
                                                                    label_image=eroded_label_image,
                                                                    props=props,
                                                                    distances=distances,
                                                                    angles=angles,
                                                                    sep=sep,
                                                                    regionprops_kwargs=regionprops_kwargs,
                                                                    suffix = "0")

        # Initialize a list to collect measurements
        results = []

        # Iterate through channels in the haralick feature map - NOTE: channels are expected in the second last axis,
        # this is the output of parallel_glcm_feature_map
        for ch in range(haralick_feature_map.shape[-2])[1:]:

            # Get the feature maps for the channel - NOTE: channels are expected in the second last axis, also, feature maps are
            # expected in the last position - this is the output of parallel_glcm_feature_map
            channel_feature_map = haralick_feature_map[...,ch,:]

            # Calculate regionproperties on eroded labels, return them as a table with appropriate names
            ch_haralick_measurements = haralick_regionprops_channel(image=channel_feature_map,
                                                                    label_image=eroded_label_image,
                                                                    props=props,
                                                                    distances=distances,
                                                                    angles=angles,
                                                                    sep=sep,
                                                                    regionprops_kwargs=regionprops_kwargs,
                                                                    suffix = str(ch))

            # Collect measurements in collection list
            haralick_measurements = ch0_haralick_measurements.merge(ch_haralick_measurements, on='label', how='left', **merge_kwargs)

    return haralick_measurements
