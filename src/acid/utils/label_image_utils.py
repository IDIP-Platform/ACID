from collections.abc import Sequence, Callable
import numpy as np
import scipy
import pandas as pd
import scipy.ndimage
from scipy.ndimage import laplace
from skimage.measure import regionprops, regionprops_table


def my_function(input_image):
    functioned_image = laplace(input_image)
    median_functioned_image = np.median(functioned_image)
    return median_functioned_image


def label_image_custom_measurement(label_image:np.array,
                                   intensity_image:np.array,
                                   function:Callable,
                                   column_name:str,
                                   index:int|Sequence|None=None,
                                   out_dtype:np.dtype|None=None,
                                   default:int|float|None=None,
                                   pass_position:bool=False,
                                   label_clm_name:str|None=None)-> pd.DataFrame:
    """
    Sequentially applies an arbitrary function (that works on array_like input) to subsets of an N-D image
    array specified by labels and index. The option exists to provide the function with positional parameters
    as the second argument.

    In other words, the function applies scipy.ndimage.labeled_comprehension
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.labeled_comprehension.html from
    which is the above description), but returns a dataframe with a column for the labels and a second
    column for the associated results of the function. For example it can be used to fast perform custom
    measurements of individual objects in an image, by passing the image to measure as intensity_image and
    the individual objects (e.g. segmented cells), to label_image as a labelled image.

    NOTE: this function expects a label image as input and will not transform the label_image input as
    it was conceptualized to be used multiple times on the same labelled image in order to then
    concatenate together the resulting data frames. This task relies on the assumption that the
    object labels remain invariant across different measurements.

    Inputs:
    - label_image. n-dimensional labelled array. The image to use to identify objects guiding the application
    of the function. Background pixels are expected to have value 0. It is passed to the argument "labels" in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.labeled_comprehension.html

    - intensity_image. n-dimensional array of the same shape of label_image. The image to which function is
    applied. For example, for extracting measurements. It is passed to the argument "input" in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.labeled_comprehension.html

    - function. Callable. Any function which receives an array as input and outputs a single value.
    It is passed to the argument "func" in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.labeled_comprehension.html

    - column_name. str. The name to give to column containing the results of function in the output dataframe.

    - index. int, sequence or None. Optional. Default None. Optional parameter specifying a subset of
    labels to analyse. It is passed to the argument "index" in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.labeled_comprehension.html

    - out_dtype. dtype or None. Optional. Default None. The data type of the output of function applied to labels.
    It is passed to the argument "out_dtype" in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.labeled_comprehension.html

    - default. int, float or None. Optional. Default None. The value to return when an element or index
    is not present in label_image. It is passed to the argument "default" in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.labeled_comprehension.html

    - pass_position. bool. Optional. Default False. If True, pass linear indices to function as a
    second argument. It is passed to the argument "pass_position" in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.labeled_comprehension.html

    - label_clm_name. str or None. Optional. Default "label". The name of the column where label values of
    measured objects are stored in the output dataframe.
    
    Output. pandas dataframe. Each row is a labelled element present in label_image. By default all the
    labelled elements are present. Index can be used to restrict them to a subset. Two columns are present:
    - label. The value of the element in label_image.
    - column_name. The result function applied to intensity_image only for the pixels specified by label.
    
    """

    # set default label column name
    if label_clm_name is None:
        label_clm_name='label'

    # get label_image's labels (here called indexes) if they are not provided as a function parameter
    if hasattr(index, "__len__"):
        pass

    else:
        number_of_index = len(np.unique(label_image))
        index = np.arange(1, number_of_index)

    # extract measurement per each desired labelled object
    label_measurement = scipy.ndimage.labeled_comprehension(input=intensity_image,
                                                            labels=label_image,
                                                            index=index,
                                                            func=function,
                                                            out_dtype=out_dtype,
                                                            default=default,
                                                            pass_positions=pass_position)
    
    # store measurements and corresponding labels in a dictionary
    label_measurement_dict = {label_clm_name:index, column_name:label_measurement}

    # use the disctionary to form a pandas dataframe
    label_measurement_df = pd.DataFrame.from_dict(label_measurement_dict)

    return label_measurement_df


def get_secondary_object(primary_mask:np.array,
                         secondary_label_image:np.array,
                         threshold_overlap_fraction:float=0.7,
                         properties:str|Sequence|None=None,
                         extra_properties:Callable|Sequence|None=None,
                         primary_label:int|None=None,
                         primary_object_name:str|None=None,
                         secondary_object_name:str|None=None,
                         **kwargs)-> tuple:
    """
    Given a primary segmentation mask (primary_mask) and a secondary labelled image (secondary_label_image)
    returns the individual, separated elements (or objects, or connected elements) in the
    secondary_label_image whose area overlaps for a fraction strictily higher than  threshold_overlap_fraction
    with the primary_mask.

    Optional: it is possible, during this process, to also measure properties of the identified secondary
    objects.

    Inputs:
    - primary_mask. n-dimensional np.array. Binary boolean mask. Background values are
    assumed to be 0 forground values can be any positive integer. NOTE: make sure that it is intepreted by
    numpy as a boolean mask (e.g. change the dytpe to bool or np.uint8) alternatively it could be interpreted
    as fancy indexing.
    
    - secondary_label_image. Labelled image (it can be obtained using https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label).
    n-dimensional np.array of the same shape of primary_mask and
    connected region . Background values are spected to be 0s.

    - threshold_overlap_fraction. float. Optional. Default 0.7. The highpass threshold for the fraction of
    the area of an object in secondary_label_image which must overlap with the primary mask for the object
    to be included in the output (aka to be linked to primary_mask as a secondary object).

    - properties. string, sequence of strings or None. Optional. Default None.
    Additional properties to measure for the individual, separated objects of the identified
    secondary objects.
    NOTE: this must be a property of skimage.measure.regionprops (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
    NOTE 2: 'area,''label', 'centroid' are not allowed to be passed to properties!

    - extra_properties. callable, sequence of callables or None. Optional. Default None.
    Additional properties to measure for the individual, separated objects of the identified
    secondary objects.
    NOTE: this parameter is passed to extra_properties of skimage.measure.regionprops (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).
    NOTE 2: 'area,''label', 'centroid' are not allowed to be passed to extra_properties!

    - primary_label. int or None. Optional. Default None. The label associated to the primary_mask.
    If None, the label is automatically extracted as the only value other than 0 present in primary_mask.

    - primary_object_name. string or None. Optional. Default None. The function returns a dataframe
    with a column containing the label value of the primary object.
    If primary_object_name is None (default) this is called "label_primary_object".
    primary_object_name can be used to specify a name which will substitute {primary_object} in the
    column name of the output dataframe.

    - secondary_object_name. string or None. Optional. Default None. The function returns a dataframe
    with: 1) a column containing the label value of all indentified secondary objects, 2) a column containing
    the areas of the all secondary object, 3) a column containing the the fraction of secondary object areas
    overlapping with the primary_mask, 4) the centroid's coodinates of the identified secondary objects and
    4) any other addition property specified in properties and extra_properties. If secondary_object_name
    is None (default) these columns are named label_secondary_object, area_secondary_object,
    area_fraction_secondary_object, centroid-{int}_secondary and {property}_secondary_object.
    secondary_object_name can be used to specify a name which will substitute the {secondary_object}
    in string in the columns of the output data frame.

    - **kwargs. Any other additional paramenter to be passed to skimage.measure.regionprops
    (https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) when measuring
    the properties of all individual, separated objects in secondary_label_image.

    Outputs: tuple.
    - position 0. Pandas DataFrame. Rows are individual identified secondary objects (1 row per object).
    The present columns are present:
        - area_overlap_{secondary_object or seconday_object_name} is the area of the secondary object which overlaps with the primary_mask.
        - label_{secondary_object or seconday_object_name} is the label value of the secondary object in secondary_label_image (ref to https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).
        - area_{secondary_object or seconday_object_name} is the area of the identified secondary objects (ref to https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).
        - centroid_{int}_{secondary_object or seconday_object_name}. The coordinates of the centroid of the identified secondary object.
        As many columns as the dimensions of the input secondary_label_image each corresponding to one coordinate.
        {int} will indicate the axis the coordinate refers to (ref to https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops).
        - area_fraction_{secondary_object or seconday_object_name} is the fraction of the area of the secondary object which
            overlaps with the primary_mask.
        - label_{primary_object or primary_object_name} is the label value of the primary_mask or, the value provided through the primary_label
        argument.
        - additional_property_{secondary_object or seconday_object_name}. As many columns as provided through the arguments
        properties and extra_properties. NOTE: the name of the column depend on the additional_property to measure.
        NOTE: the position of these columns could vary.
    
    NOTE: this is an empty dataframe if no secondary object is identified.
    
    - position 1. Numpy array. The identified secondary objects in secondary_label_image. This is secondary_label_image
    after removal of all the elements which have not been identified as secondary objects.

    """
    # if custom proporties are passed to properties
    # assure tha 'area', 'label' and 'centroid are not passed as arguments
    # and instead add them (as they are mandatory)
    if properties!=None:
        if isinstance(properties,str):
            assert properties!='area', "don't include 'area' in properties"
            assert properties!='label', "don't include 'label' in properties"
            assert properties!='centroid', "don't include 'centroid' in properties"
            properties=[properties,'area','label','centroid']
        else:
            assert 'area' not in properties, "don't include 'area' in properties"
            assert 'label' not in properties, "don't include 'label' in properties"
            assert 'centroid' not in properties, "don't include 'centroid' in properties"
            properties.append('area')
            properties.append('label')
            properties.append('centroid')
    
    # if no custom proporty is passed to properties, assure that 'area,''label', 'centroid' are still measured
    else:
        properties=['area','label','centroid']

    # measure the areas of secondary_label_image's individual, separated objects and link them to their
    # respective label value. Further measure the properties and extra_properties of interest
    properties_all_secondary_object = pd.DataFrame(regionprops_table(secondary_label_image,
                                                                         properties=properties,
                                                                         extra_properties=extra_properties,
                                                                         **kwargs))

    # set all label values in secondary_label_image to 0 if they don't overlap with primary mask.
    # NOTE: maintain the values of the secondary_label_image!
    primary_filtered_secondary_label_image = np.where(primary_mask>0,secondary_label_image,0)

    # re-measure the areas of the secondary_label_image's labelled objects after having removed every pixel
    # which does not overlap with the primary_mask.
    # NOTE: link these new area measurements to their respective label value
    new_properties_all_secondary_object = pd.DataFrame(regionprops_table(primary_filtered_secondary_label_image, properties=['area', 'label']))

    # obtain the orginal area measurements and extra measurements for the
    # secondary_label_image's objects overlapping with primary_mask.
    # NOTE: the label values are used as linking information to select only the original area measurements of the
    # objects which overlap with primary_mask.
    # Form a dataframe with a column for the original area value (called intial_area), a column with the
    # area overlapping with primary_mask and a extra columns for the extra measurements
    label_based_merged_secondary_object_properties = new_properties_all_secondary_object.merge(properties_all_secondary_object.rename(columns={'area':'initial_area'}),
                                                                                               on='label', how='left')

    # add a column with the fraction between the area of a secondary object overlapping with primary_mask
    # and its original area
    label_based_merged_secondary_object_properties['area_fraction']=label_based_merged_secondary_object_properties['area']/label_based_merged_secondary_object_properties['initial_area']

    # select rows (aka objects identified in secondary_label_image) whose area overlaps with primary_mask for
    # a fraction strictly higher than threshold_overlap_fraction
    properties_true_secondary_objects=label_based_merged_secondary_object_properties[label_based_merged_secondary_object_properties['area_fraction']>threshold_overlap_fraction]

    # get the labels of the identified secondary objects
    secondary_object_labels = properties_true_secondary_objects['label'].to_numpy()

    # only select identified secondary objects in secondary_label_image
    secondary_object_label_array = np.where(np.isin(secondary_label_image, secondary_object_labels), secondary_label_image,0)

    # instantiate a dictionary to collect the column names and map them to their new names
    new_clm_mapper = {}

    # iterate through the columnss
    for clm in label_based_merged_secondary_object_properties.columns:
        
        if clm=='area':
            if secondary_object_name!=None:
                new_clm=f"area_overlap_{secondary_object_name}"
            else:
                new_clm="area_overlap_secondary_object"

        # rename the initial_area column (corresponding to the area of the secondary_object)
        # to 'area_secondary_object' or 'area_{secondary_object_name}'
        # if secondary_object_name argument is provided
        elif clm=='initial_area':
            if secondary_object_name!=None:
                new_clm=f"area_{secondary_object_name}"
            else:
                new_clm="area_secondary_object"
            
        # rename the rest of the columns to '{column}_secondary_object' or
        # '{column}_{secondary_object_name}' if secondary_object_name argument is provided
        else:
            if secondary_object_name!=None:
                new_clm=f"{clm}_{secondary_object_name}"
            else:
                new_clm=f"{clm}_secondary_object"
            
        # map old and new column name in the mapping dictionary
        new_clm_mapper[clm]=new_clm
        
    # rename measured properties columns
    properties_secondary_objects_df = properties_true_secondary_objects.rename(mapper=new_clm_mapper, axis=1, copy=True)
    
    # get primary label if not provided
    if primary_label==None:
        primary_label=np.unique(primary_mask)
        primary_label=primary_label[primary_label!=0][0]
    
    # add a column with the primary label call the column 'label_primary_object' or 'label_{primary_object_name}'
    # if primary_object_name is provided
    if primary_object_name!=None:
        primary_label_clm_name = f"label_{primary_object_name}"
    else:
        primary_label_clm_name="label_primary_object"
    properties_secondary_objects_df[primary_label_clm_name]=[primary_label for i in range(properties_secondary_objects_df.shape[0])]

    return properties_secondary_objects_df, secondary_object_label_array


def measure_label_object_overlap(label_image_1:np.array,
                                 label_image_2:np.array,
                                 combined_dtype:np.dtype=np.uint32,
                                 include_background:bool=False,
                                 check_dtype_safety:bool=False,
                                 label_1_clm:str|None=None,
                                 label_2_clm:str|None=None,
                                 counts_clm:str|None=None)-> pd.DataFrame:
    """
    Return a data frame where each row is every possible pair of overlapping labels in label_image_1 and label_image_2,
    (with the exclusion of pairs involving label_image_1 value 0 if include_background=False). Columns are the label value in
    label_image_1, the corresponding label value in label_image_2 and the count of pixels overlapping.

    NOTE: if include_background is False,this means that all the values (aka objects) in label_image_1 are present in the data frame,
    however not all the values in label_image_2 are necessarily present. They are only present if they overlap with a label in label_image_1.
    If instead include_background is True, then all the values in label_image_1 and label_image_2 are present in the data frame.

    Inputs:
    - label_image_1. n-dimensional np.array. Labelled image. Background pixels are expected to have value 0.
    - label_image_2. n-dimensional np.array. Labelled image. Background pixels are expected to have value 0.
    - combined_dtype. np.dtype. Optional. Default np.uint32. The data type to use for the combined label values (SEE NOTE BELOW).
    - include_background. bool. Optional. Default False. If True, include background pixels in the analysis.
    - label_1_clm. str or None. Optional. Default "label_1". The name of the column where label values of
    measured objects from the label_image_1 are stored in the output dataframe.
    - label_2_clm. str or None. Optional. Default "label_2". The name of the column where label values of
    measured objects from the label_image_2 are stored in the output dataframe.
    - counts_clm. str or None. Optional. Default "counts". The number of overlapping pixels between a label_1 - label_2
    object pair.

    NOTE: THIS METHOD RELIES ON THE MULTIPLICATION OF THE HIGHEST VALUE IN label_image_1 AND THE HIGHEST VALUE IN
    label_image_2. THIS MULTIPLICATION CAN LEAD TO NUMBERS NOT HANDLED BY THE INITIAL DATA TYPE OF THE IMAGES THUS
    LEADING TO VALUE OVERFLOW (ULTIMATELY INFORMATION LOSS). TO AVOID THIS ENSURE THAT DATA ARE CONVERTED TO
    AN APPROPRIATE DATA TYPE BEFORE THE MULTIPLICATION, BY SETTING THE VARIABLE combined_dtype. BY DEFAULT
    combined_dtype IS np.uint32. SUCH DATA TYPE CAN HANDLE SITUATIONS WHERE BOTH label_image_1 AND label_image_2 HAVE
    MAXIMUM 65536 LABELLED OBJECTS. IF MORE OBJECTS ARE PRESENT SET combined_dtype to np.uint64 (THIS WILL BE
    COMPUTATIONALLY INTENSE) WHICH CAN HANDLE APPROXIMATELY 4.3 BILLION OBJECTS. IF YOUR MASKS HAVE
    LOWER THAN 256 OBJECTS YOU CAN SET combined_dtype TO np.uint16.
    YOU CAN CHECK APPROXIMATELY THE DATA TYPE NEEDED AS FOLLOW:
    1) CALCULATE (label_image_1.max() * (label_image_2.max() +1)) + label_image_2.max() . THIS IS THE HIGHEST VALUE
    WHICH THE DATA TYPE INDICATED IN combined_dtype SHOULD BE ABLE TO CONTAIN.
    2) CALCULATE THE HIGHEST VALUE WHICH CAN BE CONTAINED IN A NUMPY DATA TYPE USING np.iinfo() ON THE DATA TYPE (e.g.
    np.iinfo(np.uint32)).
    THE VALUE OF POINT 1 MUST BE STRICTLY LOWER THAN THE VALUE IN POINT 2. IF NOT, YOU SHOULD USE DATA TYPE WHICH
    ALLOWS HIGHER VALUES.
    """
    # set default names for output dataframe columns
    if label_1_clm is None:
        label_1_clm="label_1"
    
    if label_2_clm is None:
        label_2_clm="label_2"
    
    if counts_clm is None:
        counts_clm="counts"

    if check_dtype_safety:
        max_combined = (label_image_1.max() * (label_image_2.max() + 1)) + label_image_2.max()
        dtype_limit = np.iinfo(combined_dtype).max
        if max_combined >= dtype_limit:
            raise ValueError(f"combined_dtype {combined_dtype} too small. Max required: {max_combined}, dtype can hold: {dtype_limit}")
    
    # Flatten both images
    labels1 = label_image_1.ravel().astype(combined_dtype)
    labels2 = label_image_2.ravel().astype(combined_dtype)

    # Mask out background (if desired)
    if not include_background:
        mask = labels1 > 0  # Optional: remove background from label_image_1
        labels1 = labels1[mask].astype(combined_dtype)
        labels2 = labels2[mask].astype(combined_dtype)

    # Combine pairs of label values into unique indices
    max_label2 = labels2.max() + 1
    combined = labels1 * max_label2 + labels2

    # Count unique combinations - personal note: the risk of overflow (aka information loss due the fact that unique pairs and/or counts may exceed dtype limits)
    # has been assessed and is not expected to happen in practice, as long as the combined_dtype is set appropriately or the number of overlapping pixels
    # is in the order of gigapixels.
    unique_pairs, unique_pairs_counts = np.unique(combined, return_counts=True)

    # Split back into label1 and label2
    label1_ids = unique_pairs // max_label2
    label2_ids = unique_pairs % max_label2
    
    # Create a data frame
    labels_overlap_count_dict = {label_1_clm:label1_ids,
                                label_2_clm: label2_ids,
                                counts_clm: unique_pairs_counts}
    labels_overlap_counts_df = pd.DataFrame.from_dict(labels_overlap_count_dict)

    return labels_overlap_counts_df


def measure_label_stack_overlap(label_image_stack,
                                axis:int=0,
                                combined_dtype:np.dtype=np.uint32,
                                include_background:bool=True,
                                check_dtype_safety:bool=False,
                                label_clm:str|None=None,
                                overlap_clm:str|None=None,
                                sep:str|None=None)-> pd.DataFrame:
    """
    Measure and collect the overlap between all the labels of all pairs of label images in a stack.
    The label_image_stack is expected to be a 2D or higher dimensional array where each sub-array along the specified axis
    is a labelled image. The function measures the overlap between all pairs of sub-arrays along the specified axis
    and returns a dataframe with the results.
    
    Inputs:
    - label_image_stack. n-dimensional np.array. Must have at least 2 dimensions. A stack of labelled images. Each sub-array along the specified axis is a labelled image.
    - axis. int. Optional. Default 0. The axis along which to split the label_image_stack into sub-arrays.
    - combined_dtype. np.dtype. Optional. Default np.uint32. The data type to use for the combined label values. NOTE: this is used to avoid overflow when combining label values from different label images.
    It is recommended to set this to np.uint32 for most cases, but it can be set to np.uint64 for very large label images or np.uint16 for smaller label images.
    NOTE: if the combined_dtype is too small, it can lead to overflow and loss of information. See the function's docstring (measure_label_object_overlap) for more details.

    - include_background. bool. Optional. Default True. If True, include background pixels in the analysis.

    - check_dtype_safety. bool. Optional. Default False. If True, check that the combined_dtype is safe for the maximum label values in the label_image_stack.

    - label_clm. :str or None. Optional. Default "label". The prefix of names of the columns storing object label
    values per each label image of label_image_stack.

    - overlap_clm. str or None. Optional. Default "areaoverlap". The prefix of names of the columns storing the
    overlap between pixels per each pair of labelled objects in the label_image_stack.

    - sep. str or None. Optional. Default "sep". The separator for tokens in the output dictionary column names.

    Outputs: pandas DataFrame. Each row is a pair of labels from the label images in the label_image_stack (the sub-arrays
    resulting by splitting label_image_stack on the specified axis). All and only existing pairs of labels are present in the output dataframe.

    The columns are:
    
    - {label_clm}{sep}0, {label_clm}{sep}1, ...: the label values from the label images in the label_image_stack. The final number indicates the index of the label image
    in the label_image_stack along the indicated axis.

    - {overlap_clm}{sep}0{sep}1, {overlap_clm}{sep}0{sep}2, ...: the area of overlap between the labels from the label images in the label_image_stack. The final numbers
    indicate the indices of the label image pair in the label_image_stack along the indicated axis.
    For example, area_overlap_0_1 is the area of overlap between label_0 and label_1.
    
    - area{sep}0, area{sep}1, ...: the area of the labels from the label images in the label_image_stack. The final number indicates the index of the label image
    in the label_image_stack along the indicated axis.
    
    - centroid-y{sep}0, centroid-x{sep}0, ...: the coordinates of the centroids of the labels from the label images in the label_image_stack. The
    second last number indicates the axis along which the centroid is measured, the last number indicates the index of the label image in the
    label_image_stack along the indicated axis.
    (e.g. centroid-y_0 is the y-coordinate of the centroid of label_0, centroid-x_1 is the x-coordinate of the centroid of label_1).

    The function can be used for selecting labeled regions of different sub-arrays based on their overlap.
    Here is an example for selecting tertiary objects (from label_image_3) that overlap with secondary objects (from label_image_2)
    which themselves overlap with a specific primary object (target_label) in label_image_1:
    
    1) select only the rows containing the target_label of the primary object: df_1 = output_df[output_df['label_0']==target_label]
    
    2) calculate what fraction of each secondary object's area overlaps with the primary object (not vice versa):
    df_1['overlap_fraction_1_0'] = df_1['area_overlap_0_1'] / df_1['area_1']
    
    3) further select the rows of the secondary objects overlapping more than 70% with the primary object:
    df_2 = df_1[df_1['overlap_fraction_1_0']>0.7]

    4) get the label values of the secondary objects:
    secondary_object_labels = df_2['label_1'].to_numpy()

    5) select only the rows containing the secondary objects:
    df_3 = output_df[output_df['label_1'].isin(secondary_object_labels)]

    6) calculate what fraction of each tertiary object's area overlaps with the secondary object (not vice versa):
    df_3['overlap_fraction_2_1'] = df_3['area_overlap_1_2'] / df_3['area_2']

    7) further select the rows of the tertiary objects overlapping more than 70% with the secondary object:
    df_4 = df_3[df_3['overlap_fraction_2_1']>0.7]
    NOTE that the tertiary object might overlap with the primary object less than 70%.
    
    6 - optional) To strictly restrict to only those tertiary objects that overlap with both the secondary and the primary object,
    you can further calculate the fraction of the area of the tertiary objects which overlap with the primary object, and then filter:
    tertiary_object_labels = df_4['label_2'].to_numpy()
    df_5 = output_df[output_df['label_2'].isin(tertiary_object_labels)]
    df_5['overlap_fraction_2_0'] = df_5['area_overlap_0_2'] / df_5['area_2']
    df_6 = df_5[df_5['overlap_fraction_2_0']>0.7]
    """

    # set default variables for column naming
    if label_clm is None:
        label_clm="label"
    
    if overlap_clm is None:
        overlap_clm="areaoverlap"
    
    if sep is None:
        sep="_"

    # check that label_image_stack is at least 2D and that the axis is valid
    assert label_image_stack.ndim > 1, "label_image_stack must be at least 2D"
    assert axis < label_image_stack.ndim, f"axis {axis} is out of bounds for label_image_stack with {label_image_stack.ndim} dimensions"
    assert axis >= 0, f"axis {axis} must be non-negative"
    assert label_image_stack.shape[axis] > 1, f"label_image_stack must have more than one label image along axis {axis}"

    # split the label_image_stack along the specified axis
    label_image_list = [np.squeeze(a, axis=axis) for a in np.split(label_image_stack, indices_or_sections=label_image_stack.shape[axis], axis=axis)]

    # initialize a list to collect the overlap dataframes
    overlap_df_collection = []

    # collect the expected columns in the output dataframe
    expected_columns = [f"{label_clm}{sep}{i}" for i in range(len(label_image_list))]
    for i in range(len(label_image_list)):
        for j in range(i+1, len(label_image_list)):
            expected_column = f"{overlap_clm}{sep}{i}{sep}{j}"
            expected_columns.append(expected_column)

    # iterate through the pairs of label images and measure overlaps
    for label_image_index_1, label_image_1 in enumerate(label_image_list):
        label_image_index_2 = label_image_index_1+1
        for label_image_2 in label_image_list[label_image_index_2:]:

            # measure the overlap between the two label images - NOTE: the default naming of the output columns is used
            overlap_df_i = measure_label_object_overlap(label_image_1=label_image_1,
                                                            label_image_2=label_image_2,
                                                            combined_dtype=combined_dtype,
                                                            include_background=include_background,
                                                            check_dtype_safety=check_dtype_safety)
                
            # rename the columns of the overlap dataframe to match the label image indices
            # NOTE: the remapping is based on the default column naming from measure_label_object_overlap
            overlap_df = overlap_df_i.rename(columns={'label_1': f"{label_clm}{sep}{label_image_index_1}",
                                                          'label_2': f"{label_clm}{sep}{label_image_index_2}",
                                                          'counts': f"{overlap_clm}{sep}{label_image_index_1}{sep}{label_image_index_2}"})
            
            # add the label columns for the other label images
            for clumn in expected_columns:
                if clumn not in overlap_df.columns:
                    overlap_df[clumn] = np.nan
                
            # append the overlap dataframe to the collection
            overlap_df_collection.append(overlap_df)

        # increment label_image_index_2 to avoid re-analysing the same pair of label images
        label_image_index_2 = label_image_index_2 + 1

    # Concatenate all the pairwise dataframes
    output_overlap_df = pd.concat(overlap_df_collection, ignore_index=True)

    # iterate through the label images and measure the area and centroid of each label
    # NOTE: properties are fixed in this case, it is not possible to choose them ad hoc
    for label_image_index, label_image in enumerate(label_image_list):

        label_image_area_coord_df_i = pd.DataFrame(regionprops_table(label_image, properties=['area', 'label', 'centroid']))
        
        # rename the columns of the area dataframe to match the label image index
        column_mapper = {}
        for clm in label_image_area_coord_df_i.columns:
            new_clm = f"{clm}{sep}{label_image_index}"
            column_mapper[clm] = new_clm

        label_image_area_coord_df = label_image_area_coord_df_i.rename(columns=column_mapper)

        # merge the area and centroid information with the output dataframe
        output_overlap_df_i = pd.merge(output_overlap_df,
                                        label_image_area_coord_df,
                                        how='left', on=f"{label_clm}{sep}{label_image_index}", copy=True)
        
        # update the output dataframe with the new area and centroid information
        output_overlap_df = output_overlap_df_i

    # reorder the columns of the output dataframe
    # to have label columns first, then overlap columns, then area columns and finally centroid columns
    # this is done to make the output dataframe more readable and easier to interpret
    
    # collect the column names in the desired order
    label_clm_collection = []
    overlap_clm_collection = []
    area_clm_collection = []
    centroid_clm_collection = []

    # iterate through the columns of the output dataframe
    # and collect the column names in the desired order
    for clm in output_overlap_df.columns:
        if f"{label_clm}{sep}" in clm:
            label_clm_collection.append(clm)
        
        elif "label" in clm:
            label_clm_collection.append(clm)
        
        elif overlap_clm in clm:
            overlap_clm_collection.append(clm)
        
        elif "area" in clm :
            area_clm_collection.append(clm)
        
        else:
            centroid_clm_collection.append(clm)
    
    # create the new column order
    # first label columns, then overlap columns, then area columns and finally centroid columns
    new_clm_order = label_clm_collection + overlap_clm_collection + area_clm_collection + centroid_clm_collection

    # reorder columns
    output_overlap_df = output_overlap_df[new_clm_order]

    return output_overlap_df


def extract_value_per_label(intensity_image:np.array,
                            label_image:np.array)-> dict:
    """
    Extract the values from an image based on labelled objects.

    This function takes an image (intensity_image) and a label image, flattens them, and returns a dictionary
    where each key is a label from the label image and each value is an array with the values of the pixels in the intensity_image
    corresponding to that label.

    NOTE: all labels in label image are present in the output dictionary, inlcuding 0, irrespective of the
    the possibility that a value corresponds to the background.

    Inputs:
    - intensity_image. n-dimensional np.array. The image from which to extract values.
    - label_image. n-dimensional np.array. The label image where each pixel's value corresponds
    to a label of an object in the intensity_image.

    Outputs: dict. A dictionary where each key is a label from the label_image and each value is an array with the values of
    the pixels in the intensity_image corresponding to that label.
    """

    # flatten the intensity_image and label_image
    # this is done to make it easier to group the values by label
    flat_intensity = intensity_image.ravel()
    flat_labels = label_image.ravel()

    # Create a DataFrame to hold the values and labels
    # This allows us to group the values by label easily
    df = pd.DataFrame({'value': flat_intensity, 'label': flat_labels})

    # Group values by label and apply a function to convert them to numpy arrays, finally convert to a dictionary
    value_per_label_dict = df.groupby('label')['value'].apply(np.array).to_dict()

    return value_per_label_dict


def permute_values_per_label(intensity_image_stack:np.array,
                             label_image:np.array,
                             axis:int=0,
                             replace:bool=False,
                             **kwargs)-> pd.DataFrame:
    """
    Permute the values in a stack of images based on labels in a label image. Values are independently permuted per each image of the
    stack. Values are permuted within each labelled object. The function preserves the link between the values and the labelled object.

    Given a stack of images (intensity_image_stack) and objects in the image identified by labels in label_image (NOTE: the same object is
    applied to all the images), the function, per each labelled object, and independently per each image of the stack, takes the pixel values
    associated to the labelled object which are in the image, and permutes their position. The link between the pixel value and the label value is preserved.

    The function can be used to generate object-specific, channel-independed pixel value permutations, to be used as control measurements.
    
    Example: one has a stack of two images. The images are the same field of view, but each containing a staining of a different protein. Individual cells are identified
    in the field of view by a segmentation mask transformed in label image, so that each cell is associated to a unique label value.
    The function allows to generate random purmutations of the pixels of each individual cell, and the permutation (aka, random repositining of the pixels)
    is done within the individual cell. In addition, the permutation is done independently for the two stainings. The output is a
    dataframe where each label corresponding to individual cells is associated to the origianl values, but in a random order.
    It can be useful, for example, if one is interested in the signal correlation of between the two staining, at cell-level (per each individual cell, thus not
    for the entire field of view) and wants to understand what would be the correlation coefficient obtained by chance.

    NOTE: despite the fact that the link between the pixel values and the labels is preserved, the position of the pixels in the image is not preserved.

    NOTE: all labels in label image are threated equally, including 0, irrespective of the
    the possibility that a value corresponds to the background.

    Inputs:
    - intensity_image_stack. n-dimensional array with a minimum of 2 dimensions. The stack of images whose objects' pixels should be permuted.
    - label_image. n-dimensional array. The shape of label_image must match the shape of the sub-arrays obtained by splitting intensity_image_stack
    on axis.
    - axis. int. Optional. Default 0. The axis to use for splitting intensity_image_stack in sub-arrays.
    - replace. bool. Optional. Default False. If The same value can be repeated multiple times when generating permutations (random sampling with
    replacement). The parameter is passed to the argument "replace" in pandas.DataFrame.sample (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html).
    - **kwargs. Any other additional parameter to be passed to pandas.DataFrame.sample (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html),
    with the exception of "replace" which is passed as a separate argument.

    Outputs: pandas DataFrame.
    The output dataframe has a row per each pixel of the images to compare. If replace is False (default), each label and pixel value of the input images is
    present and their count is preserved. If replace is True, the pixel values are randomly sampled with replacement, thus the count of each label and of each
    pixel value is not preserved.
    The output dataframe has a column for each image in the intensity_image_stack, and a column for the label values.
    The columns are named "value_0", "value_1", ..., where the number indicates the index of the image in the intensity_image_stack along the specified axis.
    The column "label" contains the label values from the label_image. The values in each column are the permuted pixel values for each labelled object.

    Strategy:
    1) Split the intensity_image_stack along the specified axis.
    2) Flatten the intensity values of the first image and the label image. At this stage, it would be possible to recover the original pixel positions.
    3) Randomly permute the order of the intensity values while keeping links to their labels.
    """

    assert replace not in kwargs, "replace can't be passed to kwargs. Use the replace argument instead."
    assert len(intensity_image_stack.shape)>1, "intensity_image_stack must have more than 2 dimensions."

    # split the intensity_image_stack along the specified axis
    intensity_image_list = [np.squeeze(a, axis=axis) for a in np.split(intensity_image_stack, indices_or_sections=intensity_image_stack.shape[axis], axis=axis)]
    
    # flatten the intensity sub-arrays and the label image
    flat_intensity_0 = intensity_image_list[0].ravel()
    flat_labels = label_image.ravel()

    # Link the flattened intensity values of the first intensity image to their labels in different dataframes
    df_0 = pd.DataFrame({'label': flat_labels, 'value_0': flat_intensity_0})
    
    # permute the intensity values of the first intensity image while keeping links to their labels
    permutation_df_0 = df_0.sample(frac=1, replace=replace, **kwargs).reset_index(drop=True)
    
    # sort the dataframe by label to ensure that the labels are aligned
    permutation_df_0.sort_values(by='label', inplace=True)
    
    # reset the index of the dataframe
    permutation_df_0.reset_index(drop=True, inplace=True)

    # iterate through the remaining intensity images and permute their values
    # while keeping links to their labels
    for c, intensity_image in enumerate(intensity_image_list[1:]):
        # flatten the intensity values of the current intensity image
        flat_intensity = intensity_image.ravel()

        # permute the intensity values while keeping links to their labels
        df_i = pd.DataFrame({'label': flat_labels, f'value_{c+1}': flat_intensity})

        permutation_df_i = df_i.sample(frac=1, replace=replace,**kwargs).reset_index(drop=True)
        
        # sort the dataframes by label to ensure that the labels are aligned
        permutation_df_i.sort_values(by='label', inplace=True)
        
        # reset the index of the dataframes
        permutation_df_i.reset_index(drop=True, inplace=True)

        # add the permuted values from df_1 to df_0
        # this ensures that the values are aligned by label
        permutation_df_0[f'value_{c+1}'] = permutation_df_i[f'value_{c+1}']

    return permutation_df_0


def match_labels(container_labels, contained_labels, strict=True, verbose=True):
    """
    Relabels a 'contained' label image so that each object inside it 
    inherits the label of the 'container' object it lies within.

    Includes validation for:
      - identical array shape,
      - full containment of 'contained' objects,
      - one-to-one mapping between container and contained objects.

    Parameters
    ----------
    container_labels : ndarray of int
        Labeled image of container regions (e.g., cells, tissues, etc.).
    contained_labels : ndarray of int
        Labeled image of contained regions (e.g., nuclei, organelles, etc.).
    strict : bool, optional
        If True (default), raises a ValueError when containment rules are violated.
        If False, continues but prints warnings and attempts best effort.
    verbose : bool, optional
        If True (default), prints information about detected violations.

    Returns
    -------
    contained_labels_matched : ndarray of int
        New label image where each contained object has the same label 
        as its corresponding container object.
    mapping : dict
        Dictionary mapping {old_contained_label: new_container_label}.

    Raises
    ------
    AssertionError
        If container_labels and contained_labels have different shapes.
    ValueError
        If containment or one-to-one mapping assumptions are violated 
        and `strict=True`.
    """

    # --- Sanity check: arrays must have the same shape ---
    # Ensures that both labeled images correspond pixel-by-pixel (or voxel-by-voxel).
    assert container_labels.shape == contained_labels.shape, (
        f"Shape mismatch: container_labels has shape {container_labels.shape}, "
        f"but contained_labels has shape {contained_labels.shape}."
    )

    # Initialize a new label image that will hold the remapped contained labels.
    # It starts filled with zeros (background).
    contained_labels_matched = np.zeros_like(contained_labels, dtype=container_labels.dtype)

    # Dictionary mapping old contained labels → new container labels.
    mapping = {}

    # Reverse dictionary to detect if multiple contained regions map to the same container.
    reverse_mapping = {}

    # Iterate over each labeled object in the 'contained' image.
    for region in regionprops(contained_labels):
        contained_label = region.label  # The integer label of this contained region.
        coords = region.coords          # Array of pixel/voxel coordinates belonging to this region.

        # Determine which container labels these coordinates overlap with.
        overlapping_container_labels = np.unique(container_labels[tuple(coords.T)])

        # Remove any zeros (background) from the list of overlapping container labels.
        overlapping_container_labels = overlapping_container_labels[overlapping_container_labels != 0]

        # --- Validation #1: Each contained object must overlap exactly one container ---
        if len(overlapping_container_labels) == 0:
            # This contained region is floating in background — not inside any container.
            msg = f"Contained label {contained_label} is not inside any container."
            if strict:
                raise ValueError(msg)  # Stop execution if strict mode is on.
            if verbose:
                print(f"WARNING: {msg}")  # Otherwise, print a warning and skip this region.
            continue

        if len(overlapping_container_labels) > 1:
            # This contained region overlaps more than one container.
            msg = (f"Contained label {contained_label} overlaps multiple containers: "
                   f"{list(overlapping_container_labels)}")
            if strict:
                raise ValueError(msg)
            if verbose:
                print(f"WARNING: {msg}")

            # Non-strict mode: choose the container with the largest area of overlap.
            counts = [(c, np.sum(container_labels[tuple(coords.T)] == c))
                      for c in overlapping_container_labels]
            container_label = max(counts, key=lambda x: x[1])[0]  # Container with max overlap.
        else:
            # Exactly one overlapping container — ideal case.
            container_label = overlapping_container_labels[0]

        # --- Validation #2: One-to-one mapping between contained and container ---
        # Ensure each container has at most one contained region assigned.
        if container_label in reverse_mapping:
            msg = (f"Multiple contained labels map to container {container_label}: "
                   f"{reverse_mapping[container_label]} and {contained_label}")
            if strict:
                raise ValueError(msg)
            if verbose:
                print(f"WARNING: {msg}")
                
            continue  # Skip this region to avoid overwriting.

        # Record the mapping (for reporting or reproducibility).
        mapping[contained_label] = container_label
        reverse_mapping[container_label] = contained_label

        # Update the matched label image:
        # All pixels of this contained region are relabeled to match its container.
        contained_labels_matched[contained_labels == contained_label] = container_label

    # Return the remapped label image and the mapping dictionary.
    return contained_labels_matched, mapping
