from collections.abc import Callable
import numpy as np
import pandas as pd
from bioio import BioImage

def extract_bioio_scene_metadata(bioio_scene:Callable,
                                 dtype_name:str|bool|None=None,
                                 dims_order_name:str|bool|None=None,
                                 t_name:str|bool|None=None,
                                 c_name:str|bool|None=None,
                                 z_name:str|bool|None=None,
                                 xy_prefix_name:str|bool|None=None,
                                 physical_size_prefix_name:str|bool|None=None,
                                 **kwargs)->tuple:
    
    """
    Extract metadata from a bioio.BioImage scene object, organizes them in a dictionary and a pandas.Series.

    By default, the following metadata are extracted (and expected):
    - data type.
    
    - order of dimensions.
    
    - size of time, channel, z-axis, y-axis and x-axis in, respectively, number of timepoints, number of channels
    number of z-planes, pixels and pixels.
    
    - physical size of:
        - time interval, if more than a timepoint is present (not implemented as of 2025/06/30).
        - z-step, if more and a plane is present.
        - y-pixel dimension.
        - x-pixel dimension.

    It is possible to exclude individual metadata by passing False to the corresponding metadata name in the
    function arguments.
        
    Any additional parameter which one wants to record in the metadata can be passed to kwargs.
        
    NOTE: saving of the unit of the t,z,y,x physical sizes is not implemented (as of 2025/06/30). It is
    RECOMMENDED TO PASS THIS INFORMATION TO kwargs. 
    
    Outputs: tuple.
    
    - Position 1. Extracted metadata and all metadata passed to kwargs as a pandas.Series.
    - Position 2. Extracted metadata and all metadata passed to kwargs as a dictionary.
    """
    
    # set default metadata names
    if dtype_name is None:
        dtype_name='dtype'
    
    if dims_order_name is None:
        dims_order_name='dims_order'
    
    if t_name is None:
        t_name='size_t'
    
    if c_name is None:
        c_name='size_c'
    
    if z_name is None:
        z_name='size_z'
    
    if xy_prefix_name is None:
        xy_prefix_name='size'
    
    if physical_size_prefix_name is None:
        physical_size_prefix_name='physical_size'

    # assert correct data type for naming arguments
    assert (isinstance(dtype_name,str) or dtype_name==False), "dtype_name can only be a string or False"
    assert (isinstance(dims_order_name,str) or dims_order_name==False), "dims_order_name can only be a string or False"
    assert (isinstance(t_name,str) or t_name==False), "t_name can only be a string or False"
    assert (isinstance(c_name,str) or c_name==False), "c_name can only be a string or False"
    assert (isinstance(z_name,str) or z_name==False), "z_name can only be a string or False"
    assert (isinstance(xy_prefix_name,str) or xy_prefix_name==False), "xy_root_name can only be a string or False"
    assert (isinstance(physical_size_prefix_name,str) or physical_size_prefix_name==False), "physical_size_suffix_name can only be a string or False"

    # intialize a metadata collection dictionary
    scene_metadata={}

    # add kwargs
    for k in kwargs:
        scene_metadata[k]=kwargs[k]

    # add scene dtype to metadata dictionary if dtype_name is different than False
    if dtype_name!=False:
        scene_metadata[dtype_name]=bioio_scene.dtype
    
    # get dimesions
    input_scene_dims = bioio_scene.dims

    # if dims_order_name is different than False, initialize a list to collect dimensions order
    # the list will be used to create a string with the dimension order
    if dims_order_name!=False:
        dims_order = []
    
    # iterate through the dimensions and get their size
    for dim_name, dim_size in input_scene_dims.items():
        
        # collect the dimension name if the size is bigger than 1 (aka if the dimension exists)
        # and if if dims_order_name is different than False
        # the dimension name is collected in the list which will then be used to create a string with the
        # dimensions order
        if dim_size>1:
            if dims_order_name!=False:
                dims_order.append(dim_name)
        
        # add the number of timepoints to the metadata dictionary and if t_name is different than False
        if dim_name=="T":
            if t_name!=False:
                scene_metadata[t_name]=dim_size
            
                # to do - add time interval if the size is >1

        # add the number of imaged channels to the metadata dictionary and if c_name is different than False
        elif dim_name=="C":
            if c_name!=False:
                scene_metadata[c_name]=dim_size
        
        # add the number of planes to the metadata dictionary and if z_name is different than False
        elif dim_name=="Z":
            if z_name!=False:
                scene_metadata[z_name]=dim_size

                # only extract the physical size if there is more than 1 plane 
                if dim_size>1:
                    # get dims physical size
                    input_scene_dim_physical_size = bioio_scene.physical_pixel_sizes.Z

                    scene_metadata[f"{physical_size_prefix_name}_{dim_name.lower()}"]=input_scene_dim_physical_size

        # add the number of xy pixels, the number of z-planes and the pixel physical size to metadata dictionary
        # if xy_root_name is different than False
        else:
            if xy_prefix_name!=False:
                scene_metadata[f"{xy_prefix_name}_{dim_name.lower()}"]=dim_size
                    
                if dim_name=="Y":

                    # get dims physical size
                    input_scene_dim_physical_size = bioio_scene.physical_pixel_sizes.Y

                    scene_metadata[f"{physical_size_prefix_name}_{dim_name.lower()}"]=input_scene_dim_physical_size
                    
                else:
                    # get dims physical size
                    input_scene_dim_physical_size = bioio_scene.physical_pixel_sizes.X

                    scene_metadata[f"{physical_size_prefix_name}_{dim_name.lower()}"]=input_scene_dim_physical_size
    
    # get dims order if if dims_order_name is different than False
    if dims_order_name!=False:
        input_scene_dims_order = ''.join(dims_order)

        # add dims order to metadata dictionary
        scene_metadata[dims_order_name]=input_scene_dims_order

    # create a pandas series
    scene_metadata_series = pd.Series(scene_metadata)

    return scene_metadata_series, scene_metadata


def extract_physical_size_unit_from_lif_xml():
    """
    this is just a rough idea of how to extract the physical size unit from an XML file 
    """
    # image_0 = input_metadata.findall(".//Image")[0]
    # dimension_desc_4 = image_0.findall(".//DimensionDescription")[0]
    # length_value = dimension_desc_4.attrib.get("Length")
    # print(length_value)
    return "um"  # this is just a placeholder, the actual implementation would extract the unit from the XML file
