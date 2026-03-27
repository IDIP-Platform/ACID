import os
import numpy as np
# from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from tifffile import imwrite as tifffileimwrite


# def aicsimageio_save_ometiff(data,
#                              save_path:os.PathLike,
#                              **kwargs):
#     """
#     NOTE: with aicsimageio version 3.2.3 this function does not work. However other aicsimageio versions also
#     did not work. The function thus remains to be debugged.

#     Forms an ome.tiff file using aicsimageio.writers.ome_tiff_writer.OmeTiffWriter implementation
#     (https://allencellmodeling.github.io/aicsimageio/aicsimageio.writers.html#aicsimageio.writers.ome_tiff_writer.OmeTiffWriter.build_ome)
    
#     Inputs:
#     - data. array-like or list of array like. The data to be converted to ome.tif. Data is passed to the
#     argument data in aicsimageio.writers.ome_tiff_writer.OmeTiffWriter ref to their documentation.
#     - save_path. PathLike. The full path of the file to be saved.
#     - kwargs. The optional parameters to pass to aicsimageio.writers.ome_tiff_writer.OmeTiffWriter.save

#     Output:
#     no output is provided, but an ome.tiff object is saved at the save_path.
#     """
    
#     OmeTiffWriter.save(data,
#                         save_path,
#                         kwargs)


def tifffile_save_ometiff(save_path:os.PathLike,
                          data,
                          imagej:bool=True,
                          photometric:str|None=None,
                          **kwargs):
    """
    Forms an ome.tiff file using tifffile.imwrite (https://pypi.org/project/tifffile/) implementation.

    Inputs:
    - save_path. PathLike. The full path of the file to be saved.
    - data. array-like. The data to be converted to ome.tif. Data is passed to the
    argument data in tifffile.imwrite ref to their documentation.
    - imagej. Bool. Optional. Default True. Whether or not to create a imagej compatible file. NOTE: creating an imagej compatiple file
    implies conforming to their conventions. Ref to imagej documentation and tifffile documentation.
    - photometrics. string or None. Optional. Default 'minisblack'. How to intepret the value range of the image. The parameter is passed to
    photometrics in tifffile. Ref to their documentation.
    - kwargs. The optional parameters to pass to tifffile.imwrite. These are meant especially for metadata writing.

    Output:
    no output is provided, but an ome.tiff object is saved at the save_path.
    
    """
    if photometric is None:
        photometric="minisblack"
    
    tifffileimwrite(save_path,
                     data,
                     imagej=imagej,
                     photometric=photometric,
                     **kwargs)
    


