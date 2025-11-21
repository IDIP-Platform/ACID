import os
import numpy as np
from skimage import io
import imageio
from bioio import BioImage
# from readlif.reader import LifFile
# from readlif.utilities import get_xml
# import nd2

def ioopen_image(image_path:os.PathLike) -> np.array:
    """
    Opens an image using skimage.io. Compatible with many different formats, ma no matadata are read.
    
    Input:
    - image_path. PathLike. The full path to the image.

    Output:
    - numpy.array.
    """
    return io.imread(image_path)


def imageio_open_image(image_path:os.PathLike) -> np.array:
    """
    Opens an image using imageio. Compatible with many different formats. TO DO: this function can be tailoried to also read metadata.
    
    Input:
    - image_path. PathLike. The full path to the image.

    Output:
    - numpy.array.
    """
    return imageio.imread(image_path)

def bioio_open_image(image_path:os.PathLike,
                     return_metadata:bool=False,
                     **kwargs) -> np.array:
    """
    Opens an image using imageio. Compatible with many different formats. TO DO: this function can be tailoried to also read metadata.
    
    Input:
    - image_path. PathLike. The full path to the image.
    - return_metadata. bool. Optional. Defaul False. Whether or not to return the original metadata.
    - kwargs. Additional parameters to pass to bioio.BioImage.

    Output:
    If return_metadata==False a bioio.BioImage object is returned (https://bioio-devs.github.io/bioio/bioio.html#bioio.bio_image.BioImage).

    If return_metadata==True a tuple is returned:
    - Position 0. bioio.BioImage object.
    - Position 1. The result of calling bioio.BioImage.metadata on the BioImage object.

    TEST FILE TYPES
    - .lif objects from Leica SP8 TCS DLS Confocal and SPIM microscope.
    """
    
    # read file - import file as a bioimage object
    bioimage_file = BioImage(image_path, **kwargs)

    if return_metadata:
        # extract file orginal metadata
        file_original_metadata = bioimage_file.metadata
        
        return bioimage_file, file_original_metadata

    else:
        return bioimage_file



