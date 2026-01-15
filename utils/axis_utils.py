import os
import pandas as pd
import tifffile

def get_ch_number_shape(df:pd.DataFrame,
                        fov_dir:os.PathLike,
                        fov_clm:str='ome_tif_file_name',
                        channel_axis: int=0,
                        null_value:float|None=None)->int:
    """
    Get the number of channels and the shape of individual channels in the images indicated in the dataframe.
    
    Assumes that all images in the dataframe have the same number of channels and that all channels have the
    same shape.

     Parameters:
        df: pandas DataFrame containing the metadata information, including the file names of the fields of view.
        fov_dir: path to the directory containing the fields of view.
        fov_clm: name of the column in df containing the file names of the fields of view.
        channel_axis: int, axis corresponding to the channels in the image array.
        null_value: value to return if no valid image files are found in the dataframe.
    Returns:
        ch_number: int, number of channels in the images.
     """

    # initiate a variable to signal whether a file has been found
    file_found = False

    # initiate index variable
    i = 0

    # loop until a file is found
    while file_found is False:

        # check that index is within dataframe bounds
        if i < df.shape[0]:
            # attempt to read the image file
            try:

                # read the field of view
                fov = tifffile.imread(os.path.join(fov_dir, str(df.iloc[i, :][fov_clm])))

                # signal that a file has been found
                file_found = True

            except:
                # signal that a file hasn't been found
                file_found = False

            
            if file_found:
                # get the number of channels
                ch_number = fov.shape[channel_axis]

                # print the number of channels found
                print(f"number of channels found: {ch_number}")

                # get the channels' shape
                ch_shape = tuple([fov.shape[i] for i in range(len(fov.shape)) if i not channel_axis])

                # print the channels's shape
                print((f"shape of channels: {ch_shape}"))
            
            # except block to handle cases where the file cannot be read
            else:
                pass
        
        # if the end of the dataframe is reached
        else:

            # print a message indicating no valid image files were found
            print("No valid image files found in the dataframe.")

            # assign ch_number and ch_shape to null values
            ch_number = null_value
            ch_shape = null_value

        # increment index
        i += 1
    
    return ch_number, ch_shape

