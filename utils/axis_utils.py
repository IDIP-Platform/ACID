import os
import pandas as pd
import tifffile

def get_ch_number(df:pd.DataFrame,
                  fov_dir:os.PathLike,
                  fov_clm:str='ome_tif_file_name',
                  channel_axis: int=0,
                  null_value:float|None=None)->int:
    """
    Get the number of channels in the images indicated in the dataframe.
    
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

                # get the number of channels
                ch_number = fov.shape[channel_axis]

                # print the number of channels found
                print(f"number of channels found: {ch_number}")

                # signal that a file has been found
                file_found = True
            
            # except block to handle cases where the file cannot be read
            except:
                pass
        
        # return null value if no valid image files are found and print a message
        else:

            # print a message indicating no valid image files were found
            print("No valid image files found in the dataframe.")

            # return null value
            return null_value

        # increment index
        i += 1
    
    return ch_number