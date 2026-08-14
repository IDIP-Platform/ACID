import os
import pandas as pd
import tifffile

def get_fov_ch_shape(df:pd.DataFrame,
                     fov_dir:os.PathLike,
                     fov_clm:str|None=None,
                     channel_axis: int=0,
                     null_value:float|None=None)->int:
    """
    Get the shape of fields of view, the number of channels and
    the shape of individual channels in the images indicated in the dataframe.

    Assumes that all images in the dataframe have the same shape (and number of channels).

     Parameters:
        df: pandas DataFrame containing the metadata information, including the file names of the fields of view.
        fov_dir: path to the directory containing the fields of view.
        fov_clm: name of the column in df containing the file names of the fields of view.
        channel_axis: int, axis corresponding to the channels in the image array.
        null_value: value to return if no valid image files are found in the dataframe.
    Returns:
        ch_number: int, number of channels in the images.
     """

    # set default fov_clm name
    if fov_clm is None:
        fov_clm='ome_tif_file_name'

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
                # get fov shape
                fov_shape = fov.shape

                # print the shape of fov
                print(f"field of view shape: {fov_shape}")

                # get the number of channels
                ch_number = fov.shape[channel_axis]

                # print the number of channels found
                print(f"number of channels found: {ch_number}")

                # get the channels' shape
                ch_shape = tuple([fov.shape[i] for i in range(len(fov.shape)) if i!=channel_axis])

                # print the channels's shape
                print((f"shape of channels: {ch_shape}"))

            # except block to handle cases where the file cannot be read
            else:
                pass

        # if the end of the dataframe is reached
        else:

            # print a message indicating no valid image files were found
            print("No valid image files found in the dataframe.")

            # assign fov_shape ch_number and ch_shape to null values
            fov_shape = null_value
            ch_number = null_value
            ch_shape = null_value

        # increment index
        i += 1

    return fov_shape, ch_number, ch_shape
