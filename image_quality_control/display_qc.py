import os
import tifffile
import numpy as np
import pandas as pd
import napari

def high_low_QC_fov(metadata_df,
                    qc_clm,
                    fov_clm,
                    ch_measurement_sep,
                    fov_dir,
                    channel_ax):
    
    # copy the metadata dataframe
    metadata_df_copy = metadata_df.copy()

    # sort the dataframe by the column
    metadata_df_sorted = metadata_df_copy.sort_values(by=qc_clm, ascending=True)

    # get the fov with the lowest value
    fov_row_low = metadata_df_sorted.iloc[0][fov_clm]

    # get the fov with the highest plls value
    fov_row_high = metadata_df_sorted.iloc[-1][fov_clm]

    # open the images
    img_low = tifffile.imread(os.path.join(fov_dir, str(fov_row_low)))
    img_high = tifffile.imread(os.path.join(fov_dir, str(fov_row_high)))

    # select the channel to visualize
    img_low = img_low.take(indices=int(qc_clm.split(sep=ch_measurement_sep)[-1]), axis=channel_ax)
    img_high = img_high.take(indices=int(qc_clm.split(sep=ch_measurement_sep)[-1]), axis=channel_ax)
            
    return img_low, img_high, fov_row_low, fov_row_high


def display_high_low_qc(metadata_df,
               napari_viewer,
               qc_clm_collection,
               fov_clm,
               ch_measurement_sep,
               fov_dir,
               channel_ax):

    # try to visualize fields of view with min-max QCs in napari
    try:
        for col in qc_clm_collection:
            print(col)
            # get images with highest and lowest QC values
            img_low, img_high, fov_row_low, fov_row_high = high_low_QC_fov(metadata_df,
                                                                        qc_clm=col,
                                                                        fov_clm=fov_clm,
                                                                        ch_measurement_sep=ch_measurement_sep,
                                                                        fov_dir=fov_dir,
                                                                        channel_ax=channel_ax)
            print(img_low.shape, img_high.shape)

            # visualize using napari
            napari_viewer.add_image(img_low, name=f"lowest {col}: {fov_row_low}")
            napari_viewer.add_image(img_high, name=f"highest {col}: {fov_row_high}")

            # return napari_viewer

    # print message if visualization fails
    except:
        print("could not visualize fields of view with min-max QCs in napari")


def display_qc(df_to_display,
               napari_viewer,
               metadata_df,
               fov_column_name,
               fov_directory,
               channel_to_display,
               channel_axis,
               qc_to_display,
               ch_measurement_separator):

    for row in df_to_display.index:

        # get the fov name
        fov_to_display_name= metadata_df.loc[row, fov_column_name]

        # open the images
        fov_to_display = tifffile.imread(os.path.join(fov_directory, str(fov_to_display_name)))

        # select the channel to visualize
        fov_ch_to_display = fov_to_display.take(indices=channel_to_display, axis=channel_axis)

        # visualize using napari
        napari_viewer.add_image(fov_ch_to_display, name=f"{qc_to_display}: {metadata_df.loc[row, f"{qc_to_display}{ch_measurement_separator}{channel_to_display}"]}, {fov_to_display_name}")
