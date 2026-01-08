import os
import tifffile
import numpy as np
import pandas as pd

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