import os
import tifffile
import numpy as np
import pandas as pd
import napari
import seaborn as sns

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


def plot_qc_by_channel_condition(metadata_df,
                              qc_clm,
                              condition_clm,
                              ax,
                              num_channels,
                              ch_measurement_separator="-",
                              violin_kws=None,
                              strip_kws=None,
                              fig=None,
                              common_xlabel=None,
                              common_ylabel=None):
    """Plot QC violin+strip per channel grouped by channel and condition.

    Parameters
    - metadata_df: pandas.DataFrame with measurement and condition columns
    - qc_clm: str base name of the per-channel QC columns
    - ch_measurement_separator: str separator between base name and channel index
    - condition_clm: str name of the categorical column to use on x-axis (conditions)
    - ax: sequence of matplotlib Axes where each channel will be plotted
    - num_channels: int number of channels to plot
    - violin_kws: dict passed to sns.violinplot (defaults to {'inner': None})
    - strip_kws: dict passed to sns.stripplot (defaults to {'color':'black','size':3})
    - fig: optional Matplotlib Figure to apply common labels to (inferred from `ax` if None)
    - common_xlabel: optional string to set as a shared x-label for the figure
    - common_ylabel: optional string to set as a shared y-label for the figure
    """
    if violin_kws is None:
        violin_kws = {"inner": None}
    if strip_kws is None:
        strip_kws = {"color": "black", "size": 3}

    # ensure no 'ax' key is provided in kws (ax is passed explicitly)
    assert 'ax' not in violin_kws and 'ax' not in strip_kws, (
        "Do not pass 'ax' inside violin_kws or strip_kws; ax is provided by the caller."
    )

    # ensure ax is indexable and has enough subplots
    try:
        n_axes = len(ax)
    except TypeError:
        n_axes = 1
    assert n_axes >= num_channels, (
        f"Not enough axes to plot {num_channels} channels (got {n_axes})."
    )

    for ch in range(num_channels):
        ch_qc_column_name = f"{qc_clm}{ch_measurement_separator}{ch}"
        sns.violinplot(x=condition_clm, y=ch_qc_column_name, data=metadata_df, ax=ax[ch], **violin_kws)
        sns.stripplot(x=condition_clm, y=ch_qc_column_name, data=metadata_df, ax=ax[ch], **strip_kws)
        ax[ch].set_title(f"{qc_clm} for channel {ch} and {condition_clm}")
        ax[ch].set_xlabel(f"{condition_clm}")
        ax[ch].set_ylabel(f"{qc_clm}")

    # set common axis labels on the figure if requested
    if common_xlabel is not None or common_ylabel is not None:
        if fig is None:
            try:
                fig = ax[0].figure
            except Exception:
                # ax might be a single Axes
                fig = ax.figure

        if common_xlabel is not None:
            try:
                fig.supxlabel(common_xlabel)
            except Exception:
                fig.text(0.5, 0.02, common_xlabel, ha='center')

        if common_ylabel is not None:
            try:
                fig.supylabel(common_ylabel)
            except Exception:
                fig.text(0.02, 0.5, common_ylabel, va='center', rotation='vertical')
