import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Import required modules
    import numpy as np
    import os
    import pandas as pd
    from skimage import io
    from feature_extraction.measure_haralick import measure_haralick_features

    return io, measure_haralick_features, np, os


@app.cell
def _():
    from dask.distributed import Client
    client = Client()
    print(client)
    return (client,)


@app.cell
def _(client):
    client
    return


@app.cell
def _(io, os):
    # Indicate the path to the input image
    input_image___path = os.path.join(os.getcwd(),"secondary_output")

    # get the name of the input image
    input_image___name = "260202_fov_example.ome.tif"

    # open input image
    input_real_image = io.imread(os.path.join(input_image___path, input_image___name))
    print(input_real_image.shape)

    # the first image of the channel axis is the segmentation mask. The following images are channels, to be analysed
    return (input_real_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    the file is a 2720x2720 field of view with 7 channels.

    The first of the 7 channels is the segmentation mask.

    Objects are individual cells.

    The remaining 6 channels are different imaged structures / imaging modalities.
    """)
    return


@app.cell
def _(input_real_image, measure_haralick_features, np):
    real_data_haralick_features = measure_haralick_features(image=input_real_image[...,1:3],
                                                 label_image=input_real_image[...,0],
                                                 props=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'],
                                                 distances=[5, 15, 49],
                                                 angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                                                 channel_axis=-1,
                                                 window_shape=50,
                                                 glcm_daskbag_kwargs={'npartitions': 10})

    real_data_haralick_features

    # single channel,, window_shape==50, 6 prop, 3 distance [5, 15, 49], 4 angles [0, np.pi/4, np.pi/2, 3*np.pi/4], 10 partitions -> 8m 16s
    # double channels,, window_shape==50, 6 prop, 3 distance [5, 15, 49], 4 angles [0, np.pi/4, np.pi/2, 3*np.pi/4], 12 partitions -> 15m 33s
    return


if __name__ == "__main__":
    app.run()

