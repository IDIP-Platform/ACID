import os
import skimage.io
import dask.array as da
import dask


input_dir = ""
filenames = os.listdir(input_dir)

@dask.delayed
def load(filename):
    file_array = skimage.io.imread(os.path.join(input_dir, filename))
    mask = da.array.from_array(file_array[0,...])
    data = da.array.from_array(file_array[1:,...])
    return data, mask

@dask.delayed
def measure_hessian(data, mask):
    # generates a dataframe with measurements of data
    return # returns a dataframe - this is not compatible with map_overlap

@dask.delayed
def haralick_feature_map(data):
    # generates a feature map of the data
    return # returns an array - this is compatible with map_overlap

@dask.delayed
def measure_feature_map(data, mask):
    # generates a datafram with measurements of data
    return # returns a datafram - this is not compatible with map_overlap


def f(filenames):
    hessian_results = []
    haralick_results = []
    for filename in filenames:
        data, mask = load(filename)
        hessian_measurements = measure_hessian(data, mask)
        haralick_fm = data.map_overlap(haralick_feature_map)
        haralick_measurements = measure_feature_map(haralick_fm)
        hessian_results.append(hessian_measurements)
        haralick_results.append(haralick_measurements)

    return hessian_results, haralick_results 

dask.compute(f(filenames))

