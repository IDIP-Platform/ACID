import numpy as np
import pandas as pd
import pytest

from acid.image_processing.calculate_background_function import (
    calculate_background_function,
    import_fov,
)
from acid.utils.save_image import tifffile_save_ometiff


@pytest.fixture
def fov_stack_files(tmp_path):
    """Seven random 5-channel FOVs on disk and their metadata."""
    rng = np.random.default_rng(0)
    names = []
    for i in range(7):
        name = f"f{i}.ome.tif"
        tifffile_save_ometiff(
            tmp_path / name,
            data=rng.integers(500, 3000, (5, 32, 24)).astype(np.uint16),
            imagej=True,
            photometric="minisblack",
            metadata={},
        )
        names.append(name)
    return tmp_path, pd.DataFrame({"fov": names}, index=[4, 1, 9, 0, 3, 7, 2])


def test_import_fov_with_threads_matches_sequential_reading(fov_stack_files):
    directory, metadata_df = fov_stack_files
    kwargs = dict(
        df=metadata_df,
        fov_dir=str(directory),
        fov_clm="fov",
        fov_shape=(5, 32, 24),
        verbose=False,
    )

    sequential = import_fov(**kwargs)
    threaded = import_fov(**kwargs, n_workers=4)

    assert threaded.shape == (5, 32, 24, 7)
    np.testing.assert_array_equal(threaded, sequential)


@pytest.mark.parametrize("method", ["median", "mean"])
@pytest.mark.parametrize("n_fovs", [6, 7])
def test_calculate_background_function_with_threads_is_identical(method, n_fovs):
    stack = np.random.default_rng(1).random((5, 33, 17, n_fovs)).astype(np.float32)
    original = stack.copy()

    sequential = calculate_background_function(stack, method=method, verbose=False)
    threaded = calculate_background_function(
        stack, method=method, verbose=False, n_workers=8
    )

    assert threaded.shape == (5, 33, 17)
    assert threaded.dtype == sequential.dtype
    np.testing.assert_array_equal(threaded, sequential)
    np.testing.assert_array_equal(stack, original)


def test_calculate_background_function_rejects_unknown_method():
    with pytest.raises(ValueError, match="not recognized"):
        calculate_background_function(np.zeros((2, 2, 3)), method="mode")
