import pandas as pd

from acid.utils.fov_axis_utils import get_fov_ch_shape


def test_fov_shape_returns_null_values_when_no_image_can_be_opened(tmp_path):
    metadata = pd.DataFrame({"ome_tif_file_name": ["missing.ome.tif"]})

    result = get_fov_ch_shape(
        metadata,
        tmp_path,
        fov_clm="ome_tif_file_name",
        null_value=None,
    )

    assert result == (None, None, None)
