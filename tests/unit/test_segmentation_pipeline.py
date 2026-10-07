import numpy as np
import pytest
import tifffile

from acid.segmentation.pipeline import (
    build_segmentation_image_metadata,
    cast_mask_to_output_dtype,
    copy_selected_field_of_view_metadata,
    get_channel_shape,
    get_segmentation_metadata_columns,
    make_segmentation_success_result,
    make_segmentation_output_filename,
    resize_segmentation_mask,
    save_segmentation_mask,
    segment_objects,
)


class FakeModel:
    """Stand-in for Cellpose: one square object on the preprocessed grid."""

    name = "fake"
    version = "0.0"

    def __init__(self):
        self.calls = []

    def eval(self, image, **kwargs):
        self.calls.append(kwargs)
        shape = tuple(
            size
            for axis, size in enumerate(image.shape)
            if axis != kwargs["channel_axis"]
        )
        masks = np.zeros(shape, dtype=np.int32)
        masks[1:3, 1:3] = 1
        return masks, None, None


@pytest.fixture
def segmentation_config(stage_config, tmp_path):
    config = stage_config("object_segmentation")
    config.image_saving.directory = str(tmp_path / "masks")
    return config


def test_get_segmentation_metadata_columns_has_twelve_names(segmentation_config):
    columns = get_segmentation_metadata_columns(
        segmentation_config.metadata.dataframe_columns
    )

    assert len(columns) == 12
    assert len(set(columns)) == 12


def test_get_channel_shape_drops_channel_axis():
    assert get_channel_shape(np.zeros((5, 16, 12)), channel_axis=0) == (16, 12)
    assert get_channel_shape(np.zeros((16, 12, 5)), channel_axis=2) == (16, 12)


def test_cast_mask_to_output_dtype_keeps_mask_when_none():
    mask = np.zeros((2, 2), dtype=np.int32)

    assert cast_mask_to_output_dtype(mask, None) is mask
    assert cast_mask_to_output_dtype(mask, "uint16").dtype == np.uint16


def test_segment_objects_passes_configured_settings_to_model(segmentation_config):
    model = FakeModel()
    processing = segmentation_config.processing

    masks, _, _ = segment_objects(np.zeros((2, 8, 8)), model, segmentation_config)

    assert masks.shape == (8, 8)
    assert model.calls == [
        {
            "flow_threshold": processing.flow_threshold,
            "cellprob_threshold": processing.cellprob_threshold,
            "diameter": processing.diameter,
            "channel_axis": processing.channel_axis,
        }
    ]


def test_resize_segmentation_mask_upsamples_and_keeps_labels(segmentation_config):
    mask = np.array([[0, 1], [2, 2]], dtype=np.int32)

    resized = resize_segmentation_mask(mask, (4, 4), segmentation_config)

    assert resized.shape == (4, 4)
    assert resized.dtype == np.int32
    assert set(np.unique(resized)) == {0, 1, 2}


def test_make_segmentation_output_filename(segmentation_config):
    assert (
        make_segmentation_output_filename("a_bg.ome.tif", segmentation_config)
        == "a_bg_segmentation.ome.tif"
    )


def test_save_segmentation_mask_writes_tiff_into_given_directory(
    segmentation_config, tmp_path
):
    mask = np.arange(16, dtype=np.uint16).reshape(4, 4)

    path = save_segmentation_mask(
        "a_segmentation.ome.tif", mask, {"custom_x": "1"}, segmentation_config,
        output_directory=tmp_path / "out",
    )

    assert path == tmp_path / "out" / "a_segmentation.ome.tif"
    np.testing.assert_array_equal(tifffile.imread(path), mask)


def test_copy_selected_field_of_view_metadata_keeps_only_matching_entries(
    segmentation_config,
):
    entries = segmentation_config.metadata.image_metadata
    field_of_view_metadata = {
        f"custom_{entries.preproc_img_meta_raw_file_name_entry}": "a.nd2",
        f"custom_{entries.preproc_img_meta_x_physic_px_size_entry}": 0.65,
        "custom_unrelated_entry": "drop me",
    }

    result = copy_selected_field_of_view_metadata(
        {"custom_segmentation": "x"}, field_of_view_metadata, segmentation_config
    )

    assert result == {
        "custom_segmentation": "x",
        f"custom_{entries.preproc_img_meta_raw_file_name_entry}": "a.nd2",
        f"custom_{entries.preproc_img_meta_x_physic_px_size_entry}": 0.65,
    }


def test_build_segmentation_image_metadata_records_model_and_settings(
    segmentation_config, write_fov
):
    directory, _ = write_fov()
    entries = segmentation_config.metadata.image_metadata
    mask = np.zeros((16, 16), dtype=np.uint16)

    metadata = build_segmentation_image_metadata(
        directory / "a.ome.tif", mask, FakeModel(), segmentation_config
    )

    assert metadata[f"custom_{entries.segmentation_method_version_name}"] == "0.0"
    assert metadata[f"custom_{entries.segmented_img_meta_dtype_name}"] == "uint16"
    assert metadata[f"custom_{entries.segmented_img_meta_diameter_name}"] == (
        segmentation_config.processing.diameter
    )


def test_make_segmentation_success_result_fills_all_metadata_columns(
    segmentation_config,
):
    columns = segmentation_config.metadata.dataframe_columns

    result = make_segmentation_success_result(
        2, "a_bg.ome.tif", "a_bg_segmentation.ome.tif", np.dtype("uint16"),
        FakeModel(), segmentation_config,
    )

    assert result["success"] is True
    assert result[columns.metadata_df_method_version_clm_name] == "0.0"
    assert result[columns.metadata_df_output_dtype_name] == "uint16"
    assert set(get_segmentation_metadata_columns(columns)) <= set(result)
