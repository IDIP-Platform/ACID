import pandas as pd
import pytest

from acid.data_preparation.map_category import map_treatments
from acid.data_preparation.train_test_split import split_dataset_train_test


@pytest.fixture
def splitting_config(stage_config):
    return stage_config("dataset_splitting")


def test_map_treatments_adds_formatted_treatment_per_well(splitting_config):
    metadata_df = pd.DataFrame(
        {"experiment": ["A07.2", "A07.2", "A07.3"], "well": ["well1", "well2", "well1"]}
    )
    plate_layout_df = pd.DataFrame(
        {
            "experiment": ["A07.2", "A07.2", "A07.3"],
            "well": [1, 2, 1],
            "treatment": ["uninfected, no cpd", "uninfected + NITD-688", "infected"],
        }
    )

    result = map_treatments(metadata_df, plate_layout_df, splitting_config)

    assert result["treatment"].tolist() == [
        "uninfected_nocpd",
        "uninfected_NITD_688",
        "infected",
    ]
    assert "treatment" not in metadata_df.columns


def test_split_dataset_train_test_adds_reproducible_split_column(splitting_config):
    metadata_df = pd.DataFrame({"a": range(10)})

    first = split_dataset_train_test(metadata_df, splitting_config)
    second = split_dataset_train_test(metadata_df, splitting_config)

    column = splitting_config.dataset_split.column
    assert first[column].value_counts().to_dict() == {1: 7, 0: 3}
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("fraction", [0, 1, 1.5])
def test_split_dataset_train_test_rejects_invalid_test_fraction(
    splitting_config, fraction
):
    splitting_config.processing.test_fraction = fraction

    with pytest.raises(ValueError, match="test_fraction"):
        split_dataset_train_test(pd.DataFrame({"a": range(10)}), splitting_config)


def test_split_dataset_train_test_rejects_identical_labels(splitting_config):
    labels = splitting_config.dataset_split.labels
    labels.test = labels.train

    with pytest.raises(ValueError, match="labels must differ"):
        split_dataset_train_test(pd.DataFrame({"a": range(10)}), splitting_config)


def test_split_dataset_train_test_rejects_unknown_split_unit(splitting_config):
    splitting_config.processing.split_unit = "well"

    with pytest.raises(ValueError, match="metadata_row"):
        split_dataset_train_test(pd.DataFrame({"a": range(10)}), splitting_config)
