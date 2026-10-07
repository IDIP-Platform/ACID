import pandas as pd
import pytest

from acid.data_preparation.map_category import map_treatments


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
