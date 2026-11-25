import pytest


@pytest.fixture
def splitting_config(stage_config):
    return stage_config("dataset_splitting")
