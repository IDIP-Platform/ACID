# TODO: Add doctrings and reduce cognitive complexity


import logging
from enum import IntEnum
from pathlib import Path

import tifffile

from acid.utils.filesystem.filesystem import list_directory_entries
from acid.utils.get_defaults import (
    default_file_name,
    default_multifile_name,
)
from acid.utils.miscellaneous_utils import map_image_to_condition

# ---- Setting built-in logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# ---------------  PUBLIC INTERFACE  -----------------------
# ----------------------------------------------------------


class BackgroundFunctionStrategy(IntEnum):
    DATASET = 1
    WELL = 2
    GRID_POSITION = 3


def load_background_function(background_config, metadata_df=None):
    strategy = background_config["background_function_strategy"]
    background_files = _list_background_files(background_config)

    if not background_files:
        raise ValueError(
            f"No background function files match the selection in {background_config['directory']}"
        )

    if strategy == BackgroundFunctionStrategy.DATASET:
        return _load_strategy_1_background(
            background_config,
            background_files,
        )

    if strategy == BackgroundFunctionStrategy.WELL:
        return _load_strategy_2_background(
            background_config,
            background_files,
            metadata_df,
        )

    if strategy == BackgroundFunctionStrategy.GRID_POSITION:
        return _load_strategy_3_background(
            background_config,
            background_files,
            metadata_df,
        )

    raise ValueError(
        f"Invalid background_function_strategy: {strategy}. "
        "Please select either 1, 2 or 3."
    )


# ----------------------------------------------------------
# ---------------  HELPER FUNCTIONS  -----------------------
# ----------------------------------------------------------


def _load_strategy_1_background(
    background_config: dict,
    background_files: list[str],
):
    """Load one background function image for the whole dataset.

    Strategy 1 uses a single background function file for all fields of view.
    The file can be selected explicitly by timestamp/name fragment, or
    automatically using the default-selection settings.
    """
    background_directory = Path(background_config["directory"])
    selection = _selection_settings(background_config)
    background_function_timestamp = selection.get("filename", "default")

    if _uses_default_background_timestamp(background_function_timestamp):
        background_filename = default_file_name(
            file_list=background_files,
            from_file_name=_use_filename_date(selection),
            directory_path=background_directory,
            separator=selection.get("filename_date_separator", "_"),
            date_position=selection.get("filename_date_position", 0),
            date_format=selection.get("filename_date_format", "%Y%m%d"),
            reverse=_select_newest(selection),
        )
    else:
        matching_files = [
            file_name
            for file_name in background_files
            if background_function_timestamp in file_name
        ]

        if len(matching_files) != 1:
            raise ValueError(
                f"Expected exactly one background file matching {background_function_timestamp!r}, "
                f"but found {len(matching_files)}: {matching_files}"
            )

        background_filename = matching_files[0]

    background_path = background_directory / background_filename

    if not background_path.is_file():
        raise FileNotFoundError(
            f"Background function file not found: {background_path}"
        )

    background_function = tifffile.imread(background_path)

    return background_function, background_filename


def _load_strategy_2_background(background_config, background_files, metadata_df):
    return _load_condition_mapped_background(
        background_config=background_config,
        background_files=background_files,
        metadata_df=metadata_df,
        condition_column_name=background_config["well_column_name"],
        strategy_name="2",
    )


def _load_strategy_3_background(background_config, background_files, metadata_df):
    return _load_condition_mapped_background(
        background_config=background_config,
        background_files=background_files,
        metadata_df=metadata_df,
        condition_column_name=background_config["gridpos_column_name"],
        strategy_name="3",
    )


def _uses_default_background_timestamp(background_timestamp):
    return (
        background_timestamp is None
        or background_timestamp == ""
        or str(background_timestamp).lower() == "default"
    )


def _list_background_files(background_config):
    background_directory = Path(background_config["directory"])

    if not background_directory.is_dir():
        raise FileNotFoundError(
            f"Background function directory not found: {background_directory}"
        )

    selection = _selection_settings(background_config)
    return list_directory_entries(
        directory=background_directory,
        include=selection.get("include"),
        exclude=selection.get("exclude"),
    )


def _selection_settings(background_config):
    if "file_selection" in background_config:
        return background_config["file_selection"]

    return {
        "filename": background_config.get(
            "filename",
            background_config.get("background_function_timestamp", "default"),
        ),
        "include": background_config.get("default_bg_funct_file_target"),
        "exclude": background_config.get("default_bg_funct_file_exclude"),
        "date_source": (
            "filename"
            if background_config.get("background_from_file_name", False)
            else "modified_time"
        ),
        "select": (
            "newest"
            if background_config.get("background_default_reverse", True)
            else "oldest"
        ),
        "filename_date_separator": background_config.get(
            "background_default_separator", "_"
        ),
        "filename_date_position": background_config.get(
            "background_default_date_position", 0
        ),
        "filename_date_format": background_config.get(
            "background_default_date_format", "%Y%m%d"
        ),
    }


def _use_filename_date(selection):
    date_source = selection.get("date_source", "modified_time")
    if date_source not in {"filename", "modified_time"}:
        raise ValueError(f"Invalid background date_source: {date_source!r}")
    return date_source == "filename"


def _select_newest(selection):
    choice = selection.get("select", "newest")
    if choice not in {"newest", "oldest"}:
        raise ValueError(f"Invalid background select: {choice!r}")
    return choice == "newest"


def _load_condition_mapped_background(
    background_config,
    background_files,
    metadata_df,
    condition_column_name,
    strategy_name,
):
    if metadata_df is None:
        raise ValueError(
            f"metadata_df is required for background function strategy {strategy_name}"
        )

    if condition_column_name not in metadata_df.columns:
        raise ValueError(
            f"Column {condition_column_name!r} was not found in metadata_df"
        )

    background_directory = Path(background_config["directory"])
    selection = _selection_settings(background_config)
    background_timestamp = selection.get("filename", "default")

    if _uses_default_background_timestamp(background_timestamp):
        background_file_names = default_multifile_name(
            file_list=background_files,
            from_file_name=_use_filename_date(selection),
            directory_path=background_directory,
            separator=selection.get("filename_date_separator", "_"),
            date_position=selection.get("filename_date_position", 0),
            date_format=selection.get("filename_date_format", "%Y%m%d"),
            reverse=_select_newest(selection),
        )

        print("using the following files as default background function files:")
        for file_name in background_file_names:
            print(file_name)
    else:
        background_file_names = [
            file_name
            for file_name in background_files
            if background_timestamp in file_name
        ]

    unique_conditions = metadata_df[condition_column_name].unique()

    if len(background_file_names) != len(unique_conditions):
        raise ValueError(
            f"The number of background function files ({len(background_file_names)}) "
            f"does not match the number of unique conditions "
            f"({len(unique_conditions)}). Please check the background function "
            "directory and the metadata dataframe."
        )

    background_files_by_condition = map_image_to_condition(
        condition_list=unique_conditions,
        image_file_names=background_file_names,
        assert_file_number=background_config.get(
            "background_default_assert_file_number"
        ),
        number_of_files_expected=background_config.get(
            "background_default_number_of_files_expected"
        ),
    )

    background_by_condition = _load_backgrounds_by_condition(
        background_files_by_condition=background_files_by_condition,
        background_directory=background_directory,
    )

    return background_by_condition, background_files_by_condition


def _load_backgrounds_by_condition(
    background_files_by_condition,
    background_directory,
):
    background_by_condition = {}

    for condition, background_file_names in background_files_by_condition.items():
        if len(background_file_names) != 1:
            raise ValueError(
                f"Expected exactly one background file for condition {condition!r}, "
                f"but found {len(background_file_names)}: {background_file_names}"
            )

        background_file_name = background_file_names[0]
        background_path = background_directory / background_file_name

        if not background_path.is_file():
            raise FileNotFoundError(
                f"Background function file not found: {background_path}"
            )

        background_by_condition[condition] = tifffile.imread(background_path)

    return background_by_condition
