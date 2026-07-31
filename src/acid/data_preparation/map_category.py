import pandas as pd
from collections.abc import Sequence
from ..utils.str_utils import extract_number, split_text_and_number


def map_fov_categories(
    plate_layout_df: pd.DataFrame,
    well: int,
    experiment: str,
    well_column: str | None = None,
    experiment_column: str | None = None,
    treatment_column: str | None = None,
):

    # set defaults
    if well_column is None:
        well_column = "well"

    if experiment_column is None:
        experiment_column = "experiment"

    if treatment_column is None:
        treatment_column = "treatment"

    # copy input dataframe
    original_plate_layout_df = plate_layout_df.copy()

    return original_plate_layout_df[
        (original_plate_layout_df[experiment_column] == experiment)
        & (original_plate_layout_df[well_column] == well)
    ][treatment_column].values[0]


def map_fov_categories_df(
    metadata_df: pd.DataFrame,
    plate__layout_df: pd.DataFrame,
    well__column: str | None = None,
    experiment__column: str | None = None,
    treatment__column: str | None = None,
    wellasint_column: str | None = None,
    drop_wellasint_column: bool = True,
) -> pd.DataFrame:

    # set defaults
    if well__column is None:
        well__column = "well"

    if experiment__column is None:
        experiment__column = "experiment"

    if treatment__column is None:
        treatment__column = "treatment"

    if wellasint_column is None:
        wellasint_column = "int_well"

    # copy input dataframes
    original_metadata_df = metadata_df.copy()
    original_plate_layout_df = plate__layout_df.copy()

    # add a column transforming wells in numbers
    original_metadata_df[wellasint_column] = original_metadata_df.apply(
        lambda row: extract_number(s=row[well__column]), axis=1
    )

    # add a column with the treatment
    original_metadata_df[treatment__column] = original_metadata_df.apply(
        lambda row: map_fov_categories(
            plate_layout_df=original_plate_layout_df,
            well=row[wellasint_column],
            experiment=row[experiment__column],
            well_column=well__column,
            experiment_column=experiment__column,
            treatment_column=treatment__column,
        ),
        axis=1,
    )

    return original_metadata_df


def map_layout_condition(
    condition_list: Sequence,
    number_type: int | float | None = None,
    regex: str | None = None,
    null_value: tuple = (None, None),
    return_unique: bool = True,
) -> tuple:
    """
    Splits a string into a text prefix and a trailing numeric part.

    Parameters
    ----------
    s : str
        Input string expected to end with one or more digits (no separator).
    number_type : type or None, optional
        If None, the numeric part is returned as a string.
        If a type is provided (e.g. int, float), the numeric part is
        converted using that type.

    number_type: int, float or None. Optional. Default None.
        If int or float, the function will try to convert the second part of the splat string
        in, respectively an integer or a float.
        If None (default), non type conversion will be tried.

    regex: regex str|None.
        The regex expression to use for splitting the input string.

    null_value. tuple. Optional, default (None, None).
        The value returned if the input string can't be split using the input regex expression.


    Returns
    -------
    tuple
        (prefix, number) if the string ends with digits.
        If no trailing number is found, (None, None) is returned.
    """
    if regex is None:
        regex = r"^(.*?)(\d+)$"

    prefixes = []
    suffixes = []
    for c in condition_list:
        pre_fix, suf_fix = split_text_and_number(
            c, regex=regex, number_type=number_type, null_value=null_value
        )
        prefixes.append(pre_fix)
        suffixes.append(suf_fix)

    if return_unique:
        unique_prefixes = list(set(prefixes))
        unique_suffixes = list(set(suffixes))
        return unique_prefixes, unique_suffixes
    else:
        return prefixes, suffixes
