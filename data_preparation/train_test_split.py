import pandas as pd
from sklearn.model_selection import train_test_split


def add_train_test_split_clm(
    df: pd.DataFrame,
    test_size: float = 0.3,
    is_train_column: str | None = None,
    train_val: int = 1,
    test_val: int = 0,
    train_test_split_kwargs: dict | None = None,
    concat_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Splits a pandas DataFrame into train and test subsets, then returns a copy
    of the original DataFrame with an additional column marking which rows
    belong to the train set vs. the test set.

    Rows are reassembled and sorted so that the output has:
      • identical row order to the original DataFrame
      • identical index values
      • one new column indicating train/test membership

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    test_size : float, default=0.3
        Fraction of rows to assign to the test set.

    is_train_column : str, default="is_train"
        Name of the output column that marks train/test membership.

    train_val : int, default=1
        Value assigned to rows in the train set.

    test_val : int, default=0
        Value assigned to rows in the test set.

    train_test_split_kwargs : dict | None
        Additional keyword arguments passed to sklearn.model_selection.train_test_split.
        (The 'test_size' argument must not be included here.)

    concat_kwargs : dict | None
        Additional keyword arguments passed to pandas.concat.

    Returns
    -------
    pd.DataFrame
        The original dataframe (same order, same index), with one extra column.
    """

    # -------------------------
    # Default values
    # -------------------------
    if is_train_column is None:
        is_train_column = "is_train"

    if train_test_split_kwargs is None:
        train_test_split_kwargs = {"random_state": 42}

    if concat_kwargs is None:
        concat_kwargs = {}

    # ----------------------------------------------
    # Safety checks for arguments and existing column
    # ----------------------------------------------
    assert (
        "test_size" not in train_test_split_kwargs
    ), "test_size can't be passed in train_test_split_kwargs. Use the dedicated argument instead."

    assert (
        is_train_column not in df.columns
    ), f"Column '{is_train_column}' already exists in dataframe."

    # ------------------------
    # Copy input dataframe
    # ------------------------
    df_original = df.copy()

    # ----------------------------
    # Perform train-test split
    # ----------------------------
    train_df, test_df = train_test_split(
        df_original, test_size=test_size, **train_test_split_kwargs
    )

    # ----------------------------
    # Add membership column
    # ----------------------------
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[is_train_column] = train_val
    test_df[is_train_column] = test_val

    # ----------------------------
    # Reassemble and sort by index
    # ----------------------------
    result = pd.concat([train_df, test_df], **concat_kwargs).sort_index()

    # ----------------------------
    # Integrity checks
    # ----------------------------
    # Same index values?
    assert set(result.index) == set(df_original.index), "Index values differ."

    # Same index order?
    assert result.index.equals(
        df_original.index
    ), "Index order differs — result is not aligned to original."

    return result

