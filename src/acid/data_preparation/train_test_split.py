import pandas as pd
from omegaconf import DictConfig, OmegaConf
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



def split_dataset_train_test(metadata_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Add a reproducible train/test split column to the metadata.

    Each metadata row (field of view) is assigned to train or test with
    `add_train_test_split_clm`; the same `random_state` gives the same split.

    Args:
        metadata_df (pd.DataFrame): Metadata to split, one row per field of
            view.
        cfg (DictConfig): The `dataset_splitting` section. Reads
            `processing.split_unit` (only `"metadata_row"` is supported),
            `processing.test_fraction` (strictly between 0 and 1),
            `processing.random_state`, `processing.concat_kwargs`,
            `dataset_split.column` (name of the new column) and
            `dataset_split.labels.train`/`test` (values written into it).

    Returns:
        pd.DataFrame: Metadata with the split column added.

    Raises:
        ValueError: If `split_unit` is not `"metadata_row"`, `test_fraction`
            is not strictly between 0 and 1, or the train and test labels are
            equal.
    """
    if cfg.processing.split_unit != "metadata_row":
        raise ValueError("Only metadata_row splitting is implemented")
    if not 0 < cfg.processing.test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    labels = cfg.dataset_split.labels
    if labels.train == labels.test:
        raise ValueError("Train and test labels must differ")
    return add_train_test_split_clm(
        df=metadata_df,
        test_size=cfg.processing.test_fraction,
        is_train_column=cfg.dataset_split.column,
        train_val=labels.train,
        test_val=labels.test,
        train_test_split_kwargs={"random_state": cfg.processing.random_state},
        concat_kwargs=OmegaConf.to_container(
            cfg.processing.concat_kwargs, resolve=True
        ),
    )

