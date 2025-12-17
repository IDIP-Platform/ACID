import numpy as np
import pandas as pd
from typing import Iterable, Sequence, Optional, Any

def add_flag_column(
    df: pd.DataFrame,
    lowpass_thres: Sequence[float],
    highpass_thres: Sequence[float],
    cols: Optional[Sequence[str]] = None,
    flag_col: str = "flag",
    flag_value: Any = "flag_value",
    ok_value: Any = "ok",
) -> pd.DataFrame:
    """
    Add a flag column to a DataFrame based on per-column low/high thresholds
    using fully vectorized NumPy operations.

    A row is flagged if ANY selected column has a value below its low-pass
    threshold OR above its high-pass threshold.

    -----------------------------------------------------------------------
    Parameters
    -----------------------------------------------------------------------
    df : pandas.DataFrame
        Input DataFrame.

    lowpass_thres : iterable
        Iterable of lower thresholds.
        The order MUST match the order of the selected columns.

    highpass_thres : iterable
        Iterable of upper thresholds.
        The order MUST match the order of the selected columns.

    cols : list-like or None, optional
        Columns to apply thresholds to.
        - If None, all columns in df are used.
        - Otherwise, only df[cols] is checked.

    flag_col : str, default "flag"
        Name of the column that will store the flag values.

    flag_value : str, default "flag_value"
        Value assigned when a row violates any threshold.

    ok_value : str, default "ok"
        Value assigned when a row is fully within thresholds.

    -----------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------
    pandas.DataFrame
        DataFrame with an added flag column.

    -----------------------------------------------------------------------
    Limitations & Edge Cases
    -----------------------------------------------------------------------
    1. Column order dependency
       The threshold iterables are applied strictly by position.
       If column order and threshold order do not match, results will be
       logically incorrect without raising an error.

    2. Numeric data requirement
       Selected columns must be numeric.
       Strings, mixed dtypes, or objects may raise errors or produce
       incorrect comparisons.

    3. NaN values in data
       NaN comparisons (< or >) evaluate to False.
       Rows containing NaNs will NOT be flagged unless explicitly handled.

    4. NaN values in thresholds
       NaNs in threshold arrays disable thresholding for that column.

    5. Memory usage
       df_sel.to_numpy() creates a full copy of the selected data.
       Very large DataFrames may cause high memory consumption.

    6. Fixed thresholds per column
       Thresholds must be constant per column.
       Row-dependent or dynamic thresholds are not supported.

    7. Exclusive bounds
       The function uses strict comparisons (< low, > high).
       Inclusive bounds must be implemented explicitly if required.
    """

    # --------------------------------------------------
    # 1. Select columns to apply threshold logic
    # --------------------------------------------------
    # If cols is None, use the entire DataFrame.
    # Otherwise, restrict operations to df[cols].
    df_sel = df if cols is None else df[cols]

    # --------------------------------------------------
    # 2. Validate threshold dimensions
    # --------------------------------------------------
    # The number of thresholds must exactly match the number
    # of selected columns for broadcasting to work correctly.
    if len(df_sel.columns) != len(lowpass_thres) or len(df_sel.columns) != len(highpass_thres):
        raise ValueError(
            "lowpass_thres and highpass_thres must match the number "
            "of selected columns"
        )

    # --------------------------------------------------
    # 3. Convert thresholds to NumPy arrays
    # --------------------------------------------------
    # NumPy arrays are required for broadcasting comparisons
    # against the DataFrame's underlying NumPy matrix.
    low = np.asarray(lowpass_thres)
    high = np.asarray(highpass_thres)

    # --------------------------------------------------
    # 4. Convert selected DataFrame to NumPy array
    # --------------------------------------------------
    # Shape of `values`:
    #   (n_rows, n_columns)
    #
    # This enables vectorized column-wise comparisons
    # without Python-level loops.
    values = df_sel.to_numpy()

    # --------------------------------------------------
    # 5. Vectorized threshold comparison
    # --------------------------------------------------
    # Broadcasting logic:
    # - values has shape (n_rows, n_cols)
    # - low/high have shape (n_cols,)
    #
    # NumPy automatically aligns thresholds column-wise,
    # producing a boolean matrix of shape (n_rows, n_cols).
    out_of_bounds = (values < low) | (values > high)

    # --------------------------------------------------
    # 6. Reduce column-wise violations to row-wise flags
    # --------------------------------------------------
    # any(axis=1) checks whether at least one column in a row
    # violates its thresholds.
    row_flag = out_of_bounds.any(axis=1)

    # --------------------------------------------------
    # 7. Write flag column to DataFrame
    # --------------------------------------------------
    # np.where assigns the flag_value where row_flag is True
    # and ok_value otherwise.
    df[flag_col] = np.where(row_flag, flag_value, ok_value)

    return df
