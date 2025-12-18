import numpy as np
import pandas as pd
from typing import Sequence, Optional, Any

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
    Add a flag column to a DataFrame based on per-column low-pass and high-pass
    thresholds using fully vectorized NumPy operations.

    A row is flagged if ANY selected column contains a value that is:
    - strictly higher than its low-pass threshold (low values are allowed), OR
    - strictly lower than its high-pass threshold (high values are allowed).

    This follows the signal-processing interpretation:
    - Low-pass  → low values pass, high values are rejected
    - High-pass → high values pass, low values are rejected

    -----------------------------------------------------------------------
    Parameters
    -----------------------------------------------------------------------
    df : pandas.DataFrame
        Input DataFrame.

    lowpass_thres : Sequence[float]
        Sequence of low-pass thresholds.
        Values ABOVE these thresholds are flagged.
        Order MUST match the selected columns.

    highpass_thres : Sequence[float]
        Sequence of high-pass thresholds.
        Values BELOW these thresholds are flagged.
        Order MUST match the selected columns.

    cols : Sequence[str] or None, optional
        Columns to which the threshold logic is applied.
        - If None, all columns in df are used.
        - Otherwise, only df[cols] is evaluated.

    flag_col : str, default "flag"
        Name of the column that will store the resulting flag values.

    flag_value : Any, default "flag_value"
        Value assigned when a row violates any threshold.

    ok_value : Any, default "ok"
        Value assigned when a row satisfies all thresholds.

    -----------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------
    pandas.DataFrame
        The input DataFrame with an added (or overwritten) flag column.

    -----------------------------------------------------------------------
    Vectorization Logic
    -----------------------------------------------------------------------
    - Data is converted to a NumPy matrix of shape (n_rows, n_columns).
    - Thresholds are converted to 1D NumPy arrays of shape (n_columns,).
    - NumPy broadcasting applies column-specific thresholds in one operation.
    - A row is flagged if any column violates its pass condition.

    -----------------------------------------------------------------------
    Limitations & Edge Cases
    -----------------------------------------------------------------------
    1. Column order dependency
       Thresholds are applied strictly by position.

    2. Numeric data requirement
       Selected columns must be numeric.

    3. NaN values in data
       NaNs do not trigger flags unless explicitly handled.

    4. NaN values in thresholds
       Disable thresholding for that column.

    5. Memory usage
       df_sel.to_numpy() creates a full copy of the selected data.

    6. Fixed thresholds per column
       Row-dependent thresholds are not supported.

    7. Exclusive bounds
       Comparisons are strictly > low-pass and < high-pass.

    8. Single-column case
       Works correctly if thresholds are sequences of length 1.
    """

    # --------------------------------------------------
    # 1. Select columns to apply threshold logic
    # --------------------------------------------------
    df_sel = df if cols is None else df[cols]

    # --------------------------------------------------
    # 2. Validate threshold dimensions
    # --------------------------------------------------
    if len(df_sel.columns) != len(lowpass_thres) or len(df_sel.columns) != len(highpass_thres):
        raise ValueError(
            "lowpass_thres and highpass_thres must match the number "
            "of selected columns"
        )

    # --------------------------------------------------
    # 3. Convert thresholds to NumPy arrays
    # --------------------------------------------------
    low = np.asarray(lowpass_thres)
    high = np.asarray(highpass_thres)

    # --------------------------------------------------
    # 4. Convert selected DataFrame to NumPy array
    # --------------------------------------------------
    values = df_sel.to_numpy()

    # --------------------------------------------------
    # 5. Vectorized pass-band logic (FINAL & CORRECT)
    # --------------------------------------------------
    # A value PASSES if:
    #   (value > highpass) AND (value < lowpass)
    #
    # A value is FLAGGED if it violates the pass band:
    #   (value <= highpass) OR (value >= lowpass)
    out_of_bounds = (values <= high) | (values >= low)

    # --------------------------------------------------
    # 6. Reduce column-wise violations to row-wise flags
    # --------------------------------------------------
    row_flag = out_of_bounds.any(axis=1)

    # --------------------------------------------------
    # 7. Write flag column to DataFrame
    # --------------------------------------------------
    df[flag_col] = np.where(row_flag, flag_value, ok_value)

    return df
