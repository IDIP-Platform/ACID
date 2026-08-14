import pandas as pd

def format_series_str(input_series:pd.Series,
                      symb_ols:list|None=None,
                      spa_ces:list|None=None,
                      symb_ol_replace:str|None=None,
                      spa_ce_replace:str|None=None)->pd.Series:
    """
    Formats a pandas series of strings by:
    - substituting all symbols in symb_ols with symb_ol_replace (default '_')
    - substituting all spaces in spa_ces with spa_ce_replace (default '', which leads dropping the spaces)
    - making all caracters lowercase.

    Returns the formatted series (as a copy).
    """

    # set default symbols and spaces
    if symb_ols is None:
        symb_ols=['~', '!', '@', '#', '$', '%', '^', '&', '*',
                  '(', ')', '`', ';', '<', '>', '.', '?', ',', '[',
                  ']', '{', '}', '|', '°', '§', '/', 'ß', '+', '-']

    if spa_ces is None:
        spa_ces=[' ', '  ', '   ']

    if symb_ol_replace is None:
        symb_ol_replace='_'

    if spa_ce_replace is None:
        spa_ce_replace=''

    # copy the input series
    input_series_copy = input_series.copy()

    # replace all symbols in symb_ols with what indicatd in symb_ol_replace
    for symb_ol in symb_ols:

        input_series_copy = input_series_copy.str.replace(symb_ol,symb_ol_replace)

    # replace all spaces in spa_ces with what indicated in spa_ce_replace
    for spa_ce in spa_ces:

        input_series_copy = input_series_copy.str.replace(spa_ce, spa_ce_replace)

    # tranform all characters to lowercase
    input_series_copy.str.lower()

    return input_series_copy
