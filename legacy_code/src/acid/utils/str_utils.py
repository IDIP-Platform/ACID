import re
import pandas as pd

def format_str(in_s:str,
               symb_ols:list|None=None,
               spa_ces:list|None=None,
               symb_ol_replace:str|None=None,
               spa_ce_replace:str|None=None)->pd.Series:
    """
    Formats string by:
    - substituting all symbols in symb_ols with symb_ol_replace (default '_')
    - substituting all spaces in spa_ces with spa_ce_replace (default '', which leads dropping the spaces)
    - making all caracters lowercase.

    Returns the formatted string.
    """

    # set defaults
    if symb_ol is None:
        symb_ols=['~', '!', '@', '#', '$', '%', '^', '&', '*',
                  '(', ')', '`', ';', '<', '>', '.', '?', ',', '[',
                  ']', '{', '}', '|', '°', '§', '/', 'ß', '+', '-']

    if spa_ces is None:
        spa_ces=[' ', '  ', '   ']

    if symb_ol_replace is None:
        symb_ol_replace='_'

    if spa_ce_replace is None:
        spa_ce_replace=''

    # store input s in an output string
    out_s = in_s

    # replace all symbols in symb_ols with what indicatd in symb_ol_replace
    for symb_ol in symb_ols:

        ous_s = out_s.replace(symb_ol,symb_ol_replace)

    # replace all spaces in spa_ces with what indicated in spa_ce_replace
    for spa_ce in spa_ces:

        ous_s = ous_s.replace(spa_ce, spa_ce_replace)

    # tranform all characters to lowercase
    ous_s = ous_s.lower()

    return ous_s


def extract_number(s:str,
                   regex:str|None=None,
                   return_type:int|float=int,
                   null_value:any=None)->int|float:
    """
    Extract the number in a string (s).

    s is exprected to have the number at the very end, with no separation between a digit and the number. For example:

    strings that work: "well1", "well2", "Ale1", "Ale11", "Ale0011", "WellOfAle333"...

    strings that don't work: "well 1", "well+1", "well_99"...
    """
    if regex is None:
        regex = r'(\d+)$'

    match = re.search(regex, s)
    return return_type(match.group(1)) if match else null_value



def split_text_and_number(s:str,
                          number_type:int|float|None=None,
                          regex:str|None=None,
                          null_value:tuple=(None, None))->tuple:
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

    regex: regex str|None. Optional, default r'^(.*?)(\d+)$'
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
        regex = r'^(.*?)(\d+)$'

    match = re.search(regex, s)
    if not match:
        return null_value


    prefix = match.group(1)
    number_str = match.group(2)

    if number_type is not None:
        try:
            return prefix, number_type(number_str)
        except (ValueError, TypeError):
            raise ValueError(
                f"Cannot convert '{number_str}' to {number_type}"
            )

    return prefix, number_str
