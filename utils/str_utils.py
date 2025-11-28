import re
import pandas as pd

def format_str(in_s:str,
               symb_ols:list=['~', '!', '@', '#', '$', '%', '^', '&', '*',
                                     '(', ')', '`', ';', '<', '>', '.', '?', ',', '[',
                                     ']', '{', '}', '|', '°', '§', '/', 'ß', '+', '-'],
               spa_ces:list=[' ', '  ', '   '],
               symb_ol_replace:str='_',
               spa_ce_replace:str='')->pd.Series:
    """
    Formats string by:
    - substituting all symbols in symb_ols with symb_ol_replace (default '_')
    - substituting all spaces in spa_ces with spa_ce_replace (default '', which leads dropping the spaces)
    - making all caracters lowercase.

    Returns the formatted string.
    """

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


def extract_number(s):
    """
    Extract the number in a string (s).
    
    s is exprected to have the number at the very end, with no separation between a digit and the number. For example:

    strings that work: "well1", "well2", "Ale1", "Ale11", "Ale0011", "WellOfAle333"...

    strings that don't work: "well 1", "well+1", "well_99"...
    """
    
    match = re.search(r'(\d+)$', s)
    return int(match.group(1)) if match else None