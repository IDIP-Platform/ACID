import numpy as np

def my_line(x, a, b):
    return a*x + b

def my_polynomial(x, *coeff):
    return np.polyval(coeff, x)

def fit_polynomial(seq,
                   polyorder=1,
                   array_kwargs:dict|None=None,
                   arange_kwargs:dict|None=None,
                   polyfit_kwargs:dict|None=None):
    """
    https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html
    """
    if array_kwargs is None:
        array_kwargs={}

    if arange_kwargs is None:
        arange_kwargs={}
    
    if polyfit_kwargs is None:
        polyfit_kwargs={}

    y = np.array(seq)
    x = np.arange(len(y))
    return np.polyfit(x, y, polyorder)