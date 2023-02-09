"""
Utility functions
"""
import typing
import contextlib
import os
import numpy as np


def kprint(**kwargs) -> None:
    """Prints keyword arguments in new lines"""
    out = map(lambda k: f"{k} = {kwargs[k]}\n", kwargs)
    print(''.join(list(out)))

def ezip(*args) -> typing.Generator:
    """Zip with enumerate generator. First item is index"""
    return enumerate(zip(*args))

def ezip_r(*args) -> typing.Generator:
    """`ezip` that reverses the elements on the input iterables"""
    rargs = tuple(reversed(arg) for arg in args)
    return ezip(*rargs)

def indexable(obj :typing.Any) -> bool:
    """ Returns true if an input object is indexable.
    E.g. lists, dicts, or anything that can be addressed with obj[index]
    """
    return hasattr(obj, '__getitem__')

@contextlib.contextmanager
def supress_stdout():
    """ Supresses write calls to stdout """
    with open(os.devnull, "w", encoding="utf-8") as null:
        with contextlib.redirect_stdout(null):
            yield

def is_mors_star(obj :typing.Any) -> bool:
    """Checks if an object is an instance of 'Star' from the Mors module.
    Returns False if it fails to import Mors."""
    try:
        import Mors
    except ImportError:
        return False
    return isinstance(obj, Mors.Star)

def rebin_array(
        arr        : np.array,
        factor     : int,
        func       : typing.Callable[list[float],float] = sum
    ) -> np.array:
    """
    Rebins an array to a lower resolution by a factor,
    and returns the new array.
    Leftover values are added up on the last bin.

    Parameters
    ----------
    arr     : numpy.array, array of floats to rebin
    factor  : int, binning factor by which the number of elements are reduced.
    func    : callable (optional), function to combine a list of elements
            into a single element. The default behaviour is `sum`.

    Returns
    -------
    binned  : numpy.array, binned array
    """
    factor = int(factor)
    if factor <= 1:
        raise ValueError("binning factor must be an integer greater than 1")
    leftover = len(arr) % factor # Extra values that don't fit reshape
    leftover_ind = len(arr) - leftover
    ndata = arr[:leftover_ind].reshape((len(arr)//factor, factor))
    ndata = np.array(list(map(func, ndata)))
    # Append leftover
    if leftover > 0:
        ndata = np.append(ndata, func(arr[leftover_ind:]))
    return ndata
