# -*- coding: utf-8 -*-
#
# Auxiliaries used across all of Syncopy
#

# Builtin/3rd party package imports
import numpy as np
import inspect
import json
import subprocess

# Local imports
from syncopy.shared.errors import SPYValueError, SPYWarning, SPYTypeError

__all__ = ["StructDict", "get_defaults"]


class StructDict(dict):
    """Child-class of dict for emulating MATLAB structs

    Examples
    --------
    cfg = StructDict()
    cfg.a = [0, 25]

    """

    def __init__(self, *args, **kwargs):
        """
        Create a child-class of dict whose attributes are its keys
        (thus ensuring that attributes and items are always in sync)
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.keys():
            ppStr = "Syncopy StructDict\n\n"
            maxKeyLength = max([len(val) for val in self.keys()])
            printString = "{0:>" + str(maxKeyLength + 5) + "} : {1:}\n"
            for key, value in self.items():
                ppStr += printString.format(key, str(value))
            ppStr += "\nUse `dict(cfg)` for copy-paste-friendly format"
        else:
            ppStr = "{}"
        return ppStr


class SerializableDict(dict):

    """
    It's a dict which checks newly inserted
    values for serializability, keys should always be serializable
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # check also initial entries
        for key, value in self.items():
            self.is_json(key, value)

    def __setitem__(self, key, value):
        self.is_json(key, value)
        dict.__setitem__(self, key, value)

    def is_json(self, key, value):
        try:
            json.dumps(value)
        except TypeError:
            lgl = "serializable data type, e.g. floats, lists, tuples, ... "
            raise SPYTypeError(value, f"value for key '{key}'", lgl)
        try:
            json.dumps(key)
        except TypeError:
            lgl = "serializable data type, e.g. floats, lists, tuples, ... "
            raise SPYTypeError(value, f"key '{key}'", lgl)


def get_frontend_cfg(defaults, lcls, kwargs):

    """
    Assemble cfg dict to allow direct replay of frontend calls

    Parameters
    ----------

    defaults : dict
        The result of :func:`~get_defaults`, holding all frontend specific
        parameter names and default values
    lcls : dict
        The `locals()` within a frontend call, contains passed
        parameter names and values
    kwargs : dict
        The `kwargs` attached to every frontend signature, holding
        additional arguments, e.g. `parallel` and `select`

    Returns
    -------
    new_cfg : :class:`~StructDict`
        Holds all (default and non-default) parameter key-value
        pairs passed to the frontend

    """

    # create new cfg dict
    new_cfg = StructDict()
    for par_name in defaults:
        # check only needed for injected kwargs like `parallel`
        if par_name in lcls:
            new_cfg[par_name] = lcls[par_name]
    # attach additional kwargs (like select)
    for key in kwargs:
        new_cfg[key] = kwargs[key]

    return new_cfg


def best_match(source, selection, span=False, tol=None, squash_duplicates=False):
    """
    Find matching elements in a given 1d-array/list

    Parameters
    ----------
    source : NumPy 1d-array/list
        Reference array whose elements are to be matched by `selection`
    selection: NumPy 1d-array/list
        Array of query-values whose closest matches are to be found in `source`.
        Note that `source` and `selection` need not be the same length.
    span : bool
        If `True`, `selection` is interpreted as (closed) interval ``[lo, hi]`` and
        `source` is queried for all elements contained in the interval, i.e.,
        ``lo <= src <= hi for src in source`` (typically used for
        `toilim`/`foilim`-like selections).
    tol : None or float
        If `None` for each component of `selection` the closest value in `source`
        is selected, e.g., for ``source = [10, 20]`` and ``selection = [-50, 0, 50]``
        the closest values are `[10, 10, 20]`.
        If not `None`, ensures values in `selection` do not deviate further
        than `tol` from `source`. If any element `sel` of `selection` is outside
        a `tol`-neighborhood around `source`, i.e.,
        ``np.abs(sel - source).max() >= tol``,
        a :class:`~syncopy.shared.errors.SPYValueError` is raised.
    squash_duplicates : bool
        If `True`, identical matches are removed from the result.

    Returns
    -------
    values : NumPy 1darray
        Values of `source` that most closely match given elements in `selection`
    idx : NumPy 1darray
        Indices of `values` with respect to `source`, such that,
        ``source[idx] == values``

    Notes
    -----
    This is an auxiliary method that is intended purely for internal use. Thus,
    no error checking is performed.

    Examples
    --------
    Exact matching, ordered `source` and `selection`:

    >>> best_match(np.arange(10), [2,5])
    (array([2, 5]), array([2, 5]))

    Inexact matching, ordered `source` and `selection`:

    >>> source = np.arange(10)
    >>> selection = np.array([1.5, 1.5, 2.2, 6.2, 8.8])
    >>> best_match(source, selection)
    (array([2, 2, 2, 6, 9]), array([2, 2, 2, 6, 9]))

    Inexact matching, unordered `source` and `selection`:

    >>> source = np.array([2.2, 1.5, 1.5, 6.2, 8.8])
    >>> selection = np.array([1.9, 9., 1., -0.4, 1.2, 0.2, 9.3])
    >>> best_match(source, selection)
    (array([2.2, 8.8, 1.5, 1.5, 1.5, 1.5, 8.8]), array([0, 4, 1, 1, 1, 1, 4]))

    Same as above, but ignore duplicate matches

    >>> best_match(source, selection, squash_duplicates=True)
    (array([2.2, 8.8, 1.5]), array([0, 4, 1]))

    Interval-matching:

    >>> best_match(np.arange(10), [2.9, 6.1], span=True)
    (array([3, 4, 5, 6]), array([3, 4, 5, 6]))
    """

    # Make `source` a NumPy array if necessary
    if isinstance(source, list):
        source = np.array(source)

    # If `selection` is a scalar, convert it to 1-element list
    if np.issubdtype(type(selection), np.number):
        selection = [selection]

    # Ensure selection is within `tol` bounds from `source`
    if tol is not None:
        if not np.all([np.all((np.abs(source - value)) < tol) for value in selection]):
            lgl = "all elements of `selection` to be within a {0:2.4f}-band around `source`"
            act = "values in `selection` deviating further than given tolerance " +\
                "of {0:2.4f} from source"
            raise SPYValueError(legal=lgl.format(tol),
                                varname="selection",
                                actual=act.format(tol))

    # Do not perform O(n) potentially unnecessary sort operations...
    issorted = True

    # Interval-selections are a lot easier than discrete time-points...
    if span:
        idx = np.intersect1d(np.where(source >= selection[0])[0],
                             np.where(source <= selection[1])[0])
    else:
        issorted = True
        if source.size > 1 and np.diff(source).min() < 0:
            issorted = False
            orig = np.array(source, copy=True)
            idx_orig = np.argsort(orig)
            source = orig[idx_orig]
        idx = np.searchsorted(source, selection, side="left")
        leftNbrs = np.abs(selection - source[np.maximum(idx - 1, np.zeros(idx.shape, dtype=np.intp))])
        rightNbrs = np.abs(selection - source[np.minimum(idx, np.full(idx.shape, source.size - 1, dtype=np.intp))])
        shiftLeft = ((idx == source.size) | (leftNbrs < rightNbrs))
        idx[shiftLeft] -= 1

    # Account for potentially unsorted selections (and thus unordered `idx`)
    if squash_duplicates:
        _, xdi = np.unique(idx.astype(np.intp), return_index=True)
        idx = idx[np.sort(xdi)]

    # Re-order discrete-selection index arrays in case `source` was unsorted
    if not issorted and not span:
        idx_sort = idx_orig[idx]
        return orig[idx_sort], idx_sort
    else:
        return source[idx], idx


def get_defaults(obj):
    """
    Parse input arguments of `obj` and return dictionary

    Parameters
    ----------
    obj : function or class
        Object whose input arguments to parse. Can be either a class or
        function.

    Returns
    -------
    argdict : dictionary
        Dictionary of `argument : default value` pairs constructed from
        `obj`'s call-signature/instantiation.

    Examples
    --------
    To see the default input arguments of :meth:`syncopy.freqanalysis` use

    >>> spy.get_defaults(spy.freqanalysis)
    """

    if not callable(obj):
        raise SPYTypeError(obj, varname="obj", expected="SyNCoPy function or class")
    dct = {k: v.default for k, v in inspect.signature(obj).parameters.items()\
           if v.default != v.empty and v.name != "cfg"}
    return StructDict(dct)


    
