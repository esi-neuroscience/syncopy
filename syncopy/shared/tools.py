# -*- coding: utf-8 -*-
# 
# Auxiliaries used across all of Syncopy
# 
# Created: 2020-01-27 13:37:32
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-03-10 16:24:28>

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.shared.errors import SPYValueError

__all__ = ["StructDict", "best_match"]


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


def best_match(source, selection, span=False, tol=None, squash_duplicates=False):
    """
    Coming soon...
    
    span -> allow `toilim`/`foilim` like selections
    
    tol ->  if provided, ensures values in selection do not deviate further 
            than `tol` from source, e.g.,
            
    Error checking is *not* performed!!!
    
    return values, idx
    """
    
    # Ensure selection is within `tol` bounds from `source`
    if tol is not None:
        if not np.all([np.all((np.abs(source - value)) < tol) for value in selection]):
            lgl = "all elements of `selection` to be within a {0:2.4f}-band around `source`"
            act = "values in `selection` deviating further than given tolerance " +\
                "of {0:2.4f} from source"
            raise SPYValueError(legal=lgl.format(tol), 
                                varname="selection", 
                                actual=act.format(tol))

    # Do not perform O(n) potentially unnecessary sort operations
    issorted = True
    if np.diff(source).min() < 0:
        issorted = False
        orig = source.copy()
        idx_orig = np.argsort(orig)
        source = orig[idx_orig]

    if span:
        idx = np.intersect1d(np.where(source >= selection[0])[0], 
                             np.where(source <= selection[1])[0])
    else:
        idx = np.searchsorted(source, selection, side="left") 
        leftNbrs = np.abs(selection - source[np.maximum(idx - 1, np.zeros(idx.shape, dtype=np.intp))])
        rightNbrs = np.abs(selection - source[np.minimum(idx, np.full(idx.shape, source.size - 1, dtype=np.intp))])
        shiftLeft = ((idx == source.size) | (leftNbrs < rightNbrs))
        idx[shiftLeft] -= 1

    # Account for potentially unsorted selections (and thus unordered `idx`)
    if squash_duplicates: 
        _, xdi = np.unique(idx.astype(np.intp), return_index=True)
        idx = idx[np.sort(xdi)]

    # Re-order index arrays in case `source` was unsorted
    if not issorted:
        idx_sort = idx_orig[idx]
        return orig[idx_sort], idx_sort
    else:
        return source[idx], idx
