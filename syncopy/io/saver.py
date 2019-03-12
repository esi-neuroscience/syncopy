# -*- coding: utf-8 -*-
#
# Manager for writing various file formats
# 
# Created: 2019-02-05 12:55:36
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-08 13:34:34>

# Local imports
from syncopy.utils import SPYTypeError
from syncopy.io import save_spy

__all__ = ["save_data"]

##########################################################################################
def save_data(out_name, out, filetype=None, **kwargs):
    """
    Docstring coming soon...
    """

    # Parsing of `out_name` and `out` happens in the actual writing routines,
    # only check `filetype` in here
    if filetype is not None:
        if not isinstance(filetype, str):
            raise SPYTypeError(filetype, varname="filetype", expected="str")

    # Depending on specified output file-type, call appropriate writing routine
    if filetype is None or filetype in ".spw" or filetype in ["native", "syncopy"]:
        save_spy(out_name, out, **kwargs)
    elif filetype == "matlab" or filetype in ".mat":
        raise NotImplementedError("Coming soon...")
