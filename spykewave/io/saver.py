# saver.py - Manager for writing various file formats
# 
# Created: February  5 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-06 16:28:32>

# Local imports
from spykewave.utils import SPWTypeError
from spykewave.io import save_spw

__all__ = ["save_data"]

##########################################################################################
def save_data(fname, out, filetype=None, **kwargs):
    """
    Docstring coming soon...
    """

    # Parsing of `fname` and `out` happens in the actual writing routines,
    # only check `filetype` in here
    if filetype is not None:
        if not isinstance(filetype, str):
            raise SPWTypeError(filetype, varname="filetype", expected="str")

    # Depending on specified output file-type, call appropriate writing routine
    if filetype is None or filetype in ".spw" or filetype in ["native", "spykewave"]:
        save_spw(fname, out, **kwargs)
    elif filetype == "matlab" or filetype in ".mat":
        raise NotImplementedError("Coming soon...")
