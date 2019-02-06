# loader.py - Manager for reading a variety of file formats
# 
# Created: January 23 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-05 11:21:19>

# Builtin/3rd party package imports
import numpy as np

# Local imports
from spykewave.utils import SPWTypeError, spw_io_parser
from spykewave.io import load_binary_esi
from spykewave.datatype import BaseData

__all__ = ["load_data"]

##########################################################################################
def load_data(filename, filetype=None, out=None, **kwargs):
    """
    Docstring coming soon...
    """

    # Parsing of the actual file(s) happens later, first check `filetype` and `out`
    if filetype is not None:
        if not isinstance(filetype, str):
            raise SPWTypeError(filetype, varname="filetype", expected="str")

    # Make sure `out` is a `BaseData` instance
    if not isinstance(out, BaseData):
        raise SPWTypeError(out, varname="out", expected="SpkeWave BaseData object")

    # Convert input to list (if it is not already)
    if not isinstance(filename, (list, np.ndarray)):
        filename = [filename]
        
    # Depending on specified type, call appropriate reading routine
    if filetype is None:
        for fname in filename:
            try:
                spw_io_parser(fname, varname="filename", isfile=False, ext=".spw")
            except Exception as exc:
                raise exc
        raise NotImplementedError("Coming soon...")
        # FIXME: load_spw(filename)
        
    elif filetype == "esi":
        for fname in filename:
            try:
                spw_io_parser(fname, varname="filename", isfile=True,
                              ext=[".lfp", ".mua", ".evt", ".dpd", 
                                   ".apd", ".eye", ".pup"])
            except Exception as exc:
                raise exc
        return load_binary_esi(filename, out=out, **kwargs)
