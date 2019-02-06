# loader.py - Manager for reading a variety of file formats
# 
# Created: January 23 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-06 16:55:02>

# Local imports
from spykewave.utils import SPWTypeError
from spykewave.io import load_binary_esi, load_spw

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

    # Depending on specified type, call appropriate reading routine
    if filetype is None or filetype in ".spw" or filetype in ["native", "spykewave"]:
        return load_spw(filename, out=out, **kwargs)
        
    elif filetype in ["esi", "esi-binary"]:
        return load_binary_esi(filename, out=out, **kwargs)
