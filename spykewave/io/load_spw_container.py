# load_spw_container.py - Fill BaseData object with data from disk
# 
# Created: February  6 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-06 17:51:32>

# Builtin/3rd party package imports
import os
import json
import numpy as np
from hashlib import blake2b
from numpy.lib.format import open_memmap
from glob import iglob

# Local imports
from spykewave.utils import spw_io_parser, SPWIOError, SPWTypeError
from spykewave.datatype import BaseData
from spykewave.io import hash_file, FILE_EXT

__all__ = ["load_spw"]

##########################################################################################
def load_spw(in_name, fname=None, append_extension=True, out=None):
    """
    Docstring coming soon...

    in case 'dir' and 'dir.spw' exists, preference will be given to 'dir.spw'
    """

    # Make sure `in_name` is a valid filesystem-location
    try:
        in_name = spw_io_parser(in_name, varname="in_name", isfile=False, exists=True)
    except Exception as exc:
        raise exc

    # Either (try to) load newest fileset or look for a specific one
    if fname is None:
        in_json = max(iglob(os.path.join(in_name, "*." + FILE_EXT["json"])),
                      key=os.path.getctime, default="")
        if len(fname) == 0:
            raise SPWIOError(...)
        in_base = os.path.splitext(in_json)[0]
        in_seg = in_base + "." + FILE_EXT["seg"]
        in_dat = in_base + "." + FILE_EXT["data"]
        

    # If provided, make sure `out` is a `BaseData` instance
    if out is not None:
        if not isinstance(out, BaseData):
            raise SPWTypeError(out, varname="out", expected="SpkeWave BaseData object")
        return_out = False
    else:
        out = BaseData()
        return_out = True
        
    # mode = "r+"
    open_memmap(file, mode="r+")
    out._chunks = 
