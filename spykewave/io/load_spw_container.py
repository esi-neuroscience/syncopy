# load_spw_container.py - Fill BaseData object with data from disk
# 
# Created: February  6 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-12 12:56:44>

# Builtin/3rd party package imports
import os
import json
import numpy as np
from hashlib import blake2b
from numpy.lib.format import open_memmap
from glob import iglob

# Local imports
from spykewave.utils import spw_io_parser, SPWTypeError, SPWValueError
from spykewave.datatype import BaseData
from spykewave.io import hash_file, FILE_EXT

__all__ = ["load_spw"]



import ipdb


##########################################################################################
def load_spw(in_name, fname=None, out=None):
    """
    Docstring coming soon...

    in case 'dir' and 'dir.spw' exists, preference will be given to 'dir.spw'

    fname can be search pattern 'session1*' or base file-name ('asdf' will
    load 'asdf.<hash>.json/.dat.seg') or hash-id ('d4c1' will load 
    'asdf.d4c1.json/.dat/.seg')
    """

    # Make sure `in_name` is a valid filesystem-location: in case 'dir' and
    # 'dir.spw' exists, preference will be given to 'dir.spw'
    if not isinstance(in_name, str):
        raise SPWTypeError(in_name, varname="in_name", expected="str")
    _, in_ext = os.path.splitext(in_name)
    if in_ext != FILE_EXT["dir"]:
        in_spw = in_name + FILE_EXT["dir"]
    try:
        in_name = spw_io_parser(in_spw, varname="in_name", isfile=False, exists=True)
    except:
        try:
            in_name = spw_io_parser(in_name, varname="in_name", isfile=False, exists=True)
        except Exception as exc:
            raise exc

    # Prepare dictionary of relevant file-extensions
    f_ext = dict(FILE_EXT)
    f_ext.pop("dir")
    
    # Either (try to) load newest fileset or look for a specific one
    if fname is None:

        # Get most recent json file in `in_name`, default to "*.json" if not found
        in_file = max(iglob(os.path.join(in_name, "*" + FILE_EXT["json"])),
                      key=os.path.getctime, default="*.json")
        
    else:

        # Remove (if any) path as well as extension from provided file-name(-pattern)
        # and convert `fname` to search pattern if it does not already conatin wildcards
        fname = os.path.basename(fname)
        if "*" not in fname:
            fname = "*" + fname + "*"
        in_base, in_ext = os.path.splitext(fname)

        # If `fname` contains a dat/seg/json extension, we expect to find
        # exactly one match, otherwise we want to see exactly three files 
        if in_ext in f_ext.values():
            expected_count = 1
        elif in_ext == "":
            expected_count = 3
        else:
            legal = "no extension or " + "".join(ex + ", " for ex in f_ext.values())[:-2]
            raise SPWValueError(legal=legal, varname="fname", actual=fname)

        # Use `iglob` to not accidentally construct a gigantic list in pathological
        # situations (`fname = "*"`)
        in_count = 0
        for fk, fle in enumerate(iglob(os.path.join(in_name, fname))):
            in_count = fk + 1
            in_file = fle
        if in_count != expected_count:
            legal = "{exp:d} file(s), found {cnt:d}"
            raise SPWValueError(legal=legal.format(exp=expected_count, cnt=in_count),
                                varname="fname", actual=fname)

    # Construct dictionary of files to read from 
    in_base  = os.path.splitext(in_file)[0]
    in_files = {}
    for kind, ext in f_ext.items():
        in_files[kind] = in_base + ext
            
    # If provided, make sure `out` is a `BaseData` instance
    if out is not None:
        if not isinstance(out, BaseData):
            raise SPWTypeError(out, varname="out", expected="SpkeWave BaseData object")
        return_out = False
    else:
        out = BaseData()
        return_out = True

    ipdb.set_trace()

    # Start by loading contents of json file
    with open(in_files["json"], "r") as fle:
        json_dict = json.load(fle)

    # Assign manadatory attributes and subsequently remove them from `json_dict`
    if not set(MANDATORY_ATTRS).issubset(json_dict.keys()):
        legal = "mandatory fields " + "".join(attr + ", " for attr in MANDATORY_ATTRS)[:-2]
        raise SPWValueError(legal=legal, varname=in_files["json"])
    out._dimlabels["label"] = json_dict["label"]
    out._segmentlabel = json_dict["segmentlabel"]
    out._log = json_dict["log"]
    for attr in MANDATORY_ATTRS:
        json_dict.pop(attr)

    # Log entry: Loaded data:...  SpykeWave v. 0.1a -> use json_dict["version"]
        
    # # mode = "r+"
    # open_memmap(file, mode="r+")
    # out._chunks = 
