# save_spw_container.py - Save BaseData objects on disk
# 
# Created: Februar  5 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-06 13:02:34>

# Builtin/3rd party package imports
import os
import json
import numpy as np
from hashlib import blake2b
from numpy.lib.format import open_memmap  

# Local imports
from spykewave.utils import spw_io_parser, spw_scalar_parser, SPWIOError, SPWTypeError
from spykewave.datatype import BaseData

__all__ = ["save_spw"]

# Define SpykeWave's general file-/directory-naming conventions
FILE_EXT = {"out" : "spw",
            "json" : "info",
            "data" : "dat",
            "seg" : "seg"}

# Conversion factor b/w bytes and MB
conv_b2mb = 1024**2


import ipdb
from memory_profiler import memory_usage


##########################################################################################
def save_spw(out_name, out, fname=None, memuse=None):
    """
    Docstring
    """

    print("Memory usage @ beginning: ", memory_usage())

    # Make sure `out_name` is a writable filesystem-location and format it
    if not isinstance(out_name, str):
        raise SPWTypeError(out_name, varname="out_name", expected="str")
    out_base, out_ext = os.path.splitext(out_name)
    if out_ext != FILE_EXT["out"]:
        out_name += "." + FILE_EXT["out"]
    out_name = os.path.abspath(out_name)
    if not os.path.exists(out_name):
        try:
            os.makedirs(out_name)
        except:
            raise SPWIOError(out_name)
    else:
        if not os.path.isdir(out_name):
            raise SPWIOError(out_name)

    # Make sure `out` is a `BaseData` instance
    if not isinstance(out, BaseData):
        raise SPWTypeError(out, varname="out", expected="SpkeWave BaseData object")

    # Assign default value to `fname` or ensure validity of provided file-name
    if fname is None:
        fname = os.path.splitext(os.path.basename(out_name))[0]
    else:
        try:
            spw_io_parser(fname, varname="filename", isfile=True, exists=False)
        except Exception as exc:
            raise exc
        fbase, fext = os.path.splitext(fname)
        if fext in FILE_EXT:
            fname = fbase

    if memuse is None:
        memuse = 100 * conv_b2mb

    # Use a random salt to construct a practically unique file-name hash that
    # resembles a MD5 32-char checksum
    fname_hsh = blake2b(digest_size=16, salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()
    filename = os.path.join(out_name, fname + "." + fname_hsh + "." + "{ext:s}")

    print("Memory usage before saving seg: ", memory_usage())
    
    # Start by writing segment-related information
    with open(filename.format(ext=FILE_EXT["seg"]), "wb") as out_seg:
        np.save(out_seg, out.trialinfo, allow_pickle=False)
    
    print("Memory usage after saving seg: ", memory_usage())
    
    print("Memory usage before computing hash: ", memory_usage())
    
    # # Create checksum for data
    # data_hsh = blake2b()
    # for seg in out.segments:
    #     data_hsh.update(seg)

    print("Memory usage after computing hash: ", memory_usage())

    # import hashlib
    # def md5sum(filename, blocksize=65536):
    #     hash = hashlib.md5()
    #     with open(filename, "rb") as f:
    #         for block in iter(lambda: f.read(blocksize), b""):
    #             hash.update(block)
    #     return hash.hexdigest()    
    
    print("Memory usage before determining dtype: ", memory_usage())
    
    # Write data to disk
    # Get hierarchically "highest" dtype of chunks present in `out`
    dtypes = []
    for data in out._chunks._data:
        dtypes.append(data.dtype)
    out_dtype = np.max(dtypes)

    print("Memory usage after determining dtype: ", memory_usage())
    
    print("Memory usage before allocating memmap: ", memory_usage())
    
    # Allocate target memmap (a npy file) for saving
    dat = open_memmap(filename.format(ext=FILE_EXT["data"]),
                      mode="w+",
                      dtype=out_dtype,
                      shape=out._chunks.shape)
    
    print("Memory usage after allocating memmap: ", memory_usage())
    
    # Determine block-dimensions for pumping data from memmap to memmap
    ncol = int(memuse/(out._chunks.M * out_dtype.itemsize))
    rem = int(out._chunks.N % ncol)
    n_blocks = [ncol]*int(out._chunks.N // ncol) + [rem] * int(rem > 0)

    print("Memory usage before copying memmaps: ", memory_usage())
    
    # Copy data block-wise from source to target, force write after every block
    # to not overflow memory
    # for n, N in enumerate(n_blocks):
        # dat[:, n*ncol : n*ncol + N] = out._chunks[:, n*ncol : n*ncol + N]
        # dat.flush()
    dat = out._chunks
    del dat

    print("Memory usage after copying memmaps: ", memory_usage())
    
    # Keep record of what just happened
    out.log = "Wrote files " + filename.format(ext="[dat/info/seg]")

    # Assemble dict for JSON output
    out_dct = {}
    for attr in ["hdr", "cfg", "notes"]:
        if hasattr(out, attr):
            dct = getattr(out, attr)
            _dict_converter(dct)
            out_dct[attr] = dct
    out_dct["label"] = out.label
    out_dct["segmentlabel"] = out.segmentlabel
    out_dct["log"] = out._log
    # out_dct["checksum"] = data_hsh.hexdigest()
    
    print("Memory usage after allocating json dict: ", memory_usage())

    # Finally, write JSON
    with open(filename.format(ext=FILE_EXT["json"]), "w") as out_json:
        json.dump(out_dct, out_json, indent=4)
              
    return

def _dict_converter(dct, firstrun=True):
    """
    Convert all value in dictionary to strings

    Also works w/ nested dict of dicts and is cycle-save, i.e., it can
    handle self-referencing dictionaires. For instance, consider a nested dict 
    w/ back-edge (the dict is no longer an n-ary tree):

    dct = {}
    dct["key1"] = {}
    dct["key1"]["key1.1"] = 3
    dct["key2"]  = {}
    dct["key2"]["key2.1"] = 4000
    dct["key2"]["key2.2"] = dct["key1"]
    dct["key2"]["key2.3"] = dct

    Here, key2.2 points to the dict of key1 and key2.3 is a self-reference. 

    https://stackoverflow.com/questions/10756427/loop-through-all-nested-dictionary-values
    """
    global visited
    if firstrun:
        visited = set()
    for key, value in dct.items():
        if isinstance(value, dict):
            if key not in visited:
                visited.add(key)
                _dict_converter(dct[key], firstrun=False)
        else:
            if hasattr(value, "item"):
                value = value.item()
            dct[key] = value
    return
