# save_spw_container.py - Save BaseData objects on disk
# 
# Created: February  5 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-14 17:18:06>

# Builtin/3rd party package imports
import os
import json
import shutil
import numpy as np
from numpy.lib.format import open_memmap  
from hashlib import blake2b

# Local imports
from spykewave.utils import spw_io_parser, spw_basedata_parser, SPWIOError, SPWTypeError
from spykewave.io import hash_file, write_access, FILE_EXT

__all__ = ["save_spw"]

##########################################################################################
def save_spw(out_name, out, fname=None, append_extension=True, memuse=100):
    """
    Docstring coming soon...
    """

    # Make sure `out_name` is a writable filesystem-location and make
    # some layout changes
    if not isinstance(out_name, str):
        raise SPWTypeError(out_name, varname="out_name", expected="str")
    if append_extension:
        out_base, out_ext = os.path.splitext(out_name)
        if out_ext != FILE_EXT["dir"]:
            out_name += FILE_EXT["dir"]
    out_name = os.path.abspath(out_name)
    if not os.path.exists(out_name):
        try:
            os.makedirs(out_name)
        except:
            raise SPWIOError(out_name)
    else:
        if not os.path.isdir(out_name):
            raise SPWIOError(out_name)
    if not write_access(out_name):
        raise SPWIOError(out_name)

    # Make sure `out` is a valid (inherited) `BaseData` object
    try:
        spw_basedata_parser(out, varname="out", writable=None, empty=False)
    except Exception as exc:
        raise exc

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

    # Use a random salt to construct a hash for differentiating file-names
    fname_hsh = blake2b(digest_size=2, salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()
    filename = os.path.join(out_name, fname + "." + fname_hsh + "{ext:s}")

    # Start by writing segment-related information
    with open(filename.format(ext=FILE_EXT["seg"]), "wb") as out_seg:
        np.save(out_seg, out.seg, allow_pickle=False)
    
    # In case `out` hosts a `VirtualData` object, things are more elaborate
    if out.data.__class__.__name__ == "VirtualData":
        
        # Given memory cap, compute how many channel blocks can be grabbed
        # per swipe (divide by 2 since we're working with an add'l tmp array)
        conv_b2mb = 1024**2
        memuse *= conv_b2mb/2
        nrow = int(memuse/(out.data.N * out.data.dtype.itemsize))
        rem = int(out.data.M % nrow)
        n_blocks = [nrow]*int(out.data.M // nrow) + [rem] * int(rem > 0)

        # Here's the fun part: the awkward looking `del` and `clear` commands
        # in the loop are crucial to prevent Python from keeping all blocks
        # of the mem-maps in memory
        dat = open_memmap(filename.format(ext=FILE_EXT["data"]), mode="w+",
                          dtype=out.data.dtype, shape=(out.data.M, out.data.N))
        del dat
        for m, M in enumerate(n_blocks):
            dat = open_memmap(filename.format(ext=FILE_EXT["data"]), mode="r+")
            dat[m*nrow : m*nrow + M:, :] = out.data[m*nrow : m*nrow + M:, :]
            del dat
            out.clear()

    # Much simpler: make sure on-disk memmap is up-to-date and simply copy it
    else:
        out.data.flush()
        shutil.copyfile(out.data.filename, filename.format(ext=FILE_EXT["data"]))
        
    # Compute checksums of created binary files
    seg_hsh = hash_file(filename.format(ext=FILE_EXT["seg"]))
    dat_hsh = hash_file(filename.format(ext=FILE_EXT["data"]))

    # Write to log already here so that the entry can be exported to json
    out.log = "Wrote files " + filename.format(ext="[dat/info/seg]")

    # Assemble dict for JSON output: extract essential `BaseData` props
    out_dct = {}
    out_dct["type"] = out.__class__.__name__
    out_dct["dimord"] = out.dimord
    out_dct["segmentlabel"] = out.segmentlabel
    out_dct["version"] = out.version
    out_dct["log"] = out._log
    out_dct["label"] = out.label

    # Convert any non-standard data-types in dicts to Python builtins
    for attr in ["cfg", "notes"]:
        if hasattr(out, attr):
            dct = getattr(out, attr)
            _dict_converter(dct)
            out_dct[attr] = dct

    # Point to actual data files (readable reference for user) - use relative
    # paths here to avoid confusions in case the parent directory is moved/copied
    out_dct["data"] = os.path.basename(filename.format(ext=FILE_EXT["data"]))
    out_dct["seg"] = os.path.basename(filename.format(ext=FILE_EXT["seg"]))

    # Take care of `BaseData` subclass properties (if present)
    if out.__class__.__name__ == "AnalogData":
        out_dct["samplerate"] = out.samplerate
        if out.hdr:
            hdr = []
            for hd in out.hdr:
                _dict_converter(hd)
                hdr.append(hd)
            out_dct["hdr"] = hdr
    
    # Save computed file-hashes
    out_dct["seg_checksum"] = seg_hsh
    out_dct["data_checksum"] = dat_hsh

    # Finally, write JSON
    with open(filename.format(ext=FILE_EXT["json"]), "w") as out_json:
        json.dump(out_dct, out_json, indent=4)
              
    return

##########################################################################################
def _dict_converter(dct, firstrun=True):
    """
    Convert all dict values having NumPy dtypes to corresponding builtin types

    Also works w/ nested dict of dicts and is cycle-save, i.e., it can
    handle self-referencing dictionaires. For instance, consider a nested dict 
    w/ back-edge (the dict is no longer an n-ary tree):

    dct = {}
    dct["a"] = {}
    dct["a"]["a.1"] = 3
    dct["b"]  = {}
    dct["b"]["b.1"] = 4000
    dct["b"]["b.2"] = dct["a"]
    dct["b"]["b.3"] = dct

    Here, b.2 points to value of `a` and b.3 is a self-reference. 

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
