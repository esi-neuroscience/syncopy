# -*- coding: utf-8 -*-
#
# Save SynCoPy data objects on disk
#
# Created: 2019-02-05 13:12:58
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-04-29 11:49:02>

# Builtin/3rd party package imports
import os
import json
import h5py
import sys
import numpy as np
from collections import OrderedDict
from hashlib import blake2b

# Local imports
from syncopy.utils import (io_parser, data_parser, SPYIOError,
                           SPYTypeError, SPYValueError)
from syncopy.io import hash_file, write_access, FILE_EXT
from syncopy import __storage__

__all__ = ["save_spy"]


def save_spy(out_name, out, fname=None, append_extension=True, memuse=100):
    """
    Docstring coming soon...
    """

    # Make sure `out_name` is a writable filesystem-location and make
    # some layout changes
    if not isinstance(out_name, str):
        raise SPYTypeError(out_name, varname="out_name", expected="str")
    if append_extension:
        out_base, out_ext = os.path.splitext(out_name)
        if out_ext != FILE_EXT["dir"]:
            out_name += FILE_EXT["dir"]
    out_name = os.path.abspath(os.path.expanduser(out_name))
    if not os.path.exists(out_name):
        try:
            os.makedirs(out_name)
        except:
            raise SPYIOError(out_name)
    else:
        if not os.path.isdir(out_name):
            raise SPYIOError(out_name)
    if not write_access(out_name):
        raise SPYIOError(out_name)

    # Make sure `out` is a valid SyNCoPy data object
    try:
        data_parser(out, varname="out", writable=None, empty=False)
    except Exception as exc:
        raise exc
    if getattr(out, "samplerate", 1) is None:
        lgl = "SyNCoPy object with well-defined samplerate"
        act = "None"
        raise SPYValueError(legal=lgl, actual=act, varname="samplerate")

    # Assign default value to `fname` or ensure validity of provided file-name
    if fname is None:
        fname = os.path.splitext(os.path.basename(out_name))[0]
    else:
        try:
            io_parser(fname, varname="filename", isfile=True, exists=False)
        except Exception as exc:
            raise exc
        fbase, fext = os.path.splitext(fname)
        if fext in FILE_EXT:
            fname = fbase

    # Use a random salt to construct a hash for differentiating file-names
    fname_hsh = blake2b(digest_size=2, salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()
    filename = os.path.join(out_name, fname + "." + fname_hsh + "{ext:s}")

    # Start by creating a HDF5 container and write actual data
    h5f = h5py.File(filename.format(ext=FILE_EXT["data"]), mode="w")

    # The most generic case: `out` hosts a `h5py.Dataset` object
    if isinstance(out.data, h5py.Dataset):
        dat = h5f.create_dataset(out.__class__.__name__, data=out.data)

    # In case `out` hosts a memory-map-like object, things are more elaborate
    elif isinstance(out.data, np.memmap) or out.data.__class__.__name__ == "VirtualData":

        # Given memory cap, compute how many data blocks can be grabbed
        # per swipe (divide by 2 since we're working with an add'l tmp array)
        memuse *= 1024**2 / 2
        nrow = int(memuse / (np.prod(out.data.shape[1:]) * out.data.dtype.itemsize))
        rem = int(out.data.shape[0] % nrow)
        n_blocks = [nrow] * int(out.data.shape[0] // nrow) + [rem] * int(rem > 0)

        # Write data block-wise to dataset (use `clear` to wipe blocks of
        # mem-maps from memory)
        dat = h5f.create_dataset(out.__class__.__name__,
                                 dtype=out.data.dtype, shape=out.data.shape)
        for m, M in enumerate(n_blocks):
            dat[m * nrow: m * nrow + M, :] = out.data[m * nrow: m * nrow + M, :]
            out.clear()

    # The simplest case: `out` hosts a NumPy array in its `data` property
    else:
        dat = h5f.create_dataset(out.__class__.__name__, data=out.data)

    # Now write trial-related information
    trl = np.array(out.trialinfo)
    t0 = np.array(out.t0).reshape((out.t0.size, 1))
    trl = np.hstack([out.sampleinfo, t0, trl])
    trl = h5f.create_dataset("trialdefinition", data=trl)

    # Write to log already here so that the entry can be exported to json
    out.log = "Wrote files " + filename.format(ext="[dat/info/trl]")

    # While we're at it, write cfg entries
    out.cfg = {"method": sys._getframe().f_code.co_name,
               "files": filename.format(ext="[dat/info]")}

    # Assemble dict for JSON output: order things by their "readability"
    out_dct = OrderedDict()
    out_dct["type"] = out.__class__.__name__
    out_dct["dimord"] = out.dimord
    out_dct["version"] = out.version

    # Point to actual data file (readable reference for user) - use relative
    # path here to avoid confusion in case the parent directory is moved/copied
    out_dct["data"] = os.path.basename(filename.format(ext=FILE_EXT["data"]))
    out_dct["data_dtype"] = dat.dtype.name
    out_dct["data_shape"] = dat.shape
    out_dct["data_offset"] = dat.id.get_offset()
    out_dct["trl_dtype"] = trl.dtype.name
    out_dct["trl_shape"] = trl.shape
    out_dct["trl_offset"] = trl.id.get_offset()

    # Continue w/ scalar-valued props
    if hasattr(out, "samplerate"):
        out_dct["samplerate"] = out.samplerate

    # Computed file-hashes (placeholder)
    out_dct["data_checksum"] = None

    # Object history
    out_dct["log"] = out._log

    # Stuff that is potentially vector-valued
    if getattr(out, "hdr", None) is not None:
        hdr = []
        for hd in out.hdr:
            _dict_converter(hd)
            hdr.append(hd)
        out_dct["hdr"] = hdr
    if hasattr(out, "taper"):
        out_dct["taper"] = out.taper.tolist()   # where is a taper,
        out_dct["freq"] = out.freq.tolist()     # there is a frequency

    # Stuff that is definitely vector-valued
    if hasattr(out, "channel"):
        out_dct["channel"] = out.channel.tolist()
    if hasattr(out, "unit"):
        out_dct["unit"] = out.unit.tolist()

    # Here for some nested dicts and potentially long-winded notes
    if out.cfg is not None:
        cfg = dict(out.cfg)
        _dict_converter(cfg)
        out_dct["cfg"] = cfg
    else:
        out_dct["cfg"] = None
    if hasattr(out, "notes"):
        notes = out.notes
        _dict_converter(notes)
        out_dct["notes"] = notes

    # Save relevant stuff as HDF5 attributes
    noh5 = ["data", "data_dtype", "data_shape", "data_offset", "data_checkusm",
            "trl_dtype", "trl_shape", "trl_offset", "hdr", "cfg", "notes"]
    for key in set(out_dct.keys()).difference(noh5):
        if out_dct[key] is None:
            h5f.attrs[key] = "None"
        else:
            try:
                h5f.attrs[key] = out_dct[key]
            except RuntimeError:
                msg = "SyNCoPy save_spy: WARNING >>> Too many entries in `{}` " +\
                      "- truncating HDF5 attribute. Please refer to {} for " +\
                      "complete listing. <<<"
                info_fle = os.path.split(os.path.split(filename.format(ext=FILE_EXT["json"]))[0])[1]
                info_fle = os.path.join(info_fle, os.path.basename(
                    filename.format(ext=FILE_EXT["json"])))
                print(msg.format(key, info_fle))
                h5f.attrs[key] = [out_dct[key][0], "...", out_dct[key][-1]]

    # Close container and compute its checksum
    h5f.close()
    out_dct["data_checkusm"] = hash_file(filename.format(ext=FILE_EXT["data"]))

    # Finally, write JSON
    with open(filename.format(ext=FILE_EXT["json"]), "w") as out_json:
        json.dump(out_dct, out_json, indent=4)

    # Last but definitely not least: if source data came from HDF dataset,
    # re-assign filename after saving (and remove source in case it came
    # from `__storage__`)
    if out._filename is not None:
        if __storage__ in out._filename:
            os.unlink(out._filename)
        out.data = filename.format(ext=FILE_EXT["data"])

    return


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
        elif isinstance(value, list):
            if key not in visited:
                visited.add(key)
                for el in value:
                    if isinstance(el, dict):
                        _dict_converter(el, firstrun=False)
        else:
            if hasattr(value, "item"):
                value = value.item()
            dct[key] = value
    return
