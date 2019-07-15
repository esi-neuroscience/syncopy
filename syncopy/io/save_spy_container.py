# -*- coding: utf-8 -*-
# 
# Save SynCoPy data objects on disk
# 
# Created: 2019-02-05 13:12:58
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-07-15 18:44:01>

# Builtin/3rd party package imports
import os
import json
import h5py
import sys
import numpy as np
from collections import OrderedDict
from hashlib import blake2b

# Local imports
from syncopy.shared import (io_parser, filename_parser, 
                            data_parser, scalar_parser)
from syncopy.shared.errors import (SPYIOError, SPYTypeError, 
                                   SPYValueError, SPYError)
from syncopy.io import hash_file, write_access, FILE_EXT
from syncopy import __storage__

__all__ = ["save_spy"]

def save_spy(out, filename=None, container=None, tag=None, memuse=100):
    """

    Parameters
    ----------
        out : Syncopy data object        
        filename :  str
            path to data file. Extension will be added if omitted.       
        container : str
            path to Syncopy container folder (*.spy) to be used for saving
        tag : str
            
        memuse : scalar
            
        

    Examples
    --------
    
    save(obj, filename="session1")
    # --> os.getcwd()/session1.<dataclass>
    # --> os.getcwd()/session1.<dataclass>.info
    
    save(obj, filename="/tmp/session1")
    # --> /tmp/session1.<dataclass>
    # --> /tmp/session1.<dataclass>.info
    

    # saves to current container folder under different tag
    # --> "sessionName.spy/sessionName_someOtherTag.analog"
    dataObject.save(tag='someOtherTag', container=None)

    # saves to a different container
    # --> "aDifferentSession.spy/sessionName_someOtherTag.analog"
    dataObject.save(tag='someOtherTag', container='aDifferentSession')

    """
    
    # Make sure `out` is a valid Syncopy data object
    data_parser(out, varname="out", writable=None, empty=False)
    
    if filename is None and container is None:
        raise SPYError('filename and container cannot both be None')
    
    if container is not None and filename is None:
        # construct filename from container name
        if not isinstance(container, str):
            raise SPYError(container, varname="container", expected="str")
        fileInfo = filename_parser(container)
        filename = os.path.join(fileInfo["folder"], 
                                fileInfo["container"], 
                                fileInfo["basename"])
        # handle tag                
        if tag is not None:
            filename += tag
                              
    if not isinstance(filename, str):
        raise SPYError(filename, varname="filename", expected="str")
                                    
    # add extension if not part of the filename
    if "." not in os.path.splitext(filename)[1]:
        filename += out._classname_to_extension()

    scalar_parser(memuse, varname="memuse", lims=[0, np.inf])

    # parse filename for validity
    fileInfo = filename_parser(filename)
    
    if not fileInfo["extension"] == '.' + out._classname_to_extension():
        raise SPYError("""Extension in filename ({ext}) does not match data 
                       class ({dclass})""".format(ext=fileInfo["extension"],
                                                dclass=out.__class__.__name__))
    
    dataFile = os.path.join(fileInfo["folder"], fileInfo["filename"])
    
    if not os.path.exists(fileInfo["folder"]):
        try:
            os.makedirs(dataFile)
        except IOError:
            raise SPYIOError(dataFile)
        except Exception as exc:
            raise exc
    
    # Start by creating a HDF5 container and write actual data
    h5f = h5py.File(dataFile, mode="w")
    
    # handle memory maps
    if isinstance(out.data, np.memmap) or out.data.__class__.__name__ == "VirtualData":
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
    else:
        dat = h5f.create_dataset(out.__class__.__name__, data=out.data)

    # Now write trial-related information
    trl = np.array(out.trialinfo)
    t0 = np.array(out.t0).reshape((out.t0.size, 1))
    trl = np.hstack([out.sampleinfo, t0, trl])
    trl = h5f.create_dataset("trialdefinition", data=trl)

    # Write to log already here so that the entry can be exported to json
    out.log = "Wrote files " + filename.format(ext=fileInfo["extension"] + "/info")

    # While we're at it, write cfg entries
    out.cfg = {"method": sys._getframe().f_code.co_name,
               "files": filename.format(ext=fileInfo["extension"] + "/info")}

    # Assemble dict for JSON output: order things by their "readability"
    outDict = OrderedDict()

    outDict["filename"] = fileInfo["filename"]
    outDict["data_dtype"] = dat.dtype.name
    outDict["data_shape"] = dat.shape
    outDict["data_offset"] = dat.id.get_offset()
    outDict["trl_dtype"] = trl.dtype.name
    outDict["trl_shape"] = trl.shape
    outDict["trl_offset"] = trl.id.get_offset()    
    # placeholder as HDF5 checksum differs for open vs. closed files
    outDict["file_checksum"] = None 

    for key in out._infoFileProperties:
        value = getattr(out, key)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        # potentially nested dicts
        elif isinstance(value, dict):
            value = dict(value)
            _dict_converter(value)
        outDict[key] = value
   
    # Save relevant stuff as HDF5 attributes
    for key in out._hdfFileProperties:
        if outDict[key] is None:
            h5f.attrs[key] = "None"
        else:
            try:
                h5f.attrs[key] = outDict[key]
            except RuntimeError:
                msg = "syncopy.save: WARNING >>> Too many entries in `{}` " +\
                      "- truncating HDF5 attribute. Please refer to {} for " +\
                      "complete listing. <<<"
                info_fle = os.path.split(os.path.split(filename.format(ext=FILE_EXT["info"]))[0])[1]
                info_fle = os.path.join(info_fle, os.path.basename(
                    filename.format(ext=FILE_EXT["info"])))
                print(msg.format(key, info_fle))
                h5f.attrs[key] = [outDict[key][0], "...", outDict[key][-1]]
    

    # Close the data file
    h5f.close()

    # Re-assign filename after saving (and remove source in case it came from `__storage__`)
    if __storage__ in out.filename:
        os.unlink(out.filename)
    out.data = dataFile

    # Compute checksum and finally write JSON
    outDict["file_checksum"] = hash_file(dataFile)
    
    infoFile = dataFile + FILE_EXT["info"]
    with open(infoFile, 'w') as out_json:
        json.dump(outDict, out_json, indent=4)

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
