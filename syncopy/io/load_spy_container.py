# -*- coding: utf-8 -*-
#
# Load data from SynCoPy containers
#
# Created: 2019-02-06 11:40:56
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-06-27 15:42:02>

# Builtin/3rd party package imports
import os
import json
import inspect
import h5py
import sys
import numpy as np
from collections import OrderedDict
from glob import iglob

# Local imports
from syncopy.shared import io_parser, json_parser, data_parser, filename_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError, SPYError
from syncopy.io import hash_file, FILE_EXT, startInfoDict
import syncopy.datatype as spd


__all__ = ["load_spy"]


def load_spy(filename, fname=None, checksum=False, out=None, **kwargs):
    """
    Docstring coming soon...

    """

    if not isinstance(filename, str):
        raise SPYTypeError(filename, varname="in_name", expected="str")
    
    fileInfo = filename_parser(filename)
    
    hdfFile = os.path.join(fileInfo["folder"], fileInfo["filename"])
    jsonFile = hdfFile + FILE_EXT["info"]
    
    try: 
        _ = io_parser(hdfFile, varname="hdfFile", isfile=True, exists=True)
        _ = io_parser(jsonFile, varname="jsonFile", isfile=True, exists=True)
    except Exception as exc:
        raise exc
    
    with open(jsonFile, "r") as file:
        jsonDict = json.load(file)

    if not "dataclass" in jsonDict.keys():
        raise SPYError("Info file {file} does not contain a dataclass field".format(jsonFile))
    
    if hasattr(spd, jsonDict["dataclass"]):
        dataclass = getattr(spd, jsonDict["dataclass"])
    else:
        raise SPYError("Unknown data class {class}".format(jsonDict["dataclass"]))
        
    requiredFields = tuple(startInfoDict.keys()) + dataclass._infoFileProperties

    for key in requiredFields:
        if key not in jsonDict.keys():
            raise SPYError("Required field {field} for {cls} not in {file}"
                           .format(field=key, 
                                   cls=dataclass.__name__,
                                   file=jsonFile))                       
    
    # FIXME: add version comparison (syncopy.__version__ vs jsonDict["_version"])

    # If wanted, perform checksum matching
    if checksum:
        hsh_msg = "hash = {hsh:s}"
        hsh = hash_file(hdfFile)
        if hsh != jsonDict["file_checksum"]:
            raise SPYValueError(legal=hsh_msg.format(hsh=jsonDict["file_checksum"]),
                                varname=os.path.basename(hdfFile),
                                actual=hsh_msg.format(hsh=hsh))

    # Parsing is done, create new or check provided container
    if out is not None:
        try:
            data_parser(out, varname="out", writable=True,
                        dimord=jsonDict["dimord"], dataclass=jsonDict["dataclass"])
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = dataclass()
        new_out = True

    # Access data on disk
    out.data = hdfFile
    
    # Abuse ``definetrial`` to set trial-related props
    trialdef = h5py.File(hdfFile, mode="r")["trialdefinition"][()]
    out.definetrial(trialdef)

    # Assign metadata 
    for key in dataclass._infoFileProperties:
        if key == "dimord":
             out._dimlabels = OrderedDict(zip(jsonDict["dimord"], [None] 
                                              * len(jsonDict["dimord"])))
             continue
        setattr(out, key, jsonDict[key])
            
    # Write `cfg` entries
    out.cfg = {"method": sys._getframe().f_code.co_name,
               "files": hdfFile}

    # Write log-entry
    msg = "Read files v. {ver:s} {fname:s}"
    out.log = msg.format(ver=jsonDict["_version"], fname=hdfFile)

    # Happy breakdown
    return out if new_out else None
