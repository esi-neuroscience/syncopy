# -*- coding: utf-8 -*-
#
# Load data from SynCoPy containers
#
# Created: 2019-02-06 11:40:56
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-23 10:09:14>

# Builtin/3rd party package imports
import os
import json
import inspect
import h5py
import sys
import numpy as np
from collections import OrderedDict
from glob import glob

# Local imports
from syncopy.shared.parsers import io_parser, data_parser, filename_parser, array_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError, SPYIOError, SPYError
from syncopy.io import hash_file, FILE_EXT, startInfoDict
import syncopy.datatype as spd

__all__ = ["load"]


def load(filename, tag=None, dataclass=None, checksum=False, mode="r+", out=None):
    """
    Docstring coming soon...

    """

    # Ensure `filename` is either a valid .spy container or data file: if `filename`
    # is a directory w/o '.spy' extension, append it
    if not isinstance(filename, str):
        raise SPYTypeError(filename, varname="filename", expected="str")
    if len(os.path.splitext(os.path.abspath(os.path.expanduser(filename)))[1]) == 0:
        filename += FILE_EXT["dir"]
    try:
        fileInfo = filename_parser(filename)
    except Exception as exc:
        raise exc
    
    if tag is not None:
        if isinstance(tag, str):
            tags = [tag]
        else:
            tags = tag
        try:
            array_parser(tags, varname="tag", ntype=str)
        except Exception as exc:
            raise exc
        if fileInfo["filename"] is not None:
            raise SPYError("Only containers can be loaded with `tag` keyword!")
        for tk in range(len(tags)):
            tags[tk] = "*" + tags[tk] + "*"
    else:
        tags = "*"

    # If `dataclass` was provided, format it for our needs (e.g. 'spike' -> ['.spike'])
    if dataclass is not None:
        if isinstance(dataclass, str):
            dataclass = [dataclass]
        try:
            array_parser(dataclass, varname="dataclass", ntype=str)
        except Exception as exc:
            raise exc
        dataclass = ["." + dclass if not dclass.startswith(".") else dclass
                     for dclass in dataclass]
        extensions = set(dataclass).intersection(FILE_EXT["data"])
        if len(extensions) == 0:
                lgl = "extension(s) '" + "or '".join(ext + "' " for ext in FILE_EXT["data"])
                raise SPYValueError(legal=lgl, varname="dataclass", actual=str(dataclass))

    # Avoid any misunderstandings here...
    if not isinstance(checksum, bool):
        raise SPYTypeError(checksum, varname="checksum", expected="bool")

    # Abuse `AnalogData.mode`-setter to vet `mode`
    try:
        spd.AnalogData().mode = mode
    except Exception as exc:
        raise exc

    # If `filename` points to a spy container, `glob` what's inside, otherwise just load
    if fileInfo["filename"] is None:
        
        if dataclass is None:
            extensions = FILE_EXT["data"]
        container = os.path.join(fileInfo["folder"], fileInfo["container"])
        fileList = []
        for ext in extensions:
            for tag in tags:
                fileList.extend(glob(os.path.join(container, tag + ext)))
        if len(fileList) == 0:
            fsloc = os.path.join(container, "" + \
                                 "or ".join(tag + " " for tag in tags) + \
                                 "with extensions " + \
                                 "or ".join(ext + " " for ext in extensions))
            raise SPYIOError(fsloc, exists=False)
        if len(fileList) == 1:
            return _load(fileList[0], checksum, mode, out)
        if out is not None:
            print("syncopy.load: WARNING >>> When loading multiple objects, " +
                  "the `out` keyword is ignored. <<< ")
        objectDict = {}
        for fname in fileList:
            obj = _load(fname, checksum, mode, None)
            objectDict[os.path.basename(obj.filename)] = obj
        return objectDict

    else:

        if dataclass is not None:
            if os.path.splitext(fileInfo["filename"])[1] not in dataclass:
                lgl = "extension '" + \
                    "or '".join(dclass + "' " for dclass in dataclass)
                raise SPYValueError(legal=lgl, varname="filename",
                                    actual=fileInfo["filename"])
        return _load(filename, checksum, mode, out)


def _load(filename, checksum, mode, out):
    """
    Local helper
    """

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
        raise SPYError("Info file {} does not contain a dataclass field".format(jsonFile))

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

    # Parsing is done, create new or check provided object
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

    # First and foremost, assign dimensional information
    dimord = jsonDict.pop("dimord")
    out._dimlabels = OrderedDict(zip(dimord, [None] * len(dimord)))

    # Access data on disk (error checking is done by setters)
    out.mode = mode
    out.data = hdfFile

    # Abuse ``definetrial`` to set trial-related props
    trialdef = h5py.File(hdfFile, mode="r")["trialdefinition"][()]
    out.definetrial(trialdef)

    # Assign metadata
    for key in [prop for prop in dataclass._infoFileProperties if prop != "dimord"]:
        setattr(out, key, jsonDict[key])

    # Write `cfg` entries
    thisMethod = sys._getframe().f_code.co_name.replace("_", "")
    out.cfg = {"method": thisMethod,
               "files": [hdfFile, jsonFile]}

    # Write log-entry
    msg = "Read files v. {ver:s} ".format(ver=jsonDict["_version"])
    msg += "{hdf:s}\n\t" + (len(msg) + len(thisMethod) + 2) * " " + "{json:s}"
    out.log = msg.format(hdf=hdfFile, json=jsonFile)

    # Happy breakdown
    return out if new_out else None
