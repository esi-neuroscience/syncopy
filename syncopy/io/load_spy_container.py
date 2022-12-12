# -*- coding: utf-8 -*-
#
# Load data from Syncopy containers
#

# Builtin/3rd party package imports
import os
import json
import h5py
import sys
import numpy as np
from glob import glob

# Local imports
from syncopy.shared.filetypes import FILE_EXT
from syncopy.shared.parsers import io_parser, data_parser, filename_parser, array_parser
from syncopy.shared.errors import (SPYTypeError, SPYValueError, SPYIOError,
                                   SPYError, SPYWarning)
from syncopy.io.utils import hash_file, startInfoDict

import syncopy.datatype as spd

# to allow loading older spy containers
legacy_not_required = ['info']

__all__ = ["load"]


def load(filename, tag=None, dataclass=None, checksum=False, mode="r+", out=None):
    """
    Load Syncopy data object(s) from disk

    Either loads single files within or outside of '.spy'-containers or loads
    multiple objects from a single '.spy'-container. Loading from containers can
    be further controlled by imposing restrictions on object class(es) (via
    `dataclass`) and file-name tag(s) (via `tag`).

    Parameters
    ----------
    filename : str
        Either path to Syncopy container folder (`*.spy`, if omitted, the extension
        '.spy' will be appended) or name of data or metadata file. If `filename`
        points to a container and no further specifications are provided, the
        entire contents of the container is loaded. Otherwise, specific objects
        may be selected using the `dataclass` or `tag` keywords (see below).
    tag : None or str or list
        If `filename` points to a container, `tag` may be used to filter objects
        by filename-`tag`. Multiple tags can be provided using a list, e.g.,
        ``tag = ['experiment1', 'experiment2']``. Can be combined with `dataclass`
        (see below). Invalid if `filename` points to a single file.
    dataclass : None or str or list
        If provided, only objects of provided dataclass are loaded from disk.
        Available options are '.analog', '.spectral', .spike' and '.event'
        (as listed in  ``spy.FILE_EXT["data"]``). Multiple class specifications
        can be provided using a list, e.g., ``dataclass = ['.analog', '.spike']``.
        Can be combined with `tag` (see above) and is also valid if `filename`
        points to a single file (e.g., to ensure loaded object is of a specific
        type).
    checksum : bool
        If `True`, checksum-matching is performed on loaded object(s) to ensure
        data-integrity (impairs performance particularly when loading large files).
    mode : str
        Data access mode of loaded objects (can be 'r' for read-only, 'r+' or 'w'
        for read/write access).
    out : Syncopy data object
        Empty object to be filled with data loaded from disk. Has to match the
        type of the on-disk file (e.g., ``filename = 'mydata.analog'`` requires
        `out` to be a :class:`syncopy.AnalogData` object). Can only be used
        when loading single objects from disk (`out` is ignored when multiple
        files are loaded from a container).

    Returns
    -------
    Nothing : None
        If a single file is loaded and `out` was provided, `out` is filled with
        data loaded from disk, i.e., :func:`syncopy.load` does **not** create a
        new object
    obj : Syncopy data object
        If a single file is loaded and `out` was `None`, :func:`syncopy.load`
        returns a new object.
    objdict : dict
        If multiple files are loaded, :func:`syncopy.load` creates a new object
        for each file and places them in a dictionary whose keys are the base-names
        (sans path) of the corresponding files.

    Notes
    -----
    All of Syncopy's classes offer (limited) support for data loading upon object
    creation. Just as the class method ``.save`` can be used as a shortcut for
    :func:`syncopy.save`, Syncopy objects can be created from Syncopy data-files
    upon creation, e.g.,

    >>> adata = spy.AnalogData('/path/to/session1.analog')

    creates a new :class:`syncopy.AnalogData` object and immediately fills it
    with data loaded from the file "/path/to/session1.analog".

    Since only one object can be created at a time, this loading shortcut only
    supports single file specifications (i.e., ``spy.AnalogData("container.spy")``
    is invalid).

    Examples
    --------
    Load all objects found in the spy-container "sessionName" (the extension ".spy"
    may or may not be provided)

    >>> objectDict = spy.load("sessionName")
    >>> # --> returns a dict with base-filenames as keys

    Load all :class:`syncopy.AnalogData` and :class:`syncopy.SpectralData` objects
    from the spy-container "sessionName"

    >>> objectDict = spy.load("sessionName.spy", dataclass=['analog', 'spectral'])

    Load a specific :class:`syncopy.AnalogData` object from the above spy-container

    >>> obj = spy.load("sessionName.spy/sessionName_someTag.analog")

    This is equivalent to

    >>> obj = spy.AnalogData("sessionName.spy/sessionName_someTag.analog")

    If the "sessionName" spy-container only contains one object with the tag
    "someTag", the above call is equivalent to

    >>> obj = spy.load("sessionName.spy", tag="someTag")

    If there are multiple objects of different types using the same tag "someTag",
    the above call can be further narrowed down to only load the requested
    :class:`syncopy.AnalogData` object

    >>> obj = spy.load("sessionName.spy", tag="someTag", dataclass="analog")

    See also
    --------
    syncopy.save : save syncopy object on disk
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

    # Abuse `AnalogData.mode`-setter to check `mode`
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
            msg = "When loading multiple objects, the `out` keyword is ignored"
            SPYWarning(msg)
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

    if "dataclass" not in jsonDict.keys():
        raise SPYError("Info file {} does not contain a dataclass field".format(jsonFile))

    if hasattr(spd, jsonDict["dataclass"]):
        dataclass = getattr(spd, jsonDict["dataclass"])
    else:
        raise SPYError("Unknown data class {class}".format(jsonDict["dataclass"]))

    requiredFields = tuple(startInfoDict.keys()) + dataclass._infoFileProperties

    for key in requiredFields:
        if key not in jsonDict.keys() and key not in legacy_not_required:
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
    dimord = jsonDict.pop("dimord")
    if out is not None:
        try:
            data_parser(out, varname="out", writable=True, dataclass=jsonDict["dataclass"])
        except Exception as exc:
            raise exc
        new_out = False
        out.dimord = dimord
    else:
        out = dataclass(dimord=dimord)
        new_out = True

    # Access data on disk (error checking is done by setters)
    out.mode = mode

    # If the JSON contains `_hdfFileDatasetProperties`, load all datasets listed in there. Otherwise, load the ones
    # already defined by `out._hdfFileDatasetProperties` and defined in the respective data class.
    # This is needed to load both new files with, and legacy files without the `_hdfFileDatasetProperties` in the JSON.
    json_hdfFileDatasetProperties = jsonDict.pop("_hdfFileDatasetProperties", None) # They may not be in there for legacy files, so allow None.
    if json_hdfFileDatasetProperties is not None:
        out._hdfFileDatasetProperties = tuple(json_hdfFileDatasetProperties) # It's a list in the JSON, so convert to tuple.
    for datasetProperty in out._hdfFileDatasetProperties:
        targetProperty = datasetProperty if datasetProperty == "data" else "_" + datasetProperty
        setattr(out, targetProperty, h5py.File(hdfFile, mode="r")[datasetProperty])

    # Abuse ``definetrial`` to set trial-related props
    trialdef = h5py.File(hdfFile, mode="r")["trialdefinition"][()]
    out.definetrial(trialdef)

    # Assign metadata
    for key in [prop for prop in dataclass._infoFileProperties if
                prop != "dimord" and prop in jsonDict.keys()]:
        setattr(out, key, jsonDict[key])

    thisMethod = sys._getframe().f_code.co_name.replace("_", "")

    # Write log-entry
    msg = "Read files v. {ver:s} ".format(ver=jsonDict["_version"])
    msg += "{hdf:s}\n\t" + (len(msg) + len(thisMethod) + 2) * " " + "{json:s}"
    out.log = msg.format(hdf=hdfFile, json=jsonFile)

    # Happy breakdown
    return out if new_out else None
