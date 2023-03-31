# -*- coding: utf-8 -*-
#
# Save Syncopy data objects to disk
#

# Builtin/3rd party package imports
import os
import json
import h5py
import numpy as np
from collections import OrderedDict
import syncopy as spy

# Local imports
from syncopy.shared.filetypes import FILE_EXT
from syncopy.shared.parsers import filename_parser, data_parser
from syncopy.shared.errors import SPYIOError, SPYTypeError, SPYError, SPYWarning
from syncopy.io.utils import hash_file, startInfoDict
from syncopy import __storage__

__all__ = ["save"]


def save(out, container=None, tag=None, filename=None, overwrite=False):
    r"""Save Syncopy data object to disk

    The underlying array data object is stored in a HDF5 file, the metadata in
    a JSON file. Both can be placed inside a Syncopy container, which is a
    regular directory with the extension '.spy'.

    Parameters
    ----------
    out : Syncopy data object
        Object to be stored on disk.
    container : str
        Path to Syncopy container folder (\*.spy) to be used for saving. If
        omitted, the extension '.spy' will be added to the folder name.
    tag : str
        Tag to be appended to container basename
    filename :  str
        Explicit path to data file. This is only necessary if the data should
        not be part of a container folder. An extension (\*.<dataclass>) is
        added if omitted. The `tag` argument is ignored.
    overwrite : bool
        If `True` an existing HDF5 file and its accompanying JSON file is
        overwritten (without prompt).

    Returns
    -------
    Nothing : None

    Notes
    ------
    Syncopy objects may also be saved using the class method ``.save`` that
    acts as a wrapper for :func:`syncopy.save`, e.g.,

    >>> save(obj, container="new_spy_container")

    is equivalent to

    >>> obj.save(container="new_spy_container")

    However, once a Syncopy object has been saved, the class method ``.save``
    can be used as a shortcut to quick-save recent changes, e.g.,

    >>> obj.save()

    writes the current state of `obj` to the data/meta-data files on-disk
    associated with `obj` (overwriting both in the process). Similarly,

    >>> obj.save(tag='newtag')

    saves `obj` in the current container 'new_spy_container' under a different
    tag.

    Examples
    --------
    Save the Syncopy data object `obj` on disk in the current working directory
    without creating a spy-container

    >>> spy.save(obj, filename="session1")
    >>> # --> os.getcwd()/session1.<dataclass>
    >>> # --> os.getcwd()/session1.<dataclass>.info

    Save `obj` without creating a spy-container using an absolute path

    >>> spy.save(obj, filename="/tmp/session1")
    >>> # --> /tmp/session1.<dataclass>
    >>> # --> /tmp/session1.<dataclass>.info

    Save `obj` in a new spy-container created in the current working directory

    >>> spy.save(obj, container="container.spy")
    >>> # --> os.getcwd()/container.spy/container.<dataclass>
    >>> # --> os.getcwd()/container.spy/container.<dataclass>.info

    Save `obj` in a new spy-container created by providing an absolute path

    >>> spy.save(obj, container="/tmp/container.spy")
    >>> # --> /tmp/container.spy/container.<dataclass>
    >>> # --> /tmp/container.spy/container.<dataclass>.info

    Save `obj` in a new (or existing) spy-container under a different tag

    >>> spy.save(obj, container="session1.spy", tag="someTag")
    >>> # --> os.getcwd()/session1.spy/session1_someTag.<dataclass>
    >>> # --> os.getcwd()/session1.spy/session1_someTag.<dataclass>.info

    See also
    --------
    syncopy.load : load data created with :func:`syncopy.save`
    """

    # Make sure `out` is a valid Syncopy data object
    data_parser(out, varname="out", writable=None, empty=False)

    if filename is None and container is None:
        raise SPYError('filename and container cannot both be `None`')

    if container is not None and filename is None:
        # construct filename from container name
        if not isinstance(container, str):
            raise SPYTypeError(container, varname="container", expected="str")
        if not os.path.splitext(container)[1] == ".spy":
            container += ".spy"
        fileInfo = filename_parser(container)
        filename = os.path.join(fileInfo["folder"],
                                fileInfo["container"],
                                fileInfo["basename"])
        # handle tag
        if tag is not None:
            if not isinstance(tag, str):
                raise SPYTypeError(tag, varname="tag", expected="str")
            filename += '_' + tag

    elif container is not None and filename is not None:
        raise SPYError("container and filename cannot be used at the same time")

    if not isinstance(filename, str):
        raise SPYTypeError(filename, varname="filename", expected="str")

    # add extension if not part of the filename
    if "." not in os.path.splitext(filename)[1]:
        filename += out._classname_to_extension()

    if not isinstance(overwrite, bool):
        raise SPYTypeError(overwrite, varname="overwrite", expected="bool")

    # Parse filename for validity and construct full path to HDF5 file
    fileInfo = filename_parser(filename)
    if fileInfo["extension"] != out._classname_to_extension():
        raise SPYError("""Extension in filename ('{ext}') does not match data
                    class ({dclass}), expected '{exp}'.""".format(ext=fileInfo["extension"],
                                               dclass=out.__class__.__name__, exp=out._classname_to_extension()))
    dataFile = os.path.join(fileInfo["folder"], fileInfo["filename"])

    # If `out` is to replace its own on-disk representation, be more careful
    if overwrite and dataFile == out.filename:
        replace = True
    else:
        replace = False

    # Prevent `out` from trying to re-create its own data file
    if replace:
        out.data.flush()
        h5f = out.data.file
        dat = out.data
        trl = h5f["trialdefinition"]
    else:
        if not os.path.exists(fileInfo["folder"]):
            try:
                os.makedirs(fileInfo["folder"])
            except IOError:
                raise SPYIOError(fileInfo["folder"])
            except Exception as exc:
                raise exc
        else:
            if os.path.exists(dataFile):
                if not os.path.isfile(dataFile):
                    raise SPYIOError(dataFile)
                if overwrite:
                    try:
                        h5f = h5py.File(dataFile, mode="w")
                        h5f.close()
                    except Exception as exc:
                        msg = "Cannot overwrite {} - file may still be open. "
                        msg += "Original error message below\n{}"
                        raise SPYError(msg.format(dataFile, str(exc)))
                else:
                    raise SPYIOError(dataFile, exists=True)
        h5f = h5py.File(dataFile, mode="w")

        # Save each member of `_hdfFileDatasetProperties` in target HDF file
        for datasetName in out._hdfFileDatasetProperties:
            dataset = getattr(out, "_" + datasetName)
            if dataset is not None:
                spy.log(f"Writing dataset '{datasetName}' ({len(out._hdfFileDatasetProperties)} datasets total) to HDF5 file '{dataFile}'.", level="DEBUG")
                dat = h5f.create_dataset(datasetName, data=dataset)
            else:
                spy.log(f"Not writing 'None 'dataset '{datasetName}' ({len(out._hdfFileDatasetProperties)} datasets total) to HDF5 file '{dataFile}'.", level="DEBUG")

    # Now write trial-related information
    trl_arr = np.array(out.trialdefinition)
    if replace:
        trl[()] = trl_arr
        trl.flush()
    else:
        trl = h5f.create_dataset("trialdefinition", data=trl_arr,
                                 maxshape=(None, trl_arr.shape[1]))

    # Write to log already here so that the entry can be exported to json
    infoFile = dataFile + FILE_EXT["info"]
    out.log = "Wrote files " + dataFile + "\n\t\t\t" + 2 * " " + infoFile

    # Assemble dict for JSON output: order things by their "readability"
    outDict = OrderedDict(startInfoDict)
    outDict["filename"] = fileInfo["filename"]
    outDict["dataclass"] = out.__class__.__name__
    outDict["data_dtype"] = dat.dtype.name
    outDict["data_shape"] = dat.shape
    outDict["data_offset"] = dat.id.get_offset()
    outDict["trl_dtype"] = trl.dtype.name
    outDict["trl_shape"] = trl.shape
    outDict["trl_offset"] = trl.id.get_offset()
    if isinstance(out.data, np.ndarray):
        if np.isfortran(out.data):
            outDict["order"] = "F"
    else:
        outDict["order"] = "C"

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
    for key in out._hdfFileAttributeProperties:
        if outDict[key] is None:
            h5f.attrs[key] = "None"
        else:
            try:
                h5f.attrs[key] = outDict[key]
            except RuntimeError:
                msg = "Too many entries in `{}` - truncating HDF5 attribute. " +\
                    "Please refer to {} for complete listing."
                info_fle = os.path.split(os.path.split(filename.format(ext=FILE_EXT["info"]))[0])[1]
                info_fle = os.path.join(info_fle, os.path.basename(
                    filename.format(ext=FILE_EXT["info"])))
                SPYWarning(msg.format(key, info_fle))
                h5f.attrs[key] = [outDict[key][0], "...", outDict[key][-1]]

    # Save the dataset names that should be loaded later into the JSON.
    outDict['_hdfFileDatasetProperties'] = list(out._hdfFileDatasetProperties)


    # Re-assign filename after saving (and remove source in case it came from `__storage__`)
    if not replace:
        h5f.close()
        if __storage__ in out.filename:
            out.data.file.close()
            try:
                os.unlink(out.filename)
            except PermissionError as ex:
                print(f"Could not delete file '{out.filename}': {str(ex)}.")
        out.data = dataFile

    # Compute checksum and finally write JSON (automatically overwrites existing)
    outDict["file_checksum"] = hash_file(dataFile)

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
        elif isinstance(value, np.ndarray):
            dct[key] = value.tolist()
        else:
            if hasattr(value, "item"):
                value = value.item()
            try:
                json.dumps(value)
            except (TypeError, OverflowError):
                value = str(value)
            dct[key] = value
    return
