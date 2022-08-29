# -*- coding: utf-8 -*-
#
# Syncopy's (deep) copy function
#

# Builtin/3rd party package imports
from copy import copy as py_copy
import shutil
import h5py
import numpy as np

# Syncopy imports
from syncopy.shared.parsers import data_parser
from syncopy.shared.errors import SPYInfo

__all__ = ["copy"]


# Return a deep copy of the current class instance
def copy(spdata):
    """
    Create a copy of the entire Syncopy object `data` on disk

    Parameters
    ----------
    spdata : Syncopy data object
        Object to be copied on disk

    Returns
    -------
    cpy : Syncopy data object
        Reference to the copied data object
        on disk

    Notes
    -----
    For copying only a subset of the `data` use :func:`syncopy.selectdata` directly
    with the default `inplace=False` parameter.

    Syncopy objects may also be copied using the class method ``.copy`` that
    acts as a wrapper for :func:`syncopy.copy`

    See also
    --------
    :func:`syncopy.save` : save to specific file path
    :func:`syncopy.selectdata` : creates copy of a selection with `inplace=False`
    """

    # Make sure `data` is a valid Syncopy data object
    data_parser(spdata, varname="data", writable=None, empty=False)

    dsize = np.prod(spdata.data.shape) * spdata.data.dtype.itemsize / 1024**2
    msg = (f"Copying {dsize:.2f} MB of data "
           f"to create new {spdata.__class__.__name__} object on disk")
    SPYInfo(msg)

    # shallow copy, captures also non-default/temporary attributes
    copy_spdata = py_copy(spdata)
    spdata.clear()
    copy_filename = spdata._gen_filename()
    copy_spdata.filename = copy_filename
    copy_spdata.clear()

    print(f"copy: shallow-copied spdata object. old one has ._hdfFileDatasetProperties: {spdata._hdfFileDatasetProperties}")
    print(f"copy: shallow-copied spdata object. new one has ._hdfFileDatasetProperties: {copy_spdata._hdfFileDatasetProperties}")

    # copy data on disk
    shutil.copyfile(spdata.filename, copy_filename, follow_symlinks=False)
    #shutil.copy2(spdata.filename, copy_filename, follow_symlinks=False)

    #print(f"copy: copied file. the SOURCE h5py file has entries: {h5py.File(spdata.filename, mode='r').keys()}")
    #print(f"copy: copied file. the copied H5py file has entries: {h5py.File(copy_spdata.filename, mode='r').keys()}")

    print(f"copy: copying extra datasets manually with hdf5")
    for propertyName in spdata._hdfFileDatasetProperties:
        src_h5py_file = spdata._data.file
        #dst_h5py_file = copy_spdata._data.file # No! This is still the old file, due to shallow copy.
        dst_h5py_file = h5py.File(copy_spdata.filename, mode='r+')
        assert isinstance(src_h5py_file, h5py.File)
        assert isinstance(dst_h5py_file, h5py.File)
        print(f"copying manually from file {src_h5py_file} to {dst_h5py_file}")
        if propertyName != "data":
            #with h5py.File(copy_spdata.filename, 'w') as f_dest:
            #    with h5py.File(src_h5py_file.filename, 'r') as f_src:
            #       f_src.copy(f_src["/" + propertyName], f_dest["/" + propertyName])
            src_h5py_file.copy(src_h5py_file[propertyName], dst_h5py_file["/"], propertyName)



    # reattach properties
    for propertyName in spdata._hdfFileDatasetProperties:
        prop = getattr(spdata, "_" + propertyName)
        if isinstance(prop, h5py.Dataset):
            sourceName = prop.name
            print(f"copy: reattaching '{propertyName}': >>> hdf5 with sourceName '{sourceName}'")
            print(f"copy: reattaching '{propertyName}': >>>  the source H5py file has entries: {h5py.File(spdata.filename, mode=copy_spdata.mode).keys()}")
            print(f"copy: reattaching '{propertyName}': >>>  the copied H5py file has entries: {h5py.File(copy_filename, mode=copy_spdata.mode).keys()}")
            setattr(copy_spdata, "_" + propertyName,
                    h5py.File(copy_filename, mode=copy_spdata.mode)[sourceName])
        else:   # np.ndarray?
            print(f"copy: reattaching '{propertyName}': >>> NOT hdf5 with sourceName '{sourceName}', type is {type(propertyName)}.")
            setattr(copy_spdata, "_" + propertyName, prop)

    return copy_spdata
