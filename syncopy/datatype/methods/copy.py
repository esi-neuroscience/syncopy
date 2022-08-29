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

    # Shallow copy, captures also non-default/temporary attributes.
    copy_spdata = py_copy(spdata)
    spdata.clear()
    copy_filename = spdata._gen_filename()
    copy_spdata.filename = copy_filename
    copy_spdata.clear()

    # Copy data on disk.
    shutil.copyfile(spdata.filename, copy_filename, follow_symlinks=False)

    # Copying the data on disk does, for some reason, not copy the extra dataset.
    # Maybe the 'clear()' data flushes the HDF5 buffer, but not the O/S buffer.
    # Whatever the reason is, we need to manually copy the extra datasets.

    #print(f"copy: copied file. the SOURCE h5py file has entries: {h5py.File(spdata.filename, mode='r').keys()}")
    #print(f"copy: copied file. the copied H5py file has entries: {h5py.File(copy_spdata.filename, mode='r').keys()}")

    # Copy extra datasets manually with hdf5.
    for propertyName in spdata._hdfFileDatasetProperties:
        if propertyName != "data":
            src_h5py_file = spdata._data.file
            with h5py.File(copy_spdata.filename, mode='r+') as dst_h5py_file:
                src_h5py_file.copy(src_h5py_file[propertyName], dst_h5py_file["/"], propertyName)



    # reattach properties
    for propertyName in spdata._hdfFileDatasetProperties:
        prop = getattr(spdata, "_" + propertyName)
        if isinstance(prop, h5py.Dataset):
            sourceName = prop.name
            setattr(copy_spdata, "_" + propertyName,
                    h5py.File(copy_filename, mode=copy_spdata.mode)[sourceName])
        else:   # np.ndarray
            setattr(copy_spdata, "_" + propertyName, prop)

    return copy_spdata
