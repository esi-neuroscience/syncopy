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
def copy(spydata):
    """
    Create a copy of the entire Syncopy object `data` on disk

    Parameters
    ----------
    spydata : Syncopy data object
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
    data_parser(spydata, varname="data", writable=None, empty=False)

    dsize = np.prod(spydata.data.shape) * spydata.data.dtype.itemsize / 1024 ** 2
    msg = (
        f"Copying {dsize:.2f} MB of data "
        f"to create new {spydata.__class__.__name__} object on disk"
    )
    SPYInfo(msg)

    # Shallow copy, captures also non-default/temporary attributes.
    copy_spydata = py_copy(spydata)
    copy_filename = spydata._gen_filename()
    copy_spydata.filename = copy_filename
    spydata.clear()
    #copy_spydata.clear()


    is_backing_hdf5_file_open = spydata._data.id.valid != 0
    print(f"copy: spydata hdf5 file open: {is_backing_hdf5_file_open}")

    spydata._close()

    is_backing_hdf5_file_open = spydata._data.id.valid != 0
    print(f"copy: hdf5 file open: {is_backing_hdf5_file_open}")

    # Copy data on disk.
    shutil.copyfile(spydata.filename, copy_filename, follow_symlinks=False)

    sfile = h5py.File(spydata.filename, mode="r")
    source_keys = [k for k in sfile.keys()]
    sfile.close()

    cfile = h5py.File(copy_filename, mode="r")
    copy_keys = [k for k in cfile.keys()]
    cfile.close()
    print(f"source_keys: {source_keys}")
    print(f"copy_keys: {copy_keys}")

    spydata._reopen()

    is_backing_hdf5_file_open = spydata._data.id.valid != 0
    print(f"copy: hdf5 file open: {is_backing_hdf5_file_open}")



    # Copying the data on disk does, for some reason, not copy the extra dataset.
    # Maybe the 'clear()' data flushes the HDF5 buffer, but not the O/S buffer.
    # Whatever the reason is, we need to manually copy the extra datasets.

    # Copy extra datasets manually with hdf5.
    do_copy_with_hdf5 = False
    if do_copy_with_hdf5:
        for propertyName in spydata._hdfFileDatasetProperties:
            if propertyName != "data":
                src_h5py_file = spydata._data.file
                with h5py.File(copy_spydata.filename, mode="r+") as dst_h5py_file:
                    src_h5py_file.copy(
                        src_h5py_file[propertyName], dst_h5py_file["/"], propertyName
                    )

    # Reattach properties
    for propertyName in spydata._hdfFileDatasetProperties:
        prop = getattr(spydata, "_" + propertyName)
        if isinstance(prop, h5py.Dataset):
            sourceName = prop.name
            setattr(
                copy_spydata,
                "_" + propertyName,
                h5py.File(copy_filename, mode=copy_spydata.mode)[sourceName],
            )
        else:  # np.ndarray
            setattr(copy_spydata, "_" + propertyName, prop)

    return copy_spydata
