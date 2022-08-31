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

    spydata._close()

    # Copy data on disk.
    shutil.copyfile(spydata.filename, copy_filename, follow_symlinks=False)

    spydata._reopen()

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
