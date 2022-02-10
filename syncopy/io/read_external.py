# -*- coding: utf-8 -*-
#
# Import data from 3rd party formats
#

# Builtin/3rd party package imports
import os

# Local imports
from syncopy.shared.parsers import io_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError
from ._read_nwb import read_nwb

supportedFormats = ["nwb"]

__all__ = ["read"]


def read(filename, format=None):
    """
    Read data from non-native Syncopy file formats

    Parameters
    ----------
    filename : str
        Name of (may include full path to) file to read
    format : None or str
        If the external format cannot be inferred from `filename`, the `format`
        specifier can be used to manually set it

    Returns
    -------
    data : Syncopy data object(s)
        Depending on `filename` on or more Syncopy data objects is returned

    Notes
    -----
    This manager may be used as general purpose import function for reading
    third party file formats. However, to leverage specific functionality of
    the respective format-specific reading routines, please invoke the corresponding
    functions directly. For instance, if you want to decrease the in-memory
    footprint of reading NWB files, use :func:`~syncopy.io._read_nwb.read_nwb`
    directly.

    See also
    --------
    syncopy.io._read_nwb.read_nwb : read contents of NWB files
    """

    # Parse basal input args (thorough error checking is performed by the actual
    # importer functions)
    _, baseName = io_parser(filename, varname="filename", exists=True)
    if not isinstance(format, (type(None), str)):
        raise SPYTypeError(format, varname="format", expected="string")

    # Ensure we can actually process a provided `format`
    if format is not None:
        format = format.replace(".", "")
        if format not in supportedFormats:
            lgl = "one of the following supported formats " +\
                "".join(fmt + ", " for fmt in supportedFormats)[:-2]
            raise SPYValueError(lgl, varname="format", actual=format)

    # Try to infer file-format from file extension
    if format is None:
        _, ext = os.path.splitext(baseName)
        if ext == ".nwb":
            format = "nwb"

    # Call appropriate importer
    if format == "nwb":
        read_nwb(filename)

    return

