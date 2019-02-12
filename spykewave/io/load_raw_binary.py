# load_raw_binary.py - Read binary files from disk
# 
# Created: Januar 22 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-12 12:53:45>

# Builtin/3rd party package imports
import os
import numpy as np

# Local imports
from spykewave.utils import (spw_io_parser, spw_scalar_parser, spw_array_parser,
                             spw_basedata_parser, SPWIOError, SPWTypeError, SPWValueError)
from spykewave.datatype import AnalogData, VirtualData

__all__ = ["load_binary_esi", "read_binary_esi_header"]

##########################################################################################
def load_binary_esi(filename,
                    label="channel",
                    trialdefinition=None,
                    segmentlabel="trial",
                    out=None):
    """
    Docstring
    """

    # Make sure `out` does not contain unpleasant surprises (for this, we need
    # to parse segmentlabel first)
    if not isinstance(segmentlabel, str):
        raise SPWTypeError(segmentlabel, varname="segmentlabel", expected="str")
    if out is not None:
        try:
            spw_basedata_parser(out, varname="out", seglabel=str(segmentlabel))
        except Exception as exc:
            raise exc
        new_out = False
        out = AnalogData(out, copy=False)
    else:
        out = AnalogData()
        new_out = True

    # Convert input to list (if it is not already) - parsing is performed
    # by ``read_binary_esi_header``
    if not isinstance(filename, (list, np.ndarray)):
        filename = [filename]

    # Parse `trialdefinition`
    if trialdefinition is not None:
        try:
            spw_array_parser(trialdefinition, varname="trialdefinition", dims=2)
        except Exception as exc:
            raise exc
        if trialdefinition.shape[1] < 3:
            raise SPWValueError("array of shape (no. of trials, 3+)",
                                varname="trialdefinition",
                                actual="shape = {shp:s}".format(shp=str(trialdefinition.shape)))

    # Read headers of provided file(s) to get dimensional information
    headers = []
    tsample = []
    filename = [os.path.abspath(fname) for fname in filename]
    for fname in filename:
        hdr = read_binary_esi_header(fname)
        hdr["file"] = fname
        headers.append(hdr)
        tsample.append(hdr["tSample"])

    # Abort, if files have differing sampling times
    if not np.array_equal(tsample, [tsample[0]]*len(tsample)):
        raise SPWValueError(legal="identical sampling interval per file")

    # Allocate memmaps for each file
    dsets = []
    for fk, fname in enumerate(filename):
        dsets.append(np.memmap(fname, offset=int(headers[fk]["length"]),
                               mode="r", dtype=headers[fk]["dtype"],
                               shape=(headers[fk]["M"], headers[fk]["N"])))

    # Instantiate VirtualData class w/ constructed memmaps (error checking is done in there)
    data = VirtualData(dsets)

    # Construct/parse list of channel labels
    if isinstance(label, str):
        label = [label + str(i + 1) for i in range(data.M)]
    try:
        spw_array_parser(label, varname="label", ntype="str", dims=(data.M,))
    except Exception as exc:
        raise exc

    # If not provided construct (trivial) `trialdefinition` array
    if trialdefinition is None:
        trialdefinition = np.array([[0, data.N, 0]])

    # Write dimensional information - order matters here!
    out._dimlabels["label"] = label
    out._dimlabels["sample"] = trialdefinition[:, :2]

    # Fill up mandatory `BaseData` attributes
    out.data = data
    out._mode = "r"
    out.segmentlabel = segmentlabel
    out.cfg = {"filename" : filename,
               "segmentlabel" : segmentlabel}
    
    # Write attributes specific to `AnalogData` class
    out._hdr = headers
    out._samplerate = float(1/headers[0]["tSample"]*1e9)
    out._seg = trialdefinition
    out._time = [range(start, end) for (start, end) in out.sampleinfo]
    
    # Write log entry
    log = "loaded data:\n" +\
          "\tfile(s) = {fls:s}\n" +\
          "\tsegmentlabel = {sl:s}"
    out.log = log.format(fls="\n\t\t  ".join(fl for fl in filename),
                         sl=segmentlabel)

    # Happy breakdown
    return out if new_out else None

##########################################################################################
def read_binary_esi_header(filename):
    """
    Docstring
    """

    # SpykeWave raw binary dtype-codes
    dtype = {
        1 : 'int8',
        2 : 'uint8', 
        3 : 'int16', 
        4 : 'uint16', 
        5 : 'int32', 
        6 : 'uint32', 
        7 : 'int64', 
        8 : 'uint64', 
        9 : 'float32',
        10 : 'float64'
    }

    # First and foremost, make sure input arguments make sense
    try:
        spw_io_parser(filename, varname="filename", isfile=True,
                      ext=[".lfp", ".mua", ".evt", ".dpd", 
                           ".apd", ".eye", ".pup"])
    except Exception as exc:
        raise exc

    # Try to access file on disk, abort if this goes wrong
    try:
        fid = open(filename, "r")
    except:
        raise SPWIOError(filename)

    # Extract file header
    hdr = {}
    hdr["version"] = np.fromfile(fid,dtype='uint8',count=1)[0]
    hdr["length"] = np.fromfile(fid,dtype='uint16',count=1)[0]
    hdr["dtype"] = dtype[np.fromfile(fid,dtype='uint8',count=1)[0]]
    hdr["N"] = np.fromfile(fid,dtype='uint64',count=1)[0]
    hdr["M"] = np.fromfile(fid,dtype='uint64',count=1)[0]
    hdr["tSample"] = np.fromfile(fid,dtype='uint64',count=1)[0]
    fid.close()

    return hdr
    
