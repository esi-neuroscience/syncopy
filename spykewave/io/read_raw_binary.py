# read_raw_binary.py - Read binary files from disk
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar 22 2019
# Last modified: <2019-01-23 17:48:18>

# Builtin/3rd party package imports
import numpy as np

# Local imports
from spykewave.utils import (spw_io_parser, spw_scalar_parser, spw_array_parser,
                             SPWIOError, SPWTypeError)
from spykewave.datatype import BaseData, ChunkData

__all__ = ["read_binary_esi", "read_binary_esi_header"]

##########################################################################################
def read_binary_esi(filename,
                    label="channel",
                    trialdefinition=None,
                    t0=0,
                    segmentlabel="trial",
                    out=None):
    """
    Docstring
    """

    # Make sure we can handle `out`
    if out is not None:
        # if not isinstance(out, BaseData):
        #     raise SPWTypeError(out, varname="out", expected="SpkeWave BaseData object")
        return_out = False
    else:
        out = BaseData(segmentlabel=segmentlabel)
        return_out = True

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

    # Parse `t0`
    try:
        spw_scalar_parser(t0, varname="t0", lims=[-np.inf, np.inf])
    except Exception as exc:
        raise exc
        
    # Read headers of provided file(s) to get dimensional information
    headers = []
    tsample = []
    for fname in filename:
        hdr = read_binary_esi_header(fname)
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

    # Instantiate ChunkData class w/ constructed memmaps (error checking is done in there)
    data = ChunkData(dsets)

    # Construct/parse list of channel labels
    if isinstance(label, str):
        label = [label + str(i + 1) for i in range(data.M)]
    try:
        spw_array_parser(label, varname="label", ntype="str", dims=(data.M,))
    except Exception as exc:
        raise exc

    # If not provided construct (trivial) `trialdefinition` array
    if trialdefinition is None:
        trialdefinition = np.array([0,data.N,0])
    
    # Everything's ready, attach things to `out` (and return)
    out._segments = data
    out._dimlabels["label"] = label
    out._dimlabels["tstart"] = trialdefinition[:, 0]
    out._sampleinfo = trialdefinition[:, :2]
    # out._sampleinfo = [(trialdefinition[k, 0], trialdefinition[k, 1]) \
    #                    for k in range(trialdefinition.shape[0])]
    out._trialinfo = trialdefinition
    out._time = [range(start, end) for (start, end) in out._sampleinfo]
    out._hdr = headers[0]

    if return_out:
        return out
    else:
        return

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
        spw_io_parser(filename, varname="filename")
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
    
