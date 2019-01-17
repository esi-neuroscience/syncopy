# core.py - SpykeWave basic datatype reference implementation
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: Januar  7 2019
# Last modified: <2019-01-17 14:25:12>

# Builtin/3rd party package imports
import numpy as np
import getpass
import socket
import time
from collections import OrderedDict

# Local imports
from spykewave.utils import spw_io_parser, spw_scalar_parser, spw_array_parser, SPWIOError
from spykewave import __version__

__all__ = ["BaseData", "read_binary_header"]

# Base SpykeWave data container
class BaseData():

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

    # Class instantiation
    def __init__(self,
                 dataset="",
                 len_segment=None,
                 label="channel",
                 t0=0,
                 segmentlabel="trial"):

        # Depending on contents of `dataset`, things work differently in here
        try:
            read_ds = bool(len(dataset))
        except:
            raise SPWTypeError(dataset, varname="dataset", expected="str")

        # We only parse quantities here that are somewhat independet of `dataset`
        try:
            spw_scalar_parser(t0, varname="t0", lims=[-np.inf, np.inf])
        except Exception as exc:
            raise exc
        
        if not isinstance(segmentlabel, str):
            raise SPWTypeError(segmentlabel, varname="segmentlabel", expected="str")
        options = ["trial"]
        if segmentlabel not in options:
            raise ValueError("".join(opt + ", " for opt in options)[:-2],
                             varname="segmentlabel", actual=segmentlabel)
        self._segmentlabel = segmentlabel

        # Allocate "global" attribute dictionary
        dlbls = ["label", "sr", "tstart"]
        self._dimlabels = OrderedDict(zip(dlbls, len(dlbls)*[None]))

        # If dataset was provided, start reading it, otherwise fill in blanks
        if read_ds:
            self._stream_dataset(dataset, len_segment, t0, label)
        else:
            self._segments = [np.array([])]
            self._sampleinfo = [(t0,t0)]
            self._hdr = {"tSample" : 0}
            self._time = []
            
        # In case the segments are trials, dynamically add a "trial" property 
        # to emulate FieldTrip usage
        if self._segmentlabel == "trial":
            setattr(BaseData, "trial", property(lambda self: self._segments))

        # Initialize log by writing header
        lhd = "\n\t\t>>> SpykeWave v. {ver:s} <<< \n\n" +\
              "Created: {timestamp:s} \n\n" +\
              "--- LOG --- "
        self._log_header = lhd.format(ver=__version__, timestamp=time.asctime())
        self._log = self._log_header + ""

        # Write first entry to log
        log = "Instantiated BaseData object using parameters\n" +\
              "\tdataset = {dset:s}\n" +\
              "\tlen_segment = {ls:s}\n" +\
              "\tt0 = {t0:s}\n" +\
              "\tsegmentlabel = {sl:s}"
        self.log = log.format(dset=str(dataset) if len(dataset) else "None",
                              ls=str(len_segment),
                              t0=str(t0),
                              sl=segmentlabel)

    # FIXME: this shouldn't be a class method but a routine imported from an I/O package
    def _stream_dataset(self, dataset, len_segment, t0, label):

        # Get header of dataset
        self._hdr = read_binary_header(dataset)

        # If no desired segment-length for data chunking was provided, use everything at once
        if len_segment is None:
            len_segment = self._hdr["N"]
        try:
            spw_scalar_parser(len_segment, varname="len_segment",
                              ntype="int_like", lims=[1, self._hdr["N"]])
        except Exception as exc:
            raise exc
        
        # Construct/parse list of channel labels
        if isinstance(label, str):
            label = [label + str(i + 1) for i in range(self._hdr["M"])]
        try:
            spw_array_parser(label, varname="label", ntype="str", dims=(self._hdr["M"],))
        except Exception as exc:
            raise exc
        self._dimlabels["label"] = label
        
        # Split up raw data based on `len_segment`
        rem = int(self._hdr["N"] % len_segment)
        n_seg = np.array([len_segment]*int(self._hdr["N"]//len_segment) + [rem]*int(rem >0))

        # Segment timing
        self._dimlabels["tstart"] = np.cumsum(n_seg) - n_seg + t0
        self._sampleinfo = [(ts, ts + ls) for (ts, ls) in zip(self._dimlabels['tstart'], n_seg)]
        self._time = [range(start, end) for (start, end) in self._sampleinfo]
        
        # Cycle through segments and allocate memmaps (and use the fact that
        # by construction if `len(n_seg) > 1` all segments except the
        # last one have length `len_segment`)
        self._segments = []
        for k, N in enumerate(n_seg):
            self._segments.append(np.memmap(dataset,
                                            offset=int(self._hdr["length"] + k*len_segment),
                                            mode="r", dtype=self._hdr["dtype"],
                                            shape=(self._hdr["M"], N)))

    @property
    def dimlabels(self):
        return self._dimlabels
    
    @property
    def hdr(self):
        return self._hdr

    @property
    def label(self):
        return self._dimlabels["label"]
    
    @property
    def log(self):
        print(self._log)

    @log.setter
    def log(self, msg):
        prefix = "\n\n|=== {user:s}@{host:s}: {time:s} ===|\n\n\t"
        self._log += prefix.format(user=getpass.getuser(),
                                   host=socket.gethostname(),
                                   time=time.asctime()) + msg

    @property
    def segments(self):
        return self._segments

    @property
    def segmentlabel(self):
        return self._segmentlabel

    @property
    def time(self, unit="ns"):
        converter = {"h": 1/360*1e-9, "min": 1/60*1e-9, "s" : 1e-9, "ms" : 1e-6, "ns" : 1}
        factor = self._hdr["tSample"]*converter[unit]
        return [np.arange(start, end)*factor for (start, end) in self._sampleinfo]

##########################################################################################
# FIXME: this should be in an I/O package
def read_binary_header(dataset=""):
    """
    Docstring
    """

    # First and foremost, make sure input arguments make sense
    try:
        spw_io_parser(dataset, varname="dataset")
    except Exception as exc:
        raise exc

    # Try to access dataset on disk, abort if this goes wrong
    try:
        fid = open(dataset, "r")
    except:
        raise SPWIOError(dataset)

    # Extract file header
    hdr = {}
    hdr["version"] = np.fromfile(fid,dtype='uint8',count=1)[0]
    hdr["length"] = np.fromfile(fid,dtype='uint16',count=1)[0]
    hdr["dtype"] = BaseData.dtype[np.fromfile(fid,dtype='uint8',count=1)[0]]
    hdr["N"] = np.fromfile(fid,dtype='uint64',count=1)[0]
    hdr["M"] = np.fromfile(fid,dtype='uint64',count=1)[0]
    hdr["tSample"] = np.fromfile(fid,dtype='uint64',count=1)[0]
    fid.close()

    return hdr
