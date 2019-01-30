# core.py - SpykeWave basic datatype reference implementation
# 
# Created: January 7 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-01-30 13:44:01>

# Builtin/3rd party package imports
import numpy as np
import getpass
import socket
import time
import numbers
import inspect
from collections import OrderedDict, Iterator

# Local imports
from spykewave.utils import (spw_scalar_parser, spw_array_parser,
                             SPWTypeError, SPWValueError, spw_warning)
from spykewave import __version__
import spykewave as sw

__all__ = ["BaseData", "ChunkData"]

##########################################################################################
class BaseData():

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
        return map(self.get_segment, range(self._trialinfo.shape[0]))

    @property
    def segmentlabel(self):
        return self._segmentlabel

    @property
    def time(self, unit="ns"):
        converter = {"h": 1/360*1e-9, "min": 1/60*1e-9, "s" : 1e-9, "ms" : 1e-6, "ns" : 1}
        factor = self._hdr["tSample"]*converter[unit]
        return [np.arange(start, end)*factor for (start, end) in self._sampleinfo]

    # Class instantiation
    def __init__(self,
                 filename="",
                 filetype=None,
                 trialdefinition=None,
                 label="channel",
                 segmentlabel="trial"):
        """
        Main SpykeWave data container
        """

        # Depending on contents of `filename`, class instantiation invokes I/O routines
        try:
            read_fl = bool(len(filename))
        except:
            raise SPWTypeError(filename, varname="filename", expected="str")

        # We only parse quantities here that are converted to class attributes
        # even if `filename` is empty
        if not isinstance(segmentlabel, str):
            raise SPWTypeError(segmentlabel, varname="segmentlabel", expected="str")
        options = ["trial"]
        if segmentlabel not in options:
            raise SPWValueError("".join(opt + ", " for opt in options)[:-2],
                                varname="segmentlabel", actual=segmentlabel)
        self._segmentlabel = segmentlabel

        # Prepare "global" attribute dictionary for reading routines
        dlbls = ["label", "sr", "tstart"]
        self._dimlabels = OrderedDict(zip(dlbls, len(dlbls)*[None]))

        # If filename was provided, call appropriate reading routine
        if read_fl:
            sw.read_data(filename, filetype=filetype, label=label,
                         trialdefinition=trialdefinition, out=self)
        else:
            self._segments = [np.array([])]
            self._sampleinfo = [(0,0)]
            self._hdr = {"tSample" : 0}
            self._time = []
            self._trialinfo = np.zeros((3,))
            
        # In case the segments are trials, dynamically add a "trial" property 
        # to emulate FieldTrip usage
        if self._segmentlabel == "trial":
            setattr(BaseData, "trial", property(lambda self: self.segments))
            setattr(BaseData, "trialinfo", property(lambda self: self._trialinfo))

        # Initialize log by writing header information
        lhd = "\n\t\t>>> SpykeWave v. {ver:s} <<< \n\n" +\
              "Created: {timestamp:s} \n\n" +\
              "--- LOG --- "
        self._log_header = lhd.format(ver=__version__, timestamp=time.asctime())
        self._log = self._log_header + ""

        # Write first entry to log
        log = "Instantiated BaseData object using parameters\n" +\
              "\tfilename = {dset:s}\n" +\
              "\tsegmentlabel = {sl:s}"
        self.log = log.format(dset=str(filename) if len(filename) else "None",
                              sl=segmentlabel)

    # Helper function that leverages `ChunkData`'s getter routine to return a single segment
    def get_segment(self, segno):
        # FIXME: make sure `segno` is a valid segment number
        return self._segments[:, self._trialinfo[segno, 0]: self._trialinfo[segno, 1]]

    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make class contents readable from the command line
    def __str__(self):

        # Get list of print-worthy attributes
        ppattrs = [attr for attr in self.__dir__() if not (attr.startswith("_") or attr == "log")]
        ppattrs = [attr for attr in ppattrs \
                   if not (inspect.ismethod(getattr(self, attr)) \
                           or isinstance(getattr(self, attr), Iterator))]

        # Construct string for pretty-printing class attributes
        ppstr = "SpykeWave BaseData object with fields\n\n"
        maxKeyLength = max([len(k) for k in ppattrs])
        for attr in ppattrs:
            value = getattr(self, attr)
            if hasattr(value, 'shape'):            
                valueString = "[" + " x ".join([str(numel) for numel in value.shape]) \
                              + "] element " + str(type(value))
            elif isinstance(value, list):
                valueString = "{0} element list".format(len(value))
            elif isinstance(value, dict):
                msg = "dictionary with {nk:s}keys{ks:s}"
                keylist = value.keys()
                showkeys = len(keylist) < 7
                valueString = msg.format(nk=str(len(keylist)) + " " if not showkeys else "",
                                         ks=" '" + "', '".join(key for key in keylist) + "'" if showkeys else "")
            else:
                valueString = str(value)
            printString =  "{0:>" + str(maxKeyLength + 5) + "} : {1:}\n"
            ppstr += printString.format(attr, valueString)
        return ppstr

##########################################################################################
class ChunkData():

    @property
    def M(self):
        return self._M

    @property
    def N(self):
        return self._N

    @property
    def shape(self):
        return self._shape

    @property
    def size(self):
        return self._size
    
    # Class instantiation
    def __init__(self, chunk_list):
        """
        Docstring coming soon...
        """

        # First, make sure our one mandatary input argument does not contain
        # any unpleasant surprises
        if not isinstance(chunk_list, (list, np.ndarray)):
            raise SPWTypeError(chunk_list, varname="chunk_list", expected="array_like")
        for chunk in chunk_list:
            try:
                spw_array_parser(chunk, varname="chunk", dims=2)
            except Exception as exc:
                raise exc

        # Get row number per input chunk and raise error in case col.-no. does not match up
        shapes = [chunk.shape for chunk in chunk_list]
        if not np.array_equal([shape[1] for shape in shapes], [shapes[0][1]]*len(shapes)):
            raise SPWValueError(legal="identical number of samples per chunk",
                                varname="chunk_list")
        nrows = [shape[0] for shape in shapes]
        cumlen = np.cumsum(nrows)

        # Create list of "global" row numbers and assign "global" dimensional info
        self._nrows = nrows
        self._rows = [range(start, stop) for (start, stop) in zip(cumlen - nrows, cumlen)]
        self._M = cumlen[-1]
        self._N = chunk_list[0].shape[1]
        self._shape = (self._M, self._N)
        self._size = self._M*self._N
        self._data = chunk_list

    # Compatibility
    def __len__(self):
        return self._size

    # The only part of this class that actually does something
    def __getitem__(self, idx):

        # Extract queried row/col from input tuple `idx`
        qrow, qcol = idx
        
        # Convert input to slice (if it isn't already) or assign explicit start/stop values
        if isinstance(qrow, numbers.Number):
            try:
                spw_scalar_parser(qrow, varname="row", ntype="int_like", lims=[0, self._M])
            except Exception as exc:
                raise exc
            row = slice(int(qrow), int(qrow + 1))
        elif isinstance(qrow, slice):
            start, stop = qrow.start, qrow.stop
            if qrow.start is None:
                start = 0
            if qrow.stop is None:
                stop = self._M
            row = slice(start, stop)
        else:
            raise SPWTypeError(qrow, varname="row", expected="int_like or slice")    
        
        # Convert input to slice (if it isn't already) or assign explicit start/stop values
        if isinstance(qcol, numbers.Number):
            try:
                spw_scalar_parser(qcol, varname="col", ntype="int_like", lims=[0, self._N])
            except Exception as exc:
                raise exc
            col = slice(int(qcol), int(qcol + 1))
        elif isinstance(qcol, slice):
            start, stop = qcol.start, qcol.stop
            if qcol.start is None:
                start = 0
            if qcol.stop is None:
                stop = self._N
            col = slice(start, stop)
        else:
            raise SPWTypeError(qcol, varname="col", expected="int_like or slice")

        # Make sure queried row/col are inside dimensional limits
        err = "value between {lb:s} and {ub:s}"
        if not(0 <= row.start < self._M) or not(0 < row.stop <= self._M):
            raise SPWValueError(err.format(lb="0", ub=str(self._M)),
                                varname="row", actual=str(row))
        if not(0 <= col.start < self._N) or not(0 < col.stop <= self._N):
            raise SPWValueError(err.format(lb="0", ub=str(self._N)),
                                varname="col", actual=str(col))

        # The interesting part: find out wich chunk(s) `row` is pointing at
        i1 = np.where([row.start in chunk for chunk in self._rows])[0].item()
        i2 = np.where([(row.stop - 1) in chunk for chunk in self._rows])[0].item()

        # If start and stop are not within the same chunk, data is loaded into memory
        if i1 != i2:
            # spw_warning("Loading multiple files into memory", caller="SpykeWave core")
            data = []
            data.append(self._data[i1][row.start - self._rows[i1].start:, col])
            for i in range(i1 + 1, i2):
                data.append(self._data[i][:, col])
            data.append(self._data[i2][:row.stop - self._rows[i2].start, col])
            return np.vstack(data)

        # If start and stop are in the same chunk, return a view of the underlying memory map
        else:
            
            # Convert "global" row index to local chunk-based row-number (by subtracting offset)
            row = slice(row.start - self._rows[i1].start, row.stop - self._rows[i1].start)
            return self._data[i1][row,:][:,col]
