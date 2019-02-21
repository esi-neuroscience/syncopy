# core.py - SpykeWave basic datatype reference implementation
# 
# Created: January 7 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-21 17:59:01>

# Builtin/3rd party package imports
import numpy as np
import getpass
import socket
import time
import sys
import os
import numbers
import inspect
import scipy as sp
from collections import OrderedDict, Iterator
from copy import copy
from hashlib import blake2b
from itertools import islice
from numpy.lib.format import open_memmap
    
# Local imports
from spykewave.utils import (spw_scalar_parser, spw_array_parser,
                             SPWTypeError, SPWValueError, spw_warning)
from spykewave import __version__, __storage__, __dask__
if __dask__:
    import dask
import spykewave as sw

__all__ = ["BaseData", "AnalogData", "SpectralData", "VirtualData", "Indexer"]

##########################################################################################
class BaseData():

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, dct):
        if not isinstance(dct, dict):
            raise SPWTypeError(dct, varname="cfg", expected="dictionary")
        self._cfg = self._set_cfg(self._cfg, dct)

    @property
    def data(self):
        return self._data
   
    @property
    def dimord(self):
        return list(self._dimlabels.keys())

    @property
    def label(self):
        return self._dimlabels.get("label")

    @property
    def log(self):
        print(self._log_header + self._log)

    @log.setter
    def log(self, msg):
        if not isinstance(msg, str):
            raise SPWTypeError(msg, varname="log", expected="str")
        prefix = "\n\n|=== {user:s}@{host:s}: {time:s} ===|\n\n\t{caller:s}"
        clr = sys._getframe().f_back.f_code.co_name
        self._log += prefix.format(user=getpass.getuser(),
                                   host=socket.gethostname(),
                                   time=time.asctime(),
                                   caller=clr + ": " if clr != "<module>" else "")\
                                   + msg

    @property
    def mode(self):
        return self._mode
    
    @property
    def seg(self):
        return self._seg
    
    @property
    def segmentlabel(self):
        return self._segmentlabel

    @segmentlabel.setter
    def segmentlabel(self, seglbl):
        if not isinstance(seglbl, str):
            raise SPWTypeError(seglbl, varname="segmentlabel", expected="str")
        options = ["sample", "freq"]
        if seglbl not in options:
            raise SPWValueError("".join(opt + ", " for opt in options)[:-2],
                                varname="segmentlabel", actual=seglbl)
        if self._segmentlabel is None:
            self._segmentlabel = seglbl
        else:
            if self._segmentlabel != seglbl:
                msg = "Cannot change `segmentlabel` property from " +\
                      "'{current:s}' to '{wanted:s}'. Please create new BaseData object"
                spw_warning(msg.format(current=str(self._segmentlabel), wanted=seglbl),
                            caller="SpykeWave core")

    @property
    def shapes(self):
        if self.seg is not None:
            sid = self.dimord.index(self.segmentlabel)
            shp = [list(self._data.shape) for k in range(self._seg.shape[0])]
            for k, sg in enumerate(self._seg):
                shp[k][sid] = sg[1] - sg[0]
            return [tuple(sp) for sp in shp]

    @property
    def version(self):
        return self._version

    # Class "constructor"
    def __init__(self,
                 filename=None,
                 filetype=None,
                 trialdefinition=None,
                 label=None):
        """
        Docstring
        """

        # Depending on contents of `filename`, class instantiation invokes I/O routines
        read_fl = True
        if filename is None:
            read_fl = False

        # Prepare necessary "global" parsing attributes
        self._cfg = {}
        self._data = None
        self._dimlabels = OrderedDict()
        self._filename = self._gen_filename()
        self._seg = None
        self._segmentlabel = None
        self._mode = "w"

        # Create temporary working directory if not already present
        # tmpdir = os.path.join(os.path.expanduser("~"), __storage__)
        if not os.path.exists(__storage__):
            try:
                os.mkdir(__storage__)
            except:
                raise SPWIOError(__storage__)

        # Write version
        self._version = __version__

        # Write log-header information
        lhd = "\n\t\t>>> SpykeWave v. {ver:s} <<< \n\n" +\
              "Created: {timestamp:s} \n\n" +\
              "System Profile: \n" +\
              "{sysver:s} \n" +\
              "NumPy: {npver:s}\n" +\
              "SciPy: {spver:s}\n" +\
              "Dask:  {daver:s}\n\n" +\
              "--- LOG ---"
        self._log_header = lhd.format(ver=__version__,
                                      timestamp=time.asctime(),
                                      sysver=sys.version,
                                      npver=np.__version__,
                                      spver=sp.__version__,
                                      daver=dask.__version__ if __dask__ else "--")

        # Write initial log entry
        self._log = ""
        self.log = "created {clname:s} object".format(clname=self.__class__.__name__)

        # Finally call appropriate reading routine if filename was provided
        if read_fl:
            if label is None:
                label = "channel"
            sw.load_data(filename, filetype=filetype, label=label,
                         trialdefinition=trialdefinition, out=self)

    # Helper function that reads a single segment into memory
    @staticmethod
    def _copy_segment(segno, filename, seg, hdr, dimord, segmentlabel):
        idx = [slice(None)] * len(dimord)
        idx[dimord.index(segmentlabel)] = slice(int(seg[segno, 0]), int(seg[segno, 1]))
        idx = tuple(idx)
        if hdr is None:
            # For pre-processed npy files
            return np.array(open_memmap(filename, mode="c")[idx])
        else:
            # For VirtualData objects
            dsets = []
            for fk, fname in enumerate(filename):
                dsets.append(np.memmap(fname, offset=int(hdr[fk]["length"]),
                                       mode="r", dtype=hdr[fk]["dtype"],
                                       shape=(hdr[fk]["M"], hdr[fk]["N"]))[idx])
            return np.vstack(dsets)
        
    # Helper function that grabs a single segment from memory-map(s)
    def _get_segment(self, segno):
        idx = [slice(None)] * len(self.dimord)
        sid = self.dimord.index(self.segmentlabel)
        idx[sid] = slice(int(self._seg[segno, 0]), int(self._seg[segno, 1]))
        return self._data[tuple(idx)]

    # Helper function generating pseudo-random temp file-names
    @staticmethod
    def _gen_filename():
        fname_hsh = blake2b(digest_size=4, salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()
        return os.path.join(__storage__, "spy_{}.npy".format(fname_hsh))

    # Helper function that digs into cfg dictionaries
    def _set_cfg(self, cfg, dct):
        if not cfg:
            cfg = dct
        else:
            if "cfg" in cfg.keys():
                self._set_cfg(cfg["cfg"], dct)
            else:
                cfg["cfg"] = dct
                return cfg
        return cfg

    # Convenience function, wiping attached memmap or pointing to class-specific `clear` method
    def clear(self):
        if self._data is not None:
            if hasattr(self._data, "clear"):
                self._data.clear()
            else:
                filename, mode = self._data.filename, self._data.mode
                self._data.flush()
                self._data = None
                self._data = open_memmap(filename, mode=mode)
        return
    
    # Return a (deep) copy of the current class instance
    def copy(self, deep=False):
        return copy(self)

    # Selector method
    def select(self, segments=None, deepcopy=False, **kwargs):
        if not set(kwargs.keys()).issubset(self.dimord):
            raise SPWValueError(legal=self.dimord, actual=list(kwargs.keys()))
        if segments is None:
            segments = range(self.seg.shape[0])
        if not set(segments).issubset(range(self.seg.shape[0])):
            lgl = "segment selection between 0 and {}".format(str(self.seg.shape[0]))
            raise SPWValueError(legal=lgl, varname="segments")

        # Build multi-index for selection and warn in case shallow copy is not feasible
        idx = [slice(None)] * len(self.dimord)
        target_shape = list(self.data.shape)
        for lbl, selection in kwargs.items():
            id = self.dimord.index(lbl) 
            idx[id] = selection
            if isinstance(selection, slice):
                target_shape[id] = len(range(*selection.indices(self.data.shape[id])))
            elif isinstance(selection, int):
                target_shape[id] = 1
            else:
                if not deepcopy:
                    spw_warning("Shallow copy only possible for int or slice selectors",
                                caller="SpykeWave core:select")
                    deepcopy = True
                target_shape[id] = len(selection)

        # FIXME: renumber samples in segments!
        if deepcopy:
            sid = self.dimord.index(self.segmentlabel)
            target_shape[sid] = sum([shp[sid] for shp in np.array(self.shapes)[segments]])
            target = self.copy()
            target._filename = self._gen_filename()
            dat = open_memmap(target._filename, mode="w+",
                              dtype=self..data.dtype, shape=target_shape)
            del dat
            for sk, seg in enumerate(segments):
                dat = open_memmap(target._filename, mode="r+")[
            
            

    # Wrapper that makes saving routine usable as class method
    def save(self, out_name, filetype=None, **kwargs):
        """
        Docstring that mostly points to ``save_data``
        """
        sw.save_data(out_name, self, filetype=filetype, **kwargs)

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
        ppattrs.sort()

        # Construct string for pretty-printing class attributes
        hdstr = "SpykeWave {diminfo:s}{clname:s} object with fields\n\n"
        ppstr = hdstr.format(diminfo="'" + "' x '".join(dim for dim in self.dimord) \
                             + "' " if self.dimord else "",
                             clname=self.__class__.__name__)
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
        ppstr += "\nUse `.log` to see object history"
        return ppstr

    # Destructor
    def __del__(self):
        if __storage__ in self._filename and os.path.exists(self._filename):
            del self._data
            os.unlink(self._filename)

##########################################################################################
class AnalogData(BaseData):

    @property
    def hdr(self):
        return self._hdr
    
    @property
    def sampleinfo(self):
        return self._dimlabels.get("sample")
    
    @property
    def samplerate(self):
        return self._samplerate
    
    @property
    def segments(self):
        return Indexer(map(self._get_segment, range(self._seg.shape[0])),
                                self._seg.shape[0]) if self._seg is not None else None

    @property
    def time(self, unit="s"):
        converter = {"h": 1/360, "min": 1/60, "s" : 1, "ms" : 1e3, "ns" : 1e9}
        if not isinstance(unit, str):
            raise SPWTypeError(unit, varname="unit", expected="str")
        if unit not in converter.keys():
            raise SPWValueError("".join(opt + ", " for opt in converter.keys())[:-2],
                                varname="unit", actual=unit)
        return [np.arange(start, end)*converter[unit]/self._samplerate \
                for (start, end) in self.sampleinfo] if self._samplerate else None

    @property
    def trial(self):
        return self.segments

    @property
    def trialinfo(self):
        return self.seg
    
    # Constructor
    def __new__(cls, basedataobj=None, copy=True, **kwargs):

        # Either create new instance from scratch or copy/convert existing BaseData object
        if isinstance(basedataobj, BaseData):
            if copy:
                obj = basedataobj.copy()
                basedataobj.log = "copied self to create {} object".format(cls.__name__)
                msg = "created {new:s} object from {old:s} object"
            else:
                obj = basedataobj
                msg = "converted {old:s} object to {new:s} object"
            obj.log = msg.format(old=basedataobj.__class__.__name__,
                                 new=cls.__name__)
            obj.__class__ = cls
            return obj
        else:
            return super().__new__(cls)

    # Customizer
    def __init__(self, basedataobj=None, **kwargs):

        # If we're starting from scratch, call parent class initializer
        if basedataobj is None:
            super().__init__(**kwargs)

        # Set default values for necessary attributes (if not already set
        # by reading routine invoked in `BaseData`'s `__init__`)
        if not hasattr(self, "hdr"):
            self._hdr = None
            self._samplerate = None
            self.segmentlabel = "sample"

##########################################################################################
class SpectralData(BaseData):

    @property
    def samplerate(self):
        return self._samplerate
    
    @property
    def segments(self):
        return Indexer(map(self._get_segment, range(self._seg.shape[0])),
                                self._seg.shape[0]) if self._seg is not None else None

    @property
    def trialinfo(self):
        return self.seg
    
    # Constructor
    def __new__(cls, basedataobj=None, copy=True, **kwargs):

        # Either create new instance from scratch or copy/convert existing BaseData object
        if isinstance(basedataobj, BaseData):
            if copy:
                obj = basedataobj.copy()
                basedataobj.log = "copied self to create {} object".format(cls.__name__)
                msg = "created {new:s} object from {old:s} object"
            else:
                obj = basedataobj
                msg = "converted {old:s} object to {new:s} object"
            obj.log = msg.format(old=basedataobj.__class__.__name__,
                                 new=cls.__name__)
            obj.__class__ = cls
            return obj
        else:
            return super().__new__(cls)

    # Customizer
    def __init__(self, basedataobj=None, **kwargs):

        # If we're starting from scratch, call parent class initializer
        if basedataobj is None:
            super().__init__(**kwargs)

        # Set default values for necessary attributes (if not already set
        # by reading routine invoked in `BaseData`'s `__init__`)
        if not hasattr(self, "samplerate"):
            self._samplerate = None
            self.segmentlabel = "freq"
    
##########################################################################################
class VirtualData():

    # Pre-allocate slots here - this class is *not* meant to be expanded
    # and/or monkey-patched later on
    __slots__ = ["_M", "_N", "_shape", "_size", "_nrows", "_data", "_rows", "_dtype"]

    @property
    def dtype(self):
        return self._dtype

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

        Do not confuse chunks with segments: chunks refer to actual raw binary
        data-files on disk, thus, row- *and* col-numbers MUST match!
        """

        # First, make sure our one mandatary input argument does not contain
        # any unpleasant surprises
        if not isinstance(chunk_list, (list, np.memmap)):
            raise SPWTypeError(chunk_list, varname="chunk_list", expected="array_like")

        # Do not use ``spw_array_parser`` to validate chunks to not force-load memmaps
        try:
            shapes = [chunk.shape for chunk in chunk_list]
        except:
            raise SPWTypeError(chunk_list[0], varname="chunk in chunk_list",
                               expected="2d-array-like")
        if np.any([len(shape) != 2 for shape in shapes]):
            raise SPWValueError(legal="2d-array", varname="chunk in chunk_list")

        # Get row number per input chunk and raise error in case col.-no. does not match up
        shapes = [chunk.shape for chunk in chunk_list]
        if not np.array_equal([shape[1] for shape in shapes], [shapes[0][1]]*len(shapes)):
            raise SPWValueError(legal="identical number of samples per chunk",
                                varname="chunk_list")
        nrows = [shape[0] for shape in shapes]
        cumlen = np.cumsum(nrows)

        # Get hierarchically "highest" dtype of data present in `chunk_list`
        dtypes = []
        for chunk in chunk_list:
            dtypes.append(chunk.dtype)
        cdtype = np.max(dtypes)

        # Create list of "global" row numbers and assign "global" dimensional info
        self._nrows = nrows
        self._rows = [range(start, stop) for (start, stop) in zip(cumlen - nrows, cumlen)]
        self._M = cumlen[-1]
        self._N = chunk_list[0].shape[1]
        self._shape = (self._M, self._N)
        self._size = self._M*self._N
        self._dtype = cdtype
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
            data = []
            data.append(self._data[i1][row.start - self._rows[i1].start:, col])
            for i in range(i1 + 1, i2):
                data.append(self._data[i][:, col])
            data.append(self._data[i2][:row.stop - self._rows[i2].start, col])
            return np.vstack(data)

        # If start and stop are in the same chunk, return a view of the underlying memmap
        else:
            
            # Convert "global" row index to local chunk-based row-number (by subtracting offset)
            row = slice(row.start - self._rows[i1].start, row.stop - self._rows[i1].start)
            return self._data[i1][row,:][:,col]

    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make class contents comprehensible when viewed from the command line
    def __str__(self):
        ppstr = "{shape:s} element {name:s} object mapping {numfiles:s} file(s)"
        return ppstr.format(shape="[" + " x ".join([str(numel) for numel in self.shape]) + "]",
                            name=self.__class__.__name__,
                            numfiles=str(len(self._nrows)))

    # Free memory by force-closing resident memory maps
    def clear(self):
        shapes = []
        dtypes = []
        fnames = []
        offset = []
        for mmp in self._data:
            shapes.append(mmp.shape)
            dtypes.append(mmp.dtype)
            fnames.append(mmp.filename)
            offset.append(mmp.offset)
        self._data = []
        for k in range(len(fnames)):
            self._data.append(np.memmap(fnames[k], offset=offset[k],
                                             mode="r", dtype=dtypes[k],
                                             shape=shapes[k]))
        return
    
##########################################################################################
class Indexer():

    __slots__ = ["_iterobj", "_iterlen"]
    
    def __init__(self, iterobj, iterlen):
        """
        Make an iterable object subscriptable using itertools magic
        """
        self._iterobj = iterobj
        self._iterlen = iterlen

    def __iter__(self):
        return self._iterobj

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Number):
            try:
                spw_scalar_parser(idx, varname="idx", ntype="int_like",
                                  lims=[0, self._iterlen - 1])
            except Exception as exc:
                raise exc
            return next(islice(self._iterobj, idx, idx + 1))
        elif isinstance(idx, slice):
            start, stop = idx.start, idx.stop
            if idx.start is None:
                start = 0
            if idx.stop is None:
                stop = self._iterlen
            index = slice(start, stop, idx.step)
            if not(0 <= index.start < self._iterlen) or not (0 < index.stop <= self._iterlen):
                err = "value between {lb:s} and {ub:s}"
                raise SPWValueError(err.format(lb="0", ub=str(self._iterlen)),
                                    varname="idx", actual=str(index))
            return np.hstack(islice(self._iterobj, index.start, index.stop, index.step))
        elif isinstance(idx, (list, np.ndarray)):
            try:
                spw_array_parser(idx, varname="idx", ntype="int_like",
                                 lims=[0, self._iterlen], dims=1)
            except Exception as exc:
                raise exc
            return np.hstack([next(islice(self._iterobj, int(ix), int(ix + 1))) for ix in idx])
        else:
            raise SPWTypeError(idx, varname="idx", expected="int_like or slice")
    
    def __len__(self):
        return self._iterlen

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return "{} element iterable".format(self._iterlen)
