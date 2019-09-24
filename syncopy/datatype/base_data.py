# -*- coding: utf-8 -*-
# 
# SynCoPy BaseData abstract class + helper classes
# 
# Created: 2019-01-07 09:22:33
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-09-24 15:40:02>

# Builtin/3rd party package imports
import numpy as np
import getpass
import socket
import time
import sys
import os
import numbers
import inspect
import h5py
import scipy as sp
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from copy import copy
from datetime import datetime
from hashlib import blake2b
from itertools import islice
from numpy.lib.format import open_memmap, read_magic
import shutil

# Local imports
from .data_methods import definetrial
from syncopy.shared.parsers import (scalar_parser, array_parser, io_parser, 
                                    filename_parser, data_parser)
from syncopy.shared.errors import SPYTypeError, SPYValueError, SPYError
from syncopy import __version__, __storage__, __dask__, __sessionid__
if __dask__:
    import dask
import syncopy as spy

__all__ = ["StructDict"]


class BaseData(ABC):

    # Class properties that are written to JSON/HDF upon save
    _infoFileProperties = ("dimord", "_version", "_log", "cfg",)
    _hdfFileProperties =  ("dimord", "_version", "_log",)

    # Checksum algorithm used
    _checksum_algorithm = spy.__checksum_algorithm__.__name__
    
    # Dummy allocations of class attributes that are actually initialized in subclasses
    _mode = None
    
    @property
    def cfg(self):
        """Dictionary of previous operations on data"""
        return self._cfg

    @cfg.setter
    def cfg(self, dct):
        if not isinstance(dct, dict):
            raise SPYTypeError(dct, varname="cfg", expected="dictionary-like object")
        self._cfg = self._set_cfg(self._cfg, dct)
        
    @property
    def container(self):
        if self.data is not None:
            return filename_parser(self.filename)["container"]
    
    @property
    def data(self):
        """array-like object representing data without trials"""

        if getattr(self._data, "id", None) is not None:
            if self._data.id.valid == 0:
                lgl = "open HDF5 container"
                act = "backing HDF5 container {} has been closed"
                raise SPYValueError(legal=lgl, actual=act.format(self.filename),
                                    varname="data")
        return self._data
    
    @data.setter
    def data(self, in_data):

        # Dimension count is either determined by length of dimord or 2 in case
        # of `EventData` or `SpikeData`
        if any(["DiscreteData" in str(base) for base in self.__class__.__mro__]):
            ndim = 2
        else:
            ndim = len(self.dimord)
                
        # If input is a string, try to load memmap/HDF5 dataset
        if isinstance(in_data, str):
            try:
                fpath, fname = io_parser(in_data, varname="filename", isfile=True, exists=True)
            except Exception as exc:
                raise exc
            in_data = os.path.join(fpath, fname)  # ensure `in_data` is absolute path

            md = self.mode
            if md == "w":
                md = "r+"

            is_npy = False
            is_hdf = False
            try:
                with open(in_data, "rb") as fd:
                    read_magic(fd)
                is_npy = True
            except ValueError as exc:
                err = "NumPy memorymap: " + str(exc)
            try:
                h5f = h5py.File(in_data, mode=md)
                is_hdf = True
            except OSError as exc:
                err = "HDF5: " + str(exc)
            if not is_npy and not is_hdf:
                raise SPYValueError("accessible HDF5 container or memory-mapped npy-file",
                                    actual=err, varname="data")
            
            if is_hdf:
                h5keys = list(h5f.keys())
                idx = [h5keys.count(dclass) for dclass in spy.datatype.__all__ \
                       if not (inspect.isfunction(getattr(spy.datatype, dclass)))]
                if len(h5keys) !=1 and sum(idx) != 1:
                    lgl = "HDF5 container holding one data-object"
                    act = "HDF5 container holding {} data-objects"
                    raise SPYValueError(legal=lgl, actual=act.format(str(len(h5keys))), varname="data")
                if len(h5keys) == 1:
                    self._data = h5f[h5keys[0]]
                else:
                    self._data = h5f[spy.datatype.__all__[idx.index(1)]]
            if is_npy:
                self._data = open_memmap(in_data, mode=md)
            self.filename = in_data

        # If input is already a memmap/HDF5 dataset, check its dimensions
        elif isinstance(in_data, (np.memmap, h5py.Dataset)):
            if isinstance(in_data, h5py.Dataset):
                if in_data.id.valid == 0:
                    lgl = "open HDF5 container"
                    act = "backing HDF5 container is closed"
                    raise SPYValueError(legal=lgl, actual=act, varname="data")
                md = in_data.file.mode
                fn = in_data.file.filename
            else:
                md = in_data.mode
                fn = in_data.filename
            if in_data.ndim != ndim:
                lgl = "{}-dimensional data".format(ndim)
                act = "{}-dimensional HDF5 dataset or memmap".format(in_data.ndim)
                raise SPYValueError(legal=lgl, varname="data", actual=act)
            self.mode = md
            self.filename = os.path.abspath(fn)
            self._data = in_data
            
        # If input is an array, either fill existing data property
        # or create backing container on disk
        elif isinstance(in_data, np.ndarray):
            try:
                array_parser(in_data, varname="data", dims=ndim)
            except Exception as exc:
                raise exc
            if isinstance(self._data, (np.memmap, h5py.Dataset)):
                if self.mode == "r":
                    lgl = "HDF5 dataset/memmap with write or copy-on-write access"
                    act = "read-only memmap"
                    raise SPYValueError(legal=lgl, varname="mode", actual=act)
                if self.data.shape != in_data.shape:
                    lgl = "HDF5 dataset/memmap with shape {}".format(str(self.data.shape))
                    act = "data with shape {}".format(str(in_data.shape))
                    raise SPYValueError(legal=lgl, varname="data", actual=act)
                if self.data.dtype != in_data.dtype:
                    print("SyNCoPy core - data: WARNING >> Input data-type mismatch << ")
                self._data[...] = in_data
            else:
                self.filename = self._gen_filename()
                dsetname = self.__class__.__name__
                with h5py.File(self.filename, "w") as h5f:
                    h5f.create_dataset(dsetname, data=in_data)
                md = self.mode
                if md == "w":
                    md = "r+"
                self._data = h5py.File(self.filename, md)[dsetname]

        # If input is a `VirtualData` object, make sure the object class makes sense
        elif isinstance(in_data, VirtualData):
            if self.__class__.__name__ != "AnalogData":
                lgl = "(filename of) memmap or NumPy array"
                act = "VirtualData (only valid for `AnalogData` objects)"
                raise SPYValueError(legal=lgl, varname="data", actual=act)
            self._data = in_data
            self._filename = [dat.filename for dat in in_data._data]
            self.mode = "r"

        # Whatever type the input is, it's not supported
        else:
            msg = "(filename of) memmap, NumPy array or VirtualData object"
            raise SPYTypeError(in_data, varname="data", expected=msg)

        # In case we're working with a `DiscreteData` object, fill up samples
        if any(["DiscreteData" in str(base) for base in self.__class__.__mro__]):
            self._sample = np.unique(self.data[:,self.dimord.index("sample")])

        # In case we're working with an `AnalogData` object, tentatively fill up channel labels
        if any(["ContinuousData" in str(base) for base in self.__class__.__mro__]):
            channel = ["channel" + str(i + 1) for i in range(self.data.shape[self.dimord.index("channel")])]
            self.channel = np.array(channel)

        # In case we're working with an `EventData` object, fill up eventid's
        if self.__class__.__name__ == "EventData":
            self._eventid = np.unique(self.data[:,self.dimord.index("eventid")])

    @property
    def dimord(self):
        """list(str): ordered list of data dimension labels"""
        return self._dimord
    
    @dimord.setter
    def dimord(self, dims):
        if hasattr(self, "_dimord"):
            print("Syncopy core - dimord: Cannot change `dimord` of object. " +\
                  "Functionality currently not supported")
        # Canonical way to perform initial allocation of dimensional properties 
        # (`self._channel = None`, `self._freq = None` etc.)            
        self._dimord = list(dims)
        for dim in [dlabel for dlabel in dims if dlabel != "time"]:
            setattr(self, "_" + dim, None)
            
    @property
    def filename(self):
        # implicit support for multiple backing filenames: convert list to str
        if isinstance(self._filename, list):
            outname = "".join(fname + ", " for fname in self._filename)[:-2]
        else:
            outname = self._filename
        return outname
    
    @filename.setter
    def filename(self, fname):
        if not isinstance(fname, str):
            raise SPYTypeError(fname, varname="fname", expected="str")
        self._filename = str(fname)

    @property
    def log(self):
        """str: log of previous operations on data"""
        print(self._log_header + self._log)

    @log.setter
    def log(self, msg):
        if not isinstance(msg, str):
            raise SPYTypeError(msg, varname="log", expected="str")
        prefix = "\n\n|=== {user:s}@{host:s}: {time:s} ===|\n\n\t{caller:s}"
        clr = sys._getframe().f_back.f_code.co_name
        if clr.startswith("_") and not clr.startswith("__"):
            clr = clr[1:]
        self._log += prefix.format(user=getpass.getuser(),
                                   host=socket.gethostname(),
                                   time=time.asctime(),
                                   caller=clr + ": " if clr != "<module>" else "")\
            + msg

    @property
    def mode(self):
        """str: write mode for data, 'r' for read-only, 'w' for writable

        FIXME: append/replace with HDF5?
        """
        return self._mode
    
    @property
    def tag(self):
        if self.data is not None:
            return filename_parser(self.filename)["tag"]

    @mode.setter
    def mode(self, md):

        # Ensure input makes sense and we actually have permission to change
        # the data access mode
        if not isinstance(md, str):
            raise SPYTypeError(md, varname="mode", expected="str")
        options = ["r", "r+", "w", "c"]
        if md not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(lgl, varname="mode", actual=md)
        if isinstance(self.data, VirtualData):
            print("syncopy core - mode: WARNING >> Cannot change read-only " +
                  "access mode of VirtualData datasets << ")
            return

        # If data is already attached to the object, change its access mode
        # as requested (if `md` is actually any different from `self.mode`)
        # NOTE: prevent accidental data loss by not allowing mode = "w" in h5py
        if self.data is not None:
            if md == self._mode:
                return
            if md == "w":
                md = "r+"
            self.data.flush()
            if isinstance(self.data, np.memmap):
                self._data = None
                self._data = open_memmap(self.filename, mode=md)
            else:
                dsetname = self.data.name
                self._data.file.close()
                self._data = h5py.File(self.filename, mode=md)[dsetname]

        self._mode = md
        
    @property
    def _selection(self):
        """Data selection specified by :class:`Selector`"""
        return self._selector
    
    @_selection.setter
    def _selection(self, select):
        if select is None:
            self._selector = None
        else:
            self._selector = Selector(self, select)

    @property
    def sampleinfo(self):
        """nTrials x 3 :class:`numpy.ndarray` of [start, end, offset] sample indices"""
        return self._sampleinfo

    @sampleinfo.setter
    def sampleinfo(self, sinfo):
        if self.data is None:
            print("SyNCoPy core - sampleinfo: Cannot assign `sampleinfo` without data. "+\
                  "Please assing data first")
            return
        if any(["ContinuousData" in str(base) for base in self.__class__.__mro__]):
            scount = self.data.shape[self.dimord.index("time")]
        else:
            scount = np.inf
        try:
            array_parser(sinfo, varname="sampleinfo", dims=(None, 2), hasnan=False, 
                         hasinf=False, ntype="int_like", lims=[0, scount])
        except Exception as exc:
            raise exc
        self._sampleinfo = np.array(sinfo, dtype=int)

    @property
    def t0(self):
        """FIXME: should be hidden"""
        return self._t0

    @property
    def trials(self):
        """list-like array of trials"""
        return Indexer(map(self._get_trial, range(self.sampleinfo.shape[0])),
                       self.sampleinfo.shape[0]) if self.sampleinfo is not None else None
    @property
    def trialinfo(self):
        """nTrials x M :class:`numpy.ndarray` with numeric information about each trial

        Each trial can have M properties (condition, original trial no., ...) coded by 
        """
        return self._trialinfo

    @trialinfo.setter
    def trialinfo(self, trl):
        if self.data is None:
            print("SyNCoPy core - trialinfo: Cannot assign `trialinfo` without data. "+\
                  "Please assing data first")
            return
        try:
            array_parser(trl, varname="trialinfo", dims=(self.sampleinfo.shape[0], None))
        except Exception as exc:
            raise exc
        self._trialinfo = np.array(trl)


    # Selector method
    @abstractmethod
    def selectdata(self, trials=None, deepcopy=False, **kwargs):
        """
        Docstring mostly pointing to ``selectdata``
        """
        pass

    # Helper function that grabs a single trial
    @abstractmethod
    def _get_trial(self, trialno):
        pass

    # Convenience function, wiping contents of backing device from memory
    def clear(self):
        """Clear loaded data from memory

        Calls `flush` method of HDF5 dataset or memory map. Memory maps are
        deleted and re-instantiated.        

        """
        if self.data is not None:
            self.data.flush()
            if isinstance(self.data, np.memmap):
                filename, mode = self.data.filename, self.data.mode
                self._data = None
                self._data = open_memmap(filename, mode=mode)
        return

    # Return a (deep) copy of the current class instance
    def copy(self, deep=False):
        """Create a copy of the data object in memory.

        Parameters
        ----------
            deep : bool
                If `True`, a copy of the underlying data file is created in the temporary Syncopy folder

        Returns
        -------
            BaseData
                in-memory copy of BaseData object

        See also
        --------
        save_spy

        """
        cpy = copy(self)
        if deep and isinstance(self.data, (np.memmap, h5py.Dataset)):
            self.data.flush()
            filename = self._gen_filename()
            shutil.copyfile(self.filename, filename)
            cpy.data = filename
        return cpy

    # Change trialdef of object
    def definetrial(self, trl=None, pre=None, post=None, start=None,
                    trigger=None, stop=None, clip_edges=False):
        """(Re-)define trials for data

        See also
        --------
        syncopy.definetrial

        """
        definetrial(self, trialdefinition=trl, pre=pre, post=post,
                    start=start, trigger=trigger, stop=stop,
                    clip_edges=clip_edges)


    # Wrapper that makes saving routine usable as class method
    def save(self, container=None, tag=None, filename=None, overwrite=False, memuse=100):
        """Save data object as new ``spy`` HDF container to disk (:func:`syncopy.save_data`)
        
        FIXME: update docu
        
        Parameters
        ----------                    
            container : str
                Path to Syncopy container folder (\*.spy) to be used for saving. If 
                omitted, a .spy extension will be added to the folder name.
            tag : str
                Tag to be appended to container basename
            filename :  str
                Explicit path to data file. This is only necessary if the data should
                not be part of a container folder. An extension (\*.<dataclass>) will
                be added if omitted. The `tag` argument is ignored.      
            overwrite : bool
                If `True` an existing HDF5 file and its accompanying JSON file is 
                overwritten (without prompt). 
            memuse : scalar 
                 Approximate in-memory cache size (in MB) for writing data to disk
                 (only relevant for :class:`VirtualData` or memory map data sources)

        Examples
        --------    
        >>> save_spy(obj, filename="session1")
        >>> # --> os.getcwd()/session1.<dataclass>
        >>> # --> os.getcwd()/session1.<dataclass>.info

        >>> save_spy(obj, filename="/tmp/session1")
        >>> # --> /tmp/session1.<dataclass>
        >>> # --> /tmp/session1.<dataclass>.info

        >>> save_spy(obj, container="container.spy")
        >>> # --> os.getcwd()/container.spy/container.<dataclass>
        >>> # --> os.getcwd()/container.spy/container.<dataclass>.info

        >>> save_spy(obj, container="/tmp/container.spy")
        >>> # --> /tmp/container.spy/container.<dataclass>
        >>> # --> /tmp/container.spy/container.<dataclass>.info

        >>> save_spy(obj, container="session1.spy", tag="someTag")
        >>> # --> os.getcwd()/container.spy/session1_someTag.<dataclass>
        >>> # --> os.getcwd()/container.spy/session1_someTag.<dataclass>.info

        """
        
        # Ensure `obj.save()` simply overwrites on-disk representation of object
        if container is None and tag is None and filename is None:
            if self.container is None:
                raise SPYError("Cannot create spy container in temporary " +\
                               "storage {} - please provide explicit path. ".format(__storage__))
            overwrite = True
            filename = self.filename
            
        # Support `obj.save(tag="newtag")`            
        if container is None and filename is None:
            if self.container is None:
                raise SPYError("Object is not associated to an existing spy container - " +\
                               "please save object first using an explicit path. ")
            container = filename_parser(self.filename)["folder"]
            
        spy.save(self, filename=filename, container=container, tag=tag, 
                 overwrite=overwrite, memuse=memuse)

    # Helper function generating pseudo-random temp file-names    
    def _gen_filename(self):
        fname_hsh = blake2b(digest_size=4, 
                            salt=os.urandom(blake2b.SALT_SIZE)).hexdigest()
        return os.path.join(__storage__,
                            "spy_{sess:s}_{hash:s}{ext:s}".format(
                                sess=__sessionid__, hash=fname_hsh,
                                ext=self._classname_to_extension()))

    # Helper function converting object class-name to usable file extension
    def _classname_to_extension(self):
        return "." + self.__class__.__name__.split('Data')[0].lower()

    # Helper function that digs into cfg dictionaries
    def _set_cfg(self, cfg, dct):
        dct = StructDict(dct)
        if not cfg:
            cfg = dct
        else:
            if "cfg" in cfg.keys():
                self._set_cfg(cfg["cfg"], dct)
            else:
                cfg["cfg"] = dct
                return cfg
        return cfg

    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make class contents readable from the command line
    def __str__(self):

        # Get list of print-worthy attributes
        ppattrs = [attr for attr in self.__dir__()
                   if not (attr.startswith("_") or attr in ["log", "t0"])]
        ppattrs = [attr for attr in ppattrs
                   if not (inspect.ismethod(getattr(self, attr))
                           or isinstance(getattr(self, attr), Iterator))]
        if hasattr(self, "hdr"):
            if getattr(self, "hdr") is None:
                ppattrs.remove("hdr")
        ppattrs.sort()

        # Construct string for pretty-printing class attributes
        if self.__class__.__name__ == "SpikeData":
            dinfo = " 'spike' x "
            dsep = "'-'"
        elif self.__class__.__name__ == "EventData":
            dinfo = " 'event' x "
            dsep = "'-'"
        else:
            dinfo = ""
            dsep = "' x '"
        hdstr = "SyNCoPy{diminfo:s}{clname:s} object with fields\n\n"
        ppstr = hdstr.format(diminfo=dinfo + " '"  + \
                             dsep.join(dim for dim in self.dimord) + "' ",
                             clname=self.__class__.__name__)
        maxKeyLength = max([len(k) for k in ppattrs])
        for attr in ppattrs:
            value = getattr(self, attr)
            if hasattr(value, 'shape') and attr == "data" and self.sampleinfo is not None:
                tlen = np.unique([sinfo[1] - sinfo[0] for sinfo in self.sampleinfo])
                if tlen.size == 1:
                    trlstr = "of length {} ".format(str(tlen[0]))
                else:
                    trlstr = ""
                dsize = np.prod(self.data.shape)*self.data.dtype.itemsize/1024**2
                dunit = "MB"
                if dsize > 1000:
                    dsize /= 1024
                    dunit = "GB"
                valueString = "{} trials {}defined on ".format(str(len(self.trials)), trlstr)
                valueString += "[" + " x ".join([str(numel) for numel in value.shape]) \
                              + "] {dt:s} {tp:s} " +\
                              "of size {sz:3.2f} {szu:s}"
                valueString = valueString.format(dt=self.data.dtype.name,
                                                 tp=self.data.__class__.__name__,
                                                 sz=dsize,
                                                 szu=dunit)
            elif hasattr(value, 'shape'):
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
            printString = "{0:>" + str(maxKeyLength + 5) + "} : {1:}\n"
            ppstr += printString.format(attr, valueString)
        ppstr += "\nUse `.log` to see object history"
        return ppstr

    # Destructor
    def __del__(self):
        if self.filename is not None:
            if isinstance(self._data, h5py.Dataset):
                try:
                    self._data.file.close()
                except:
                    pass
            else:
                del self._data
            if __storage__ in self.filename and os.path.exists(self.filename):
                os.unlink(self.filename)
                shutil.rmtree(os.path.splitext(self.filename)[0],
                              ignore_errors=True)

    # Class "constructor"
    def __init__(self, data=None, filename=None, dimord=None, mode="r+", **kwargs):
        """
        Docstring

        filename + data = create memmap @filename
        filename no data = read from file or memmap
        just data = try to attach data (error checking done by data.setter)
        """

        # First things first: initialize (dummy) default values
        self._cfg = {}
        self._data = None
        self.mode = mode
        self._selector = None
        self._sampleinfo = None
        self._t0 = [None]
        self._trialinfo = None
        self._filename = None
        
        # Set up dimensional architecture (`self._channel = None`, `self._freq = None` etc.)
        self.dimord = dimord

        # Depending on contents of `filename` and `data` class instantiation invokes I/O routines
        if filename is not None:

            # Case 1: filename + data = memmap @filename
            if data is not None:
                read_fl = False
                self.data = filename
                self.data = data

            # Case 2: filename w/o data = read from file/container
            else:
                read_fl = False
                try:
                    fileinfo = filename_parser(filename)
                    if fileinfo["filename"] is not None:
                        read_fl = True
                except:
                    pass
                if not read_fl:
                    self.data = filename
                    
        else:

            # Case 3: just data = if str, it HAS to be the name of a spy-file
            if data is not None:
                if isinstance(data, str):
                    try:
                        fileinfo = filename_parser(data)
                    except Exception as exc:
                        raise exc
                    if fileinfo["filename"] is None:
                        lgl = "explicit file-name to initialize object"
                        raise SPYValueError(legal=lgl, actual=data)
                    read_fl = True
                    filename = data
                else:
                    read_fl = False
                    self.data = data

            # Case 4: nothing here: create empty object
            else:
                read_fl = False
                self._filename = self._gen_filename()
        
        # Warn on effectless assignments
        if read_fl:
            msg = "Syncopy core - __init__: WARNING >> Cannot assign `{}` to object " +\
                  "loaded from spy container << "                
            for key, value in kwargs.items():
                if value is not None:
                    print(msg.format(key))
            
        # Prepare log + header and write first entry
        lhd = "\n\t\t>>> SyNCopy v. {ver:s} <<< \n\n" +\
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
        self._log = ""
        self.log = "created {clname:s} object".format(clname=self.__class__.__name__)

        # Write version
        self._version = __version__

        # Finally call spy loader if filename was provided
        if read_fl:
            spy.load(filename=filename, out=self)

        # Make instantiation persistent in all subclasses
        super().__init__()

        
class VirtualData():
    """Class for handling 2D-data spread across multiple files

    Arrays from individual files (chunks) are concatenated along 
    the 2nd dimension (dim=1).

    """

    # Pre-allocate slots here - this class is *not* meant to be expanded
    # and/or monkey-patched at runtime
    __slots__ = ["_M", "_N", "_shape", "_size", "_ncols", "_data", "_cols", "_dtype"]

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

        Do not confuse chunks with trials: chunks refer to actual raw binary
        data-files on disk, thus, row- *and* col-numbers MUST match!
        """

        # First, make sure our one mandatory input argument does not contain
        # any unpleasant surprises
        if not isinstance(chunk_list, (list, np.memmap)):
            raise SPYTypeError(chunk_list, varname="chunk_list", expected="array_like")

        # Do not use ``array_parser`` to validate chunks to not force-load memmaps
        try:
            shapes = [chunk.shape for chunk in chunk_list]
        except:
            raise SPYTypeError(chunk_list[0], varname="chunk in chunk_list",
                               expected="2d-array-like")
        if np.any([len(shape) != 2 for shape in shapes]):
            raise SPYValueError(legal="2d-array", varname="chunk in chunk_list")

        # Get row number per input chunk and raise error in case col.-no. does not match up
        shapes = [chunk.shape for chunk in chunk_list]
        if not np.array_equal([shape[0] for shape in shapes], [shapes[0][0]] * len(shapes)):
            raise SPYValueError(legal="identical number of samples per chunk",
                                varname="chunk_list")
        ncols = [shape[1] for shape in shapes]
        cumlen = np.cumsum(ncols)

        # Get hierarchically "highest" dtype of data present in `chunk_list`
        dtypes = []
        for chunk in chunk_list:
            dtypes.append(chunk.dtype)
        cdtype = np.max(dtypes)

        # Create list of "global" row numbers and assign "global" dimensional info
        self._ncols = ncols
        self._cols = [range(start, stop) for (start, stop) in zip(cumlen - ncols, cumlen)]
        self._M = chunk_list[0].shape[0]
        self._N = cumlen[-1]
        self._shape = (self._M, self._N)
        self._size = self._M * self._N
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
                scalar_parser(qrow, varname="row", ntype="int_like", lims=[0, self._M])
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
            raise SPYTypeError(qrow, varname="row", expected="int_like or slice")

        # Convert input to slice (if it isn't already) or assign explicit start/stop values
        if isinstance(qcol, numbers.Number):
            try:
                scalar_parser(qcol, varname="col", ntype="int_like", lims=[0, self._N])
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
            raise SPYTypeError(qcol, varname="col", expected="int_like or slice")

        # Make sure queried row/col are inside dimensional limits
        err = "value between {lb:s} and {ub:s}"
        if not(0 <= row.start < self._M) or not(0 < row.stop <= self._M):
            raise SPYValueError(err.format(lb="0", ub=str(self._M)),
                                varname="row", actual=str(row))
        if not(0 <= col.start < self._N) or not(0 < col.stop <= self._N):
            raise SPYValueError(err.format(lb="0", ub=str(self._N)),
                                varname="col", actual=str(col))

        # The interesting part: find out wich chunk(s) `col` is pointing at
        i1 = np.where([col.start in chunk for chunk in self._cols])[0].item()
        i2 = np.where([(col.stop - 1) in chunk for chunk in self._cols])[0].item()

        # If start and stop are not within the same chunk, data is loaded into memory
        if i1 != i2:
            data = []
            data.append(self._data[i1][row, col.start - self._cols[i1].start:])
            for i in range(i1 + 1, i2):
                data.append(self._data[i][row, :])
            data.append(self._data[i2][row, :col.stop - self._cols[i2].start])
            return np.hstack(data)

        # If start and stop are in the same chunk, return a view of the underlying memmap
        else:

            # Convert "global" row index to local chunk-based row-number (by subtracting offset)
            col = slice(col.start - self._cols[i1].start, col.stop - self._cols[i1].start)
            return self._data[i1][:, col][row, :]

    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make class contents comprehensible when viewed from the command line
    def __str__(self):
        ppstr = "{shape:s} element {name:s} object mapping {numfiles:s} file(s)"
        return ppstr.format(shape="[" + " x ".join([str(numel) for numel in self.shape]) + "]",
                            name=self.__class__.__name__,
                            numfiles=str(len(self._ncols)))

    # Free memory by force-closing resident memory maps
    def clear(self):
        """Clear read data from memory

        Reinstantiates memory maps of all open files.

        """
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

    # Ensure compatibility b/w `VirtualData`, HDF5 datasets and memmaps
    def flush(self):
        self.clear()


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
                scalar_parser(idx, varname="idx", ntype="int_like",
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
                raise SPYValueError(err.format(lb="0", ub=str(self._iterlen)),
                                    varname="idx", actual=str(index))
            return np.hstack(islice(self._iterobj, index.start, index.stop, index.step))
        elif isinstance(idx, (list, np.ndarray)):
            try:
                array_parser(idx, varname="idx", ntype="int_like", hasnan=False,
                             hasinf=False, lims=[0, self._iterlen], dims=1)
            except Exception as exc:
                raise exc
            return np.hstack([next(islice(self._iterobj, int(ix), int(ix + 1))) for ix in idx])
        else:
            raise SPYTypeError(idx, varname="idx", expected="int_like or slice")

    def __len__(self):
        return self._iterlen

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{} element iterable".format(self._iterlen)

    
class SessionLogger():

    __slots__ = ["sessionfile", "_rm"]

    def __init__(self):
        sess_log = "{user:s}@{host:s}: <{time:s}> started session {sess:s}"
        self.sessionfile = os.path.join(__storage__,
                                        "session_{}_log.id".format(__sessionid__))
        with open(self.sessionfile, "w") as fid:
            fid.write(sess_log.format(user=getpass.getuser(),
                                      host=socket.gethostname(),
                                      time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                      sess=__sessionid__))
        self._rm = os.unlink # workaround to prevent Python from garbage-collectiing ``os.unlink``

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Session {}".format(__sessionid__)

    def __del__(self):
        self._rm(self.sessionfile)


class StructDict(dict):
    """Child-class of dict for emulating MATLAB structs

    Examples
    --------
    cfg = StructDict()
    cfg.a = [0, 25]

    """
    
    def __init__(self, *args, **kwargs):
        """
        Create a child-class of dict whose attributes are its keys
        (thus ensuring that attributes and items are always in sync)
        """
        super().__init__(*args, **kwargs)
        self.__dict__ = self        


class FauxTrial():
    """
    Stand-in mockup of NumPy arrays representing trial data
    
    Parameters
    ----------
    shape : tuple
        Shape of source trial array 
    idx : tuple
        Tuple of slices for extracting trial-data from source object's `data`
        dataset. The provided tuple **has** to be a proper indexing sequence, 
        i.e., if `idx` refers to the `k`-th trial in `obj`, then ``obj.data[idx]``
        must slice `data` correctly so that ``obj.data[idx] == obj.trials[k]``
    dtype : :class:`numpy.dtype`
        Datatype of source trial array
        
    Returns
    -------
    faux_trl : FauxTrial object
        An instance of `FauxTrial` that essentially parrots :class:`numpy.ndarray`
        objects and can, thus, be used to feed "fake" trials into a 
        :meth:`syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
        to get the `noCompute` runs out of the way w/o actually loading trials 
        into memory. 
        
    See also
    --------
    syncopy.continuous_data.ContinuousData._preview_trial : makes use of this class
    """
    
    def __init__(self, shape, idx, dtype):
        self.shape = tuple(shape)
        self.idx = tuple(idx)
        self.dtype = dtype
        
    def __str__(self):
        msg = "Trial placeholder of shape {} and datatype {}"
        return msg.format(str(self.shape), str(self.dtype))

    def __repr__(self):
        return self.__str__()

    def squeeze(self):
        """
        Remove 1's from shape and return a new `FauxTrial` instance 
        (parroting the NumPy original :func:`numpy.squeeze`)
        """
        shp = list(self.shape)
        while 1 in shp:
            shp.remove(1)
        return FauxTrial(shp, self.idx, self.dtype)

    @property
    def T(self):
        """
        Return a new `FauxTrial` instance with reversed dimensions
        (parroting the NumPy original :func:`numpy.transpose`)
        """
        return FauxTrial(self.shape[::-1], self.idx[::-1], self.dtype)


class Selector():
    """
    Auxiliary class for data selection

    Parameters
    ----------
    data : Syncopy data object
        A non-empty Syncopy data object
    select : dict or :class:`StructDict` or None
        Python dictionary or Syncopy :class:`StructDict` formatted for
    data selection. Supported keys are

    * 'trials' : list of integers
      trial numbers to be selected; can include repetitions and need not
      be sorted (e.g., ``trials = [0, 1, 0, 0, 2]`` is valid) but must
      be finite and not NaN. 
    * 'channels' : list (integers or strings), slice or range
      channel-specification; can be a list of channel names
      (``['channel3', 'channel1']``), a list of channel indices (``[3, 5]``),
      slice (``slice(3, 10)``) or range (``range(3, 10)``). Note that
      channels are counted starting at zero, i.e., ``channels = [0, 1, 2]``
      or ``channels = slice(0, 3)`` selects the first up to (and including)
      the third channel. Selections can be unsorted and may include
      repetitions but must match exactly, be finite and not NaN. 
    * 'toi' : list
      time-points to be selected (in seconds) in each trial. Timing is
      expected to be on a by-trial basis (e.g., relative to trigger onsets). 
      Selections can be approximate, unsorted and may include repetitions
      but must be finite and not NaN. Fuzzy matching is performed for
      approximate selections (i.e., selected time-points are close but not
      identical to timing information found in `data`) using a nearest-
      neighbor search for elements of `toi` in `data.time`. 
    * 'toilim' : list
      time-window ``[tmin, tmax]`` (in seconds) to be extracted from
      each trial. Window specifications must be sorted (e.g., ``[2.2, 1.1]``
      is invalid) and not NaN but may be unbounded (e.g., ``[1.1, np.inf]``
      is valid).
    * 'foi' : list
      frequencies to be selected (in Hz). Selections can be approximate,
      unsorted and may include repetitions but must be finite and not NaN.
      Fuzzy matching is performed for approximate selections (i.e., selected
      frequencies are close but not identical to frequencies found in
      `data`) using a nearest-neighbor search for elements of `foi` in
      `data.freq`. 
    * 'foilim' : list
      frequency-window ``[fmin, fmax]`` (in Hz) to be extracted. Window
      specifications must be sorted (e.g., ``[90, 70]`` is invalid) and
      not NaN but may be unbounded (e.g., ``[-np.inf, 60.5]`` is valid).
    * 'tapers' : list (integers or strings), slice or range
      taper-specification; can be a list of taper names
      (``['dpss-win-1', 'dpss-win-3']``), a list of taper indices
      (``[3, 5]``), slice (``slice(3, 10)``) or range (``range(3, 10)``).
      Note that tapers are counted starting at zero, i.e., ``tapers = [0, 1, 2]``
      or ``tapers = slice(0, 3)`` selects the first up to (and including)
      the third taper. Selections can be unsorted and may include
      repetitions but must match exactly, be finite and not NaN. 
    * 'units' : list (integers or strings), slice or range
      unit-specification; can be a list of unit names
      (``['unit10', 'unit3']``), a list of unit indices (``[3, 5]``),
      slice (``slice(3, 10)``) or range (``range(3, 10)``). Note that
      units are counted starting at zero, i.e., ``units = [0, 1, 2]``
      or ``units = slice(0, 3)`` selects the first up to (and including)
      the third unit. Selections can be unsorted and may include
      repetitions but must match exactly, be finite and not NaN.
    * 'eventids' : list of integers, slice or range
      event-id-specification; can be a list of event-id codes (``[2, 0, 1]``),
      slice (``slice(0, 2)``) or range (``range(0, 2)``). Selections can
      be unsorted and may include repetitions but must match exactly, be
      finite and not NaN.

    Any property of `data` that is not specifically accessed via one of
    the above keys is taken as is, e.g., ``select = {'trials': [1, 2]}``
    selects the entire contents of trials no. 2 and 3, while
    ``select = {'channels': range(0, 50)}`` selects the first 50 channels
    of `data` across all defined trials. Consequently, if `select` is
    `None`, the entire contents of `data` is selected. 
    
    Returns
    -------
    selection : Syncopy :class:`Selector` object
        An instance of this class whose properties are either lists or slices
        to be used as (fancy) indexing tuples. Note that the properties `time`, 
        `unit` and `eventid` are **by-trial** selections, i.e., list of lists 
        and/or slices encoding per-trial sample-indices, e.g., ``selection.time[0]`` 
        is intended to be used with ``data.trials[selection.trials[0]]``. 

    Notes
    -----
    Whenever possible, this class performs extensive input parsing to ensure
    consistency of provided selectors. Some exceptions to this rule include
    `toi` and `toilim`: depending on the size of `data` and the number of
    defined trials, `data.time` might be a list of arrays of substantial
    size. To not overflow memory and slow down computations, neither `toi`
    nor `toilim` is checked for consistency with respect to `data.time`, i.e.,
    the code does not verify that min/max of `toi`/`toilim` are within the
    bounds of `data.time` for each selected trial.

    For objects that have a `time` property, a suitable new `trialdefinition`
    array (accessible via the identically named `Selector` class property)
    is automatically constructed based on the provided selection. For unsorted
    time-selections with or without repetitions, the `timepoints` property
    encodes the timing of the selected (discrete) points. To permit this
    functionality, the input object's samplerate is stored in the identically
    named hidden attribute `_samplerate`. In addition, the hidden `_timeShuffle`
    attribute is a binary flag encoding whether selected time-points are
    unordered and/or contain repetitions (`Selector._timeShuffle = True`).

    By default, each selection property tries to convert a user-provided
    selection to a contiguous slice-indexer so that simple NumPy array
    indexing can be used for best performance. However, after setting all
    selection indices appropriate for the input object, a consistency
    check is performed by :meth:`_make_consistent` to ensure that the
    calculated indices can actually be jointly used on a multi-dimensional
    NumPy array without violating indexing arithmetic. Thus, if a given
    Selector instance ends up containing more than two conjoint index-lists,
    all other selection properties are converted (if necessary) to lists as well
    for use with :func:`numpy.ix_`. These selections require special array
    manipulation techniques (colloquially referred to as "fancy" or "advanced"
    indexing) and the :class:`Selector` marks such indexers by setting the
    hidden `self._useFancy` attribute to `True`. Note that :func:`numpy.ix_`
    always creates copies of the indexed reference array, hence, the attempt
    to use slice-based indexing whenever possible. 

    Examples
    --------
    See :func:`syncopy.selectdata` for usage examples.

    See also
    --------
    syncopy.selectdata : extract data selections from Syncopy objects
    """
    
    def __init__(self, data, select):
        
        # Ensure input makes sense
        try:
            data_parser(data, varname="data", empty=False)
        except Exception as exc:
            raise exc
        if select is None:
            select = {}
        if not isinstance(select, dict):
            raise SPYTypeError(select, "select", expected="dict")
        supported = ["trials", "channels", "toi", "toilim", "foi", "foilim",
                     "tapers", "units", "eventids"]
        if not set(select.keys()).issubset(supported):
            lgl = "dict with one or all of the following keys: '" +\
                  "'".join(opt + "', " for opt in supported)[:-2]
            act = "dict with keys '" +\
                  "'".join(key + "', " for key in select.keys())[:-2]
            raise SPYValueError(legal=lgl, varname="select", actual=act)
        
        # Save class of input object for posterity
        self._dataClass = data.__class__.__name__
        
        # Set up lists of (a) all selectable properties and (b) trial-dependent ones
        self._allProps = ["channel", "time", "freq", "taper", "unit", "eventid"]
        self._byTrialProps = ["time", "unit", "eventid"]
        
        # Assign defaults (trials are not a "real" property, handle it separately, 
        # same goes for `trialdefinition`)
        self._trials = None
        self._trialdefinition = None
        for prop in self._allProps:
            setattr(self, "_{}".format(prop), None)
        self._useFancy = False  # flag indicating whether fancy indexing is necessary
        self._samplerate = None  # for objects supporting time-selections
        self._timeShuffle = False  # flag indicating whether time-points are repeated/unordered
        
        # We first need to know which trials are of interest here (assuming 
        # that any valid input object *must* have a `trials` attribute)
        self.trials = (data, select)

        # Now set any possible selection attribute (depending on type of `data`)
        # Note: `trialdefinition` is set by `time.setter` - it only makes sense
        # for objects that have a `time` property to update `trialdefinition`
        for prop in self._allProps:
            setattr(self, prop, (data, select))
        
        # Ensure correct indexing: convert everything to lists for use w/`np.ix_`
        # if we ended up w/more than 2 list selectors
        self._make_consistent(data)
        
    @property
    def trials(self):
        """Index list of selected trials"""
        return self._trials
    
    @trials.setter
    def trials(self, dataselect):
        data, select = dataselect
        trlList = list(range(len(data.trials)))
        trials = select.get("trials", trlList)
        vname = "select: trials"
        try:
            array_parser(trials, varname=vname, ntype="int_like", hasinf=False,
                         hasnan=False, lims=[0, len(data.trials)], dims=1)
        except Exception as exc:
            raise exc
        if not set(trials).issubset(trlList):
            lgl = "List/array of values b/w 0 and {}".format(trlList[-1])
            act = "Values b/w {} and {}".format(min(trials), max(trials))
            raise SPYValueError(legal=lgl, varname=vname, actual=act)
        self._trials = trials
        
    @property
    def channel(self):
        """List or slice encoding channel-selection"""
        return self._channel
    
    @channel.setter
    def channel(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "channel", "channels")
        
    @property
    def time(self):
        """List of lists/slices of by-trial time-selections"""
        return self._time
    
    @time.setter
    def time(self, dataselect):
        
        # Unpack input and perform error-checking
        data, select = dataselect
        timeSpec = select.get("toi")
        checkLim = False
        checkInf = False
        vname = "select: toi/toilim"
        if timeSpec is None:
            timeSpec = select.get("toilim")
            checkLim = True
            checkInf = None
        else:
            if select.get("toilim") is not None:
                lgl = "either `toi` or `toilim` specification"
                act = "both"
                raise SPYValueError(legal=lgl, varname=vname, actual=act)
        hasTime = hasattr(data, "time") or hasattr(data, "trialtime")
        if timeSpec is not None and hasTime is False:
            lgl = "Syncopy data object with time-dimension"
            raise SPYValueError(legal=lgl, varname=vname, actual=data.__class__.__name__)

        # If `data` has a `time` property, fill up `self.time`
        if hasTime:
            if timeSpec is not None:
                try:
                    array_parser(timeSpec, varname=vname, hasinf=checkInf, hasnan=False, dims=1)
                except Exception as exc:
                    raise exc
                if checkLim:
                    if len(timeSpec) != 2:
                        lgl = "`select: toilim` selection with two components"
                        act = "`select: toilim` with {} components".format(len(timeSpec))
                        raise SPYValueError(legal=lgl, varname=vname, actual=act)
                    if timeSpec[0] >= timeSpec[1]:
                        lgl = "`select: toilim` selection with `toilim[0]` < `toilim[1]`"
                        act = "selection range from {} to {}".format(timeSpec[0], timeSpec[1])
                        raise SPYValueError(legal=lgl, varname=vname, actual=act)
            timing = data._get_time(self.trials, toi=select.get("toi"), toilim=select.get("toilim"))
            
            # Determine, whether time-selection is unordered/contains repetitions
            # and set `self._timeShuffle` accordingly
            if timeSpec is not None:
                for tsel in timing:
                    if isinstance(tsel, list):
                        if len(tsel) > 1:
                            if np.diff(tsel).min() <= 0:
                                self._timeShuffle = True
                                break 

            # Assign timing selection and copy over samplerate from source object
            self._time = timing
            self._samplerate = data.samplerate
            
            # Prepare new `trialdefinition` array corresponding to selection
            self.trialdefinition = data
        else:
            return

    @property
    def trialdefinition(self):
        """N x 3+ :class:`numpy.ndarray` encoding trial-information of selection"""
        return self._trialdefinition

    @trialdefinition.setter
    def trialdefinition(self, data):
        
        # Get original `trialdefinition` array for reference
        # FIXME: obsolte w/new trialdefinition arrays
        # trl = data.trialdefinition
        trl = np.array(data.trialinfo)
        t0 = np.array(data.t0).reshape((data.t0.size, 1))
        trl = np.hstack([data.sampleinfo, t0, trl])

        # Build new trialdefinition array using `t0`-offsets        
        trlDef = np.zeros((len(self.trials), trl.shape[1]))
        counter = 0
        for tk, trlno in enumerate(self.trials):
            tsel = self.time[tk]
            if isinstance(tsel, slice):
                start, stop, step = tsel.start, tsel.stop, tsel.step
                if start is None:
                    start = 0
                if stop is None:
                    trlTime = data._get_time([trlno], toilim=[-np.inf, np.inf])[0]
                    if isinstance(trlTime, list):
                        stop = np.max(trlTime)
                    else:
                        stop = trlTime.stop
                if step is None:
                    step = 1
                nSamples = (stop - start)/step
                endSample = stop + data._t0[tk]
                t0 = int(endSample - nSamples)
            else:
                nSamples = len(tsel)
                if nSamples == 0:
                    t0 = 0
                else:
                    t0 = data._t0[tk]
            trlDef[tk, :3] = [counter, counter + nSamples, t0]
            counter += nSamples
        self._trialdefinition = trlDef
        
    @property
    def timepoints(self):
        """len(self.trials) list of lists encoding timing information of unordered `toi` selections"""
        if self._timeShuffle:
            return [[(tvec[tp] + self.trialdefinition[tk, 2]) / self._samplerate 
                     for tp in range(len(tvec))] for tk, tvec in enumerate(self.time)]

    @property
    def freq(self):
        """List or slice encoding frequency-selection"""
        return self._freq
    
    @freq.setter
    def freq(self, dataselect):
        
        # Unpack input and perform error-checking
        data, select = dataselect
        freqSpec = select.get("foi")
        checkLim = False
        checkInf = False
        vname = "select: foi/foilim"
        if freqSpec is None:
            freqSpec = select.get("foilim")
            checkLim = True
            checkInf = None
        else:
            if select.get("foilim") is not None:
                lgl = "either `foi` or `foilim` specification"
                act = "both"
                raise SPYValueError(legal=lgl, varname=vname, actual=act)
        hasFreq = hasattr(data, "freq")
        if freqSpec is not None and hasFreq is False:
            lgl = "Syncopy data object with freq-dimension"
            raise SPYValueError(legal=lgl, varname=vname, actual=data.__class__.__name__)
        
        # If `data` has a `freq` property, fill up `self.freq`
        if hasFreq:
            if freqSpec is not None:
                try:
                    array_parser(freqSpec, varname=vname, hasinf=checkInf, hasnan=False, 
                                lims=[data.freq.min(), data.freq.max()], dims=1)
                except Exception as exc:
                    raise exc
                if checkLim:
                    if len(freqSpec) != 2:
                        lgl = "`select: foilim` selection with two components"
                        act = "`select: foilim` with {} components".format(len(freqSpec))
                        raise SPYValueError(legal=lgl, varname=vname, actual=act)
                    if freqSpec[0] >= freqSpec[1]:
                        lgl = "`select: foilim` selection with `foilim[0]` < `foilim[1]`"
                        act = "selection range from {} to {}".format(freqSpec[0], freqSpec[1])
                        raise SPYValueError(legal=lgl, varname=vname, actual=act)
            self._freq = data._get_freq(foi=select.get("foi"), foilim=select.get("foilim"))
        else:
            return

    @property
    def taper(self):
        """List or slice encoding taper-selection"""
        return self._taper

    @taper.setter
    def taper(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "taper", "tapers")

    @property
    def unit(self):
        """List or slice encoding unit-selection"""
        return self._unit

    @unit.setter
    def unit(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "unit", "units")

    @property
    def eventid(self):
        """List or slice encoding event-id-selection"""
        return self._eventid

    @eventid.setter
    def eventid(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "eventid", "eventids")

    # Helper function to process provided selections        
    def _selection_setter(self, data, select, dataprop, selectkey):
        """
        Converts user-provided selection key-words to indexing lists/slices

        Parameters
        ----------
        data : Syncopy data object
            Non-empty Syncopy data object
        select : dict or :class:`StructDict`
            Python dictionary or Syncopy :class:`StructDict` formatted for
            data selection. See :class:`Selector` for a list of valid
            key-value pairs.
        dataprop : str
            Name of property in `data` to select from
        selectkey : str
            Name of key in `select` holding selection pertinent to `dataprop`

        Returns
        -------
        Nothing : None

        Notes
        -----
        This class method processes and (if necessary converts) user-provided
        selections. Valid selectors are slices, ranges, lists or arrays. If
        possible, all selections are converted to contiguous slices, otherwise
        regular Python lists are used. Selections can be unsorted and may
        include repetitions but must match exactly, be finite and not NaN. 
        Converted selections are stored in the respective (hidden) class
        attributes (e.g., ``self._channel``, ``self._unit`` etc.).

        See also
        --------
        syncopy.selectdata : extract data selections from Syncopy objects
        """
        
        # Unpack input and perform error-checking
        selection = select.get(selectkey)
        target = getattr(data, dataprop, None)
        selector = "_{}".format(dataprop)
        vname = "select: {}".format(selectkey)
        if selection is not None and target is None:
            lgl = "Syncopy data object with {}".format(selectkey)
            raise SPYValueError(legal=lgl, varname=vname, actual=data.__class__.__name__)
        
        if target is not None:

            if np.issubdtype(target.dtype, np.dtype("str").type):
                slcLims = [0, target.size]
                arrLims = None
                hasnan = None
                hasinf = None
            else:
                slcLims = [target[0], target[-1] + 1]
                arrLims = [target[0], target[-1]]
                hasnan = False
                hasinf = False
                
            # Take entire inventory sitting in `dataprop`
            if selection is None:
                if dataprop in ["unit", "eventid"]:
                    setattr(self, selector, [slice(None, None, 1)] * len(self.trials))
                else:
                    setattr(self, selector, slice(None, None, 1))
                
            # Check consistency of slice-selections and convert ranges to slices
            elif isinstance(selection, (slice, range)):
                selLims = [-np.inf, np.inf]
                if selection.start is not None:
                    selLims[0] = selection.start
                if selection.stop is not None:
                    selLims[1] = selection.stop
                if selLims[0] >= selLims[1]:
                    lgl = "selection range with min < max"
                    act = "selection range from {} to {}".format(selLims[0], selLims[1])
                    raise SPYValueError(legal=lgl, varname=vname, actual=act)
                # check slice/range boundaries: take care of things like `slice(-10, -3)`
                if np.isfinite(selLims[0]) and (selLims[0] < -slcLims[1] or selLims[0] >= slcLims[1]):
                    lgl = "selection range with min >= {}".format(slcLims[0])
                    act = "selection range starting at {}".format(selLims[0])
                    raise SPYValueError(legal=lgl, varname=vname, actual=act)
                if np.isfinite(selLims[1]) and (selLims[1] > slcLims[1] or selLims[1] < -slcLims[1]):
                    lgl = "selection range with max <= {}".format(slcLims[1])
                    act = "selection range ending at {}".format(selLims[1])
                    raise SPYValueError(legal=lgl, varname=vname, actual=act)

                # The 2d-arrays in `DiscreteData` objects require some additional hand-holding
                # performed by the respective `_get_unit` and `_get_eventid` class methods
                if dataprop in ["unit", "eventid"]:
                    if selection.start is selection.stop is None:
                        setattr(self, selector, [slice(None, None, 1)] * len(self.trials))
                    else:
                        if isinstance(selection, slice):
                            if np.issubdtype(target.dtype, np.dtype("str").type):
                                target = np.arange(target.size)
                            selection = list(target[selection])
                        else:
                            selection = list(selection)
                        setattr(self, selector, getattr(data, "_get_" + dataprop)(self.trials, selection))
                else:
                    if selection.start is selection.stop is None:
                        setattr(self, selector, slice(None, None, 1))
                    else:
                        if selection.step is None:
                            step = 1
                        else:
                            step = selection.step
                        setattr(self, selector, slice(selection.start, selection.stop, step))
                
            # Selection is either a valid list/array or bust
            else:
                try:
                    array_parser(selection, varname=vname, hasinf=hasinf, 
                                 hasnan=hasnan, lims=arrLims, dims=1)
                except Exception as exc:
                    raise exc
                selection = np.array(selection)
                if np.issubdtype(selection.dtype, np.dtype("str").type):
                    targetArr = target
                else:
                    targetArr = np.arange(target.size)
                if not set(selection).issubset(targetArr):
                    lgl = "List/array of {} names or indices".format(dataprop)
                    raise SPYValueError(legal=lgl, varname=vname)
                
                # Preserve order and duplicates of selection - don't use `np.isin` here!
                idxList = []
                for sel in selection:
                    idxList += list(np.where(targetArr == sel)[0])
                    
                if dataprop in ["unit", "eventid"]:
                    setattr(self, selector, getattr(data, "_get_" + dataprop)(self.trials, idxList))
                else:                
                    # if possible, convert range-arrays (`[0, 1, 2, 3]`) to slices for better performance
                    if len(idxList) > 1:
                        steps = np.diff(idxList)
                        if steps.min() == steps.max() == 1:
                            idxList = slice(idxList[0], idxList[-1] + 1, 1)
                    setattr(self, selector, idxList)
                    
        else:
            return

    # Local helper that converts slice selectors to lists (if necessary)        
    def _make_consistent(self, data):
        """
        Consolidates array selection tuples
        
        Parameters
        ----------
        data : Syncopy data object
            Non-empty Syncopy data object

        Returns
        -------
        Nothing : None

        Notes
        -----
        This class method is called after all user-provided selections have
        been (successfully) processed and (if necessary) converted to
        lists/slices. The integrity of conjoint multi-dimensional selections
        is ensured by guaranteeing that cross-dimensional selections are
        finite (i.e., lists) and no more than two lists are used simultaneously
        for a selection. If the current Selector instance contains multiple
        index lists, the contents of all selection properties is converted
        (if required) to lists so that multi-dimensional array-indexing can
        be readily performed via :func:`numpy.ix_`. 

        See also
        --------
        numpy.ix_ : Mesh-construction for array indexing
        """

        # Get list of all selectors that don't depend on trials
        dimProps = list(self._allProps)
        for prop in self._byTrialProps:
            dimProps.remove(prop)
        
        # Count how many lists we got
        listCount = 0
        for prop in dimProps:
            if isinstance(getattr(self, prop), list):
                listCount += 1

        # Now go through trial-dependent selectors to see if any by-trial selection is a list
        for prop in self._byTrialProps:
            selList = getattr(self, prop)
            if selList is not None:
                for tsel in selList:
                    if isinstance(tsel, list):
                        listCount += 1
                        break
                
        # If (on a by-trial basis) we have two or more lists, we need fancy indexing, 
        # thus convert all slice- to list-selectors
        if listCount >= 2:
            for tk, tsel in enumerate(self.time):
                if isinstance(tsel, slice):
                    start, stop, step = tsel.start, tsel.stop, tsel.step
                    if start is None:
                        start = 0
                    if stop is None:
                        trlTime = data._get_time([self.trials[tk]], toilim=[-np.inf, np.inf])[0]
                        if isinstance(trlTime, list):
                            stop = np.max(trlTime)
                        else:
                            stop = trlTime.stop
                    if step is None:
                        step = 1
                    self.time[tk] = list(range(start, stop, step))
            for prop in dimProps:
                sel = getattr(self, prop)
                if isinstance(sel, slice):
                    start, stop, step = sel.start, sel.stop, sel.step
                    if start is None:
                        start = 0
                    if stop is None:
                        stop = getattr(data, prop).size
                    if step is None:
                        step = 1
                    setattr(self, "_{}".format(prop), list(range(start, stop, step)))
            self._useFancy = True
        
        return
        
    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make selection readable from the command line
    def __str__(self):
        
        # Get list of print-worthy attributes
        ppattrs = [attr for attr in self.__dir__() if not attr.startswith("_")]
        ppattrs.sort()
        
        # Construct dict of pretty-printable property info
        ppdict = {}
        for attr in ppattrs:
            val = getattr(self, attr)
            if val is not None and attr in self._byTrialProps:
                val = val[0]
            if isinstance(val, slice):
                if val.start is val.stop is None:
                    ppdict[attr] = "all {}{}, ".format(attr, 
                                                       "s" if not attr.endswith("s") else "")
                elif val.start is None or val.stop is None:
                    ppdict[attr] = "{}-range, ".format(attr)
                else:
                    ppdict[attr] = "{0:d} {1:s}{2:s}, ".format(int(np.ceil((val.stop - val.start) / val.step)),
                                                               attr,
                                                               "s" if not attr.endswith("s") else "")
            elif isinstance(val, list):
                ppdict[attr] = "{0:d} {1:s}{2:s}, ".format(len(val), 
                                                           attr, 
                                                           "s" if not attr.endswith("s") else "")
            else:
                ppdict[attr] = ""
    
        # Construct string for printing
        msg = "Syncopy {} selector with ".format(self._dataClass)
        for pout in ppdict.values():
            msg += pout
                
        return msg[:-2]
