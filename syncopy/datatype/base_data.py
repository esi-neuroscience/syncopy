# -*- coding: utf-8 -*-
#
# Syncopy's main abstract base class + helpers
#

# Builtin/3rd party package imports
import getpass
import socket
import time
import sys
import os
from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import blake2b
from itertools import islice
from functools import reduce
from inspect import signature
import shutil
import numpy as np
import h5py
import scipy as sp

# Local imports
import syncopy as spy
from .methods.arithmetic import _process_operator
from .methods.selectdata import selectdata
from .methods.show import show
from syncopy.shared.tools import SerializableDict
from syncopy.shared.parsers import (
    scalar_parser,
    array_parser,
    io_parser,
    filename_parser,
    data_parser,
)
from syncopy.shared.errors import SPYInfo, SPYTypeError, SPYValueError, SPYError, SPYWarning
from syncopy.datatype.methods.definetrial import definetrial as _definetrial
from syncopy import __version__, __storage__, __acme__, __sessionid__, __storagelimit__

if __acme__:
    import acme
    import dask


__all__ = []


class BaseData(ABC):
    """
    Abstract base class for all data classes

    Data classes in Syncopy manage storing array data and metadata in HDF5 and
    JSON files, respectively. This base class contains the fundamental
    functionality shared across all data classes, that is,

    * properties for arrays that have a corresponding HDF5 datasets ('dataset
      properties') and the associated I/O
    * properties for data history (`BaseData.log` and `BaseData.cfg`)
    * methods and properties for defining trials on the data

    Further properties and methods are defined in subclasses, e.g.
    `syncopy.AnalogData`.
    """

    #: properties that are written into the JSON file and HDF5 attributes upon save
    _infoFileProperties = ("dimord", "_version", "_log", "cfg", "info")
    _hdfFileAttributeProperties = (
        "dimord",
        "_version",
        "_log",
    )
    # all data types have a `trials` property
    _selectionKeyWords = ('trials',)
    #: properties that are mapped onto HDF5 datasets
    _hdfFileDatasetProperties = ()

    # Checksum algorithm
    _checksum_algorithm = spy.__checksum_algorithm__.__name__

    # Dummy allocations of class attributes that are actually initialized in subclasses
    _mode = None
    _stackingDimLabel = None

    # Set caller for `SPYWarning` to not have it show up as '<module>'
    _spwCaller = "BaseData.{}"

    # Attach data selection and output routines to make them available as class methods
    selectdata = selectdata
    show = show

    # Initialize hidden attributes used by all children
    _filename = None
    _trialdefinition = None
    _dimord = None
    _mode = None
    _lhd = (
        "\n\t\t>>> SyNCopy v. {ver:s} <<< \n\n"
        + "Created: {timestamp:s} \n\n"
        + "System Profile: \n"
        + "{sysver:s} \n"
        + "ACME:  {acver:s}\n"
        + "Dask:  {daver:s}\n"
        + "NumPy: {npver:s}\n"
        + "SciPy: {spver:s}\n\n"
        + "--- LOG ---"
    )
    _log_header = _lhd.format(
        ver=__version__,
        timestamp=time.asctime(),
        sysver=sys.version,
        acver=acme.__version__ if __acme__ else "--",
        daver=dask.__version__ if __acme__ else "--",
        npver=np.__version__,
        spver=sp.__version__,
    )
    _log = ""

    @property
    @classmethod
    @abstractmethod
    def _defaultDimord(cls):
        return NotImplementedError

    @property
    def _stackingDim(self):
        if any(["DiscreteData" in str(base) for base in self.__class__.__mro__]):
            return 0
        else:
            if self._stackingDimLabel is not None and self.dimord is not None:
                return self.dimord.index(self._stackingDimLabel)

    @property
    def cfg(self):
        """Dictionary of previous operations on data"""
        return self._cfg

    @cfg.setter
    def cfg(self, dct):
        """ For loading only, for processing the frontends
        extend the existing (empty) cfg dictionary """

        if not isinstance(dct, dict):
            raise SPYTypeError(dct, varname="cfg", expected="dictionary-like object")
        self._cfg = dct

    @property
    def info(self):
        """Dictionary of auxiliary meta information"""
        return self._info

    @info.setter
    def info(self, dct):

        """
        Users usually want to extend the existing info dictionary,
        however it is possible to completely overwrite with a new dict
        """

        if not isinstance(dct, dict):
            raise SPYTypeError(dct, varname="info", expected="dictionary-like object")

        self._info = SerializableDict(dct)

    @property
    def container(self):
        try:
            return filename_parser(self.filename)["container"]
        except SPYValueError:
            return None
        except Exception as exc:
            raise exc

    def _register_dataset(self, propertyName, inData=None):
        """
        Register a new dataset, so that it is handled during saving, comparison, copy and other operations.
        This dataset is not managed in any way during parallel operations and is intended for
        holding additional data things like statistics. Thus it is NOT safe to use this in a
        multi-threaded/parallel context, like in a compute function (cF).

        Parameters
        ----------
        propertyName : str
            The name for the new dataset, this will be used as the dataset name in the hdf5 container
            when saving. It will be added as an attribute named `'_' + propertyName` to this SyncopyData object.
            Note that this means that your propertyName must not clash with other attribute names of
            syncopy data objects. To ensure the latter, it is recommended to use names with a prefix like
            `'dset_'`. Clashes will be detected and result in errors.
        in_data : None or np.ndarray or h5py.Dataset
            The data to store. Must have the final number of dimensions you want.
        """
        if not propertyName in self._hdfFileDatasetProperties:
            self._hdfFileDatasetProperties = self._hdfFileDatasetProperties + (propertyName,)

        # trivial case
        if inData is None:
            setattr(self, "_" + propertyName, None)
            return

        supportedSetters = {
            np.ndarray: self._set_dataset_property_with_ndarray,
            h5py.Dataset: self._set_dataset_property_with_dataset,
        }

        try:
            # same attribute for both ndarray and hdf5 dataset
            ndim = inData.ndim
        except AttributeError:
            msg = "HDF5 dataset, or NumPy array"
            raise SPYTypeError(inData, varname="data", expected=msg)

        supportedSetters[type(inData)](inData, propertyName, ndim=ndim)

    def _unregister_dataset(self, propertyName, del_from_file=True):
        """
        Unregister and delete an additional dataset from the Syncopy data object,
        and optionally delete it from the backing hdf5 file.

        Assumes that the backing h5py file is open in writeable mode.

        Parameters
        ----------
            propertyName : str
                The name of the entry in `self._hdfFileDatasetProperties` to remove.
                The attribute named `'_' + propertyName` of this SyncopyData object will be deleted.
            del_from_file: bool
                Whether to also remove the dataset named 'propertyName' from the backing hdf5 file on disk.
        """
        if propertyName in self._hdfFileDatasetProperties:
            tmp_list = list(self._hdfFileDatasetProperties)
            tmp_list.remove(propertyName)
            self._hdfFileDatasetProperties = tuple(tmp_list)
        if hasattr(self, "_" + propertyName):
            delattr(self, "_" + propertyName)
        if del_from_file:
            if self.mode == "r":
                lgl = "HDF5 dataset with write or copy-on-write access"
                act = "read-only file"
                raise SPYValueError(legal=lgl, varname="mode", actual=act)
            if isinstance(self._data, h5py.Dataset):
                if isinstance(self._data.file, h5py.File):
                    if propertyName in self._data.file.keys():
                        del self._data.file[propertyName]
                else:
                    SPYWarning("Could not delete dataset from file.")

    def _update_dataset(self, propertyName, inData=None):
        """
        Resets an additional dataset which was already registered via
        ``_register_dataset`` to ``inData``.
        """
        if getattr(self, "_" + propertyName) is not None:
            self._unregister_dataset(propertyName)
        self._register_dataset(propertyName, inData)

    def _set_dataset_property(self, inData, propertyName, ndim=None):
        """Set property that is streamed from HDF dataset ('dataset property')

        This method automatically selects the appropriate set method
        according to the type of the input data (`dataIn`).

        Parameters
        ----------
            dataIn : str, np.ndarray, or h5py.Dataset
                Filename, array or HDF5 dataset to be stored in property
            propertyName : str
                Name of the property. The actual data must reside in the attribute
                `"_" + propertyName`
            ndim : int
                Number of expected array dimensions.

        """
        if propertyName == "data":
            if any(["DiscreteData" in str(base) for base in self.__class__.__mro__]):
                ndim = 2
            if ndim is None:
                ndim = len(self._defaultDimord)

        supportedSetters = {
            list: self._set_dataset_property_with_list,
            str: self._set_dataset_property_with_str,
            np.ndarray: self._set_dataset_property_with_ndarray,
            h5py.Dataset: self._set_dataset_property_with_dataset,
            type(None): self._set_dataset_property_with_none,
        }
        try:
            supportedSetters[type(inData)](inData, propertyName, ndim=ndim)
        except KeyError:
            msg = "filename of HDF5 file, HDF5 dataset, or NumPy array"
            raise SPYTypeError(inData, varname="data", expected=msg)

    def _set_dataset_property_with_none(self, inData, propertyName, ndim):
        """Set a dataset property to None"""
        setattr(self, "_" + propertyName, None)

    def _set_dataset_property_with_str(self, filename, propertyName, ndim):
        """Set a dataset property with a filename str

        Parameters
        ----------
            filename : str
                A filename pointing to a HDF5 file containing the dataset
                `propertyName`.
            propertyName : str
                Name of the property to be filled with the dataset
            ndim : int
                Number of expected array dimensions.
        """

        fpath, fname = io_parser(
            filename, varname="filename", isfile=True, exists=True
        )
        filename = os.path.join(fpath, fname)  # ensure `filename` is absolute path

        md = self.mode
        if md == "w":
            md = "r+"

        isHdf = False
        try:
            h5f = h5py.File(filename, mode=md)
            isHdf = True
        except OSError as exc:
            err = "HDF5: " + str(exc)
        if not isHdf:
            raise SPYValueError("accessible HDF5 file", actual=err, varname="data")

        h5keys = list(h5f.keys())
        if propertyName not in h5keys and len(h5keys) != 1:
            lgl = "HDF5 file with only one 'data' dataset or single dataset of arbitrary name"
            act = "HDF5 file holding {} data-objects"
            raise SPYValueError(
                legal=lgl, actual=act.format(str(len(h5keys))), varname=propertyName
            )
        if len(h5keys) == 1:
            setattr(self, propertyName, h5f[h5keys[0]])
        else:
            setattr(self, propertyName, h5f[propertyName])

        self.filename = filename

    def _set_dataset_property_with_ndarray(self, inData, propertyName, ndim):
        """Set a dataset property with a NumPy array

        If no data exists, a backing HDF5 dataset will be created.

        Parameters
        ----------
        inData : numpy.ndarray
            NumPy array to be stored in property of name `propertyName`
        propertyName : str
            Name of the property to be filled with `inData`. Will get an underscore (`'_'`) prefix added,
            so do not include that.
        ndim : int
            Number of expected array dimensions.
        """
        # Ensure array has right no. of dimensions
        array_parser(inData, varname=f"{propertyName}", dims=ndim)

        # Gymnastics for `DiscreteData` objects w/non-standard `dimord`s.
        # This only applies to the 'main' dataset called 'data'. The checks are not needed
        # for additional, sequential datasets which people may attach.
        if propertyName == "data":
            self._check_dataset_property_discretedata(inData)
        else:
            if not hasattr(self, "_" + propertyName):
                setattr(self, "_" + propertyName, None)  # Prevent error on gettattr call below.

        # If there is existing data, replace values if shape and type match
        if isinstance(getattr(self, "_" + propertyName), h5py.Dataset):
            prop = getattr(self, "_" + propertyName)
            if self.mode == "r":
                lgl = "dataset with write or copy-on-write access"
                act = "read-only file"
                raise SPYValueError(legal=lgl, varname="mode", actual=act)
            if prop.shape != inData.shape:
                lgl = "dataset with shape {}".format(str(prop.shape))
                act = "data with shape {}".format(str(inData.shape))
                raise SPYValueError(legal=lgl, varname="data", actual=act)
            if prop.dtype != inData.dtype:
                lgl = "dataset of type {}".format(prop.dtype.name)
                act = "data of type {}".format(inData.dtype.name)
                raise SPYValueError(legal=lgl, varname="data", actual=act)
            prop[...] = inData

        # or create backing file on disk
        else:
            if self.filename is None:
                self.filename = self._gen_filename()

            if propertyName not in self._hdfFileDatasetProperties:
                if getattr(self, "_" + propertyName) is not None and not isinstance(getattr(self, "_" + propertyName), h5py.Dataset):
                    raise SPYValueError(legal="propertyName that does not clash with existing attributes",
                                        varname=propertyName, actual=propertyName)

            h5f = self._get_backing_hdf5_file_handle()
            if h5f is None:
                with h5py.File(self.filename, "w") as h5f:
                    h5f.create_dataset(propertyName, data=inData)
            else:
                h5f.create_dataset(propertyName, data=inData)


        md = self.mode
        if md == "w":
            md = "r+"
        setattr(self, "_" + propertyName, h5py.File(self.filename, md)[propertyName])


    def _set_dataset_property_with_dataset(self, inData, propertyName, ndim):
        """Set a dataset property with an already loaded HDF5 dataset

        Parameters
        ----------
        inData : h5py.Dataset
            HDF5 dataset to be stored in property of name `propertyName`
        propertyName : str
            Name of the property to be filled with the dataset
        ndim : int
            Number of expected array dimensions.
        """

        if inData.id.valid == 0:
            lgl = "open HDF5 file"
            act = "backing HDF5 file is closed"
            raise SPYValueError(legal=lgl, actual=act, varname="data")

        # Ensure dataset has right no. of dimensions
        if inData.ndim != ndim:
            lgl = "{}-dimensional data".format(ndim)
            act = "{}-dimensional HDF5 dataset".format(inData.ndim)
            raise SPYValueError(legal=lgl, varname="data", actual=act)

        if propertyName == "data":
            self._check_dataset_property_discretedata(inData)
            self.filename = inData.file.filename
        else:
            # creates hidden attribute behind the property on the fly
            if not hasattr(self, "_" + propertyName):
                setattr(self, "_" + propertyName, None)

        self._mode = inData.file.mode
        setattr(self, "_" + propertyName, inData)

    def _set_dataset_property_with_list(self, inData, propertyName, ndim):
        """Set a dataset property with list of NumPy arrays

        Parameters
        ----------
            inData : list
                list of :class:`numpy.ndarray`s. Each array corresponds to
                a trial. Arrays are stacked together to fill dataset.
            propertyName : str
                Name of the property to be filled with the concatenated array
            ndim : int
                Number of expected array dimensions.
        """

        # Check list entries: must be numeric, finite NumPy arrays
        for val in inData:
            try:
                array_parser(val, varname="data", hasinf=False, dims=ndim)
            except Exception as exc:
                raise exc

        # Ensure we don't have a mix of real/complex arrays
        if np.unique([np.iscomplexobj(val) for val in inData]).size > 1:
            lgl = "list of numeric NumPy arrays of same numeric type (real/complex)"
            act = "real and complex NumPy arrays"
            raise SPYValueError(legal=lgl, varname="data", actual=act)

        # Requirements for input arrays differ wrt data-class (`DiscreteData` always 2D)
        if any(["ContinuousData" in str(base) for base in self.__class__.__mro__]):

            # Ensure shapes match up
            if any(val.shape != inData[0].shape for val in inData):
                lgl = "NumPy arrays of identical shape"
                act = "NumPy arrays with differing shapes"
                raise SPYValueError(legal=lgl, varname="data", actual=act)
            trialLens = [val.shape[self.dimord.index("time")] for val in inData]

        else:

            # Ensure all arrays have shape `(N, nCol)``
            if self.__class__.__name__ == "SpikeData":
                nCol = 3
            else:  # EventData
                nCol = inData[0].shape[1]
            if any(val.shape[1] != nCol for val in inData):
                lgl = "NumPy 2d-arrays with {} columns".format(nCol)
                act = "NumPy arrays of different shape"
                raise SPYValueError(legal=lgl, varname="data", actual=act)
            trialLens = [
                np.nanmax(val[:, self.dimord.index("sample")]) for val in inData
            ]

        nTrials = len(trialLens)

        # Use constructed quantities to set up trial layout matrix
        accumSamples = np.cumsum(trialLens)
        trialdefinition = np.zeros((nTrials, 3))
        trialdefinition[1:, 0] = accumSamples[:-1]
        trialdefinition[:, 1] = accumSamples
        if self.samplerate is not None:
            # set standard offset to -1s
            trialdefinition[:, 2] = -self.samplerate
        else:
            trialdefinition[:, 2] = 0

        # Finally, concatenate provided arrays and let corresponding setting method
        # perform the actual HDF magic
        data = np.concatenate(inData, axis=self._stackingDim)
        self._set_dataset_property_with_ndarray(data, propertyName, ndim)
        self.trialdefinition = trialdefinition

    def _check_dataset_property_discretedata(self, inData):
        """Check `DiscreteData` input data for shape consistency

        Parameters
        ----------
            inData : array/h5py.Dataset
                array-like to be stored as a `DiscreteData` data source
        """

        # Special case `DiscreteData`: `dimord` encodes no. of expected cols/rows;
        # ensure this is consistent w/`inData`!
        if any(["DiscreteData" in str(base) for base in self.__class__.__mro__]):
            if len(self._defaultDimord) not in inData.shape:
                lgl = "array with {} columns corresponding to dimord {}"
                lgl = lgl.format(len(self._defaultDimord), self._defaultDimord)
                act = "array with shape {}".format(str(inData.shape))
                raise SPYValueError(legal=lgl, varname="data", actual=act)

    def _is_empty(self):
        return all(
            [getattr(self, "_" + attr, None) is None for attr in self._hdfFileDatasetProperties]
        )

    @property
    def dimord(self):
        """list(str): ordered list of data dimension labels"""
        return self._dimord

    @dimord.setter
    def dimord(self, dims):

        # ensure `dims` can be safely compared to potentially existing `self._dimord`
        if dims is not None:
            try:
                array_parser(dims, varname="dims", ntype="str", dims=1)
            except Exception as exc:
                raise exc

        if self._dimord is not None and not dims == self._dimord:
            print(
                "Syncopy core - dimord: Cannot change `dimord` of object. "
                + "Functionality currently not supported"
            )

        if dims is None:
            self._dimord = None
            return

        # this enforces the _defaultDimord
        if set(dims) != set(self._defaultDimord):
            base = "dimensional labels {}"
            lgl = base.format(
                "'" + "' x '".join(str(dim) for dim in self._defaultDimord) + "'"
            )
            act = base.format("'" + "' x '".join(str(dim) for dim in dims) + "'")
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)

        # this enforces that custom dimords are set for every axis
        if len(dims) != len(self._defaultDimord):
            lgl = f"Custom dimord has length {len(self._defaultDimord)}"
            act = f"Custom dimord has length {len(dims)}"
            raise SPYValueError(legal=lgl, varname="dimord", actual=act)

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
        self._filename = os.path.abspath(os.path.expanduser(str(fname)))

    @property
    def log(self):
        """str: log of previous operations on data"""
        print(self._log_header + self._log)

    @log.setter
    def log(self, msg):
        """ This appends the assigned msg to the existing log """
        if not isinstance(msg, str):
            raise SPYTypeError(msg, varname="log", expected="str")
        prefix = "\n\n|=== {user:s}@{host:s}: {time:s} ===|\n\n\t{caller:s}"
        clr = sys._getframe().f_back.f_code.co_name
        if clr.startswith("_") and not clr.startswith("__"):
            clr = clr[1:]
        self._log += (
            prefix.format(
                user=getpass.getuser(),
                host=socket.gethostname(),
                time=time.asctime(),
                caller=clr + ": " if clr != "<module>" else "",
            )
            + msg
        )

    @property
    def mode(self):
        """str: write mode for data, 'r' for read-only, 'w' for writable

        FIXME: append/replace with HDF5?
        """
        return self._mode

    @property
    def tag(self):
        try:
            return filename_parser(self.filename)["tag"]
        except SPYValueError:
            return None
        except Exception as exc:
            raise exc

    @mode.setter
    def mode(self, md):

        # If the mode is not changing, don't do anything
        if md == self._mode:
            return

        # Ensure input makes sense and we actually have permission to change
        # the data access mode
        if not isinstance(md, str):
            raise SPYTypeError(md, varname="mode", expected="str")
        options = ["r", "r+", "w", "c"]
        if md not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(lgl, varname="mode", actual=md)

        # prevent accidental data loss by not allowing mode = "w" in h5py
        if md == "w":
            md = "r+"

        # If data is already attached to the object, flush and close. All
        # datasets need to be closed before the file can be re-opened with a
        # different mode.

        # This assumes that all datasets attached as properties are stored in
        #  the same hdf5 file, and thus closing the file for 'data' handles all others.

        for prop in self._hdfFileDatasetProperties:
            if isinstance(prop, h5py.Dataset):
                prop.flush()

        prop = getattr(self, self._hdfFileDatasetProperties[0])
        if prop is not None:
            prop.file.close()

        # Re-attach datasets
        for propertyName in self._hdfFileDatasetProperties:
            if prop is not None:
                try:
                    prop_value = h5py.File(self.filename, mode=md)[propertyName]
                except:
                    SPYInfo(f"Could not retrieve dataset '{propertyName}' from HDF5 file.")
                    prop_value = None
                prop_name = propertyName if propertyName == "data" else "_" + propertyName
                setattr(self, prop_name, prop_value)
        self._mode = md

    @property
    def selection(self):
        """Data selection specified by :class:`Selector`"""
        return self._selector

    @selection.setter
    def selection(self, select):
        if select is None:
            self._selector = None
        else:
            self._selector = Selector(self, select)

    @property
    def trialdefinition(self):
        """nTrials x >=3 :class:`numpy.ndarray` of [start, end, offset, trialinfo[:]]"""
        return np.array(self._trialdefinition)

    @trialdefinition.setter
    def trialdefinition(self, trl):
        _definetrial(self, trialdefinition=trl)

    @property
    def sampleinfo(self):
        """nTrials x 2 :class:`numpy.ndarray` of [start, end] sample indices"""
        if self._trialdefinition is not None:
            return self._trialdefinition[:, :2]
        else:
            return None

    @sampleinfo.setter
    def sampleinfo(self, sinfo):
        raise SPYError(
            "Cannot set sampleinfo. Use `BaseData._trialdefinition` instead."
        )

    @property
    def trialintervals(self):
        """nTrials x 2 :class:`numpy.ndarray` of [start, end] times in seconds """
        if self._trialdefinition is not None and self._samplerate is not None:
            # trial lengths in samples
            start_end = self.sampleinfo - self.sampleinfo[:, 0][:, None]
            start_end[:, 1] -= 1  # account for last time point
           # add offset and convert to seconds
            start_end = (start_end + self._t0[:, None]) / self._samplerate
            return start_end
        else:
            return None

    @property
    def _t0(self):
        """ These are the (trigger) offsets """
        if self._trialdefinition is not None:
            return self._trialdefinition[:, 2]
        else:
            return None

    @property
    def trials(self):
        """list-like array of trials"""

        return Indexer(map(self._get_trial, range(self.sampleinfo.shape[0])),
                       self.sampleinfo.shape[0]) if self.sampleinfo is not None else None

    @property
    def trialinfo(self):
        """nTrials x M :class:`numpy.ndarray` with numeric information about each trial

        Each trial can have M properties (condition, original trial no., ...) coded by
        numbers. This property are the fourth and onward columns of `BaseData._trialdefinition`.
        """
        if self._trialdefinition is not None:
            if self._trialdefinition.shape[1] > 3:
                return self._trialdefinition[:, 3:]
            else:
                # If trials are defined but no trialinfo return empty array with
                # nTrial rows, but 0 columns. This works well with np.hstack.
                return np.empty(shape=(len(self.trials), 0))
        else:
            return None

    @trialinfo.setter
    def trialinfo(self, trl):
        raise SPYError(
            "Cannot set trialinfo. Use `BaseData._trialdefinition` or `syncopy.definetrial` instead."
        )

    # Helper function that grabs a single trial
    @abstractmethod
    def _get_trial(self, trialno):
        pass

    # Helper function that creates a `FauxTrial` object given actual trial information
    @abstractmethod
    def _preview_trial(self, trialno):
        pass

    # Convenience function, wiping contents of backing device from memory
    def clear(self):
        """Clear loaded data from memory

        Calls `flush` method of HDF5 dataset.
        """
        for propName in self._hdfFileDatasetProperties:
            dsetProp = getattr(self, "_" + propName)
            if dsetProp is not None:
                dsetProp.flush()
        return

    def _close(self):
        """Close backing hdf5 file."""
        self.clear()
        for propertyName in self._hdfFileDatasetProperties:
            dsetProp = getattr(self, "_" + propertyName)
            if isinstance(dsetProp, h5py.Dataset):
                if dsetProp.id.valid != 0:  # Check whether backing HDF5 file is open.
                    dsetProp.file.close()

    def _get_backing_hdf5_file_handle(self):
        """Get handle to `h5py.File` instance of backing HDF5 file

        Checks all datasets in `self._hdfFileDatasetProperties` for valid handles, returns `None` if none found.

           Note that the mode of the returned instance depends on the current value of `self.mode`.
        """
        for propertyName in self._hdfFileDatasetProperties:
            dsetProp = getattr(self, "_" + propertyName)
            if isinstance(dsetProp, h5py.Dataset):
                if dsetProp.id.valid != 0:
                    return dsetProp.file
        return None

    def _reopen(self):
        """ Reattach datasets from backing hdf5 file. Respects current `self.mode`."""
        for propertyName in self._hdfFileDatasetProperties:
            dsetProp = getattr(self, "_" + propertyName)
            if isinstance(dsetProp, h5py.Dataset):
                setattr(self, "_" + propertyName, h5py.File(self.filename, mode=self.mode)[propertyName])

    def copy(self):
        """
        Create a copy of the entire object on disk.

        Returns
        -------
        cpy : Syncopy data object
            Reference to the copied data object
            on disk

        Notes
        -----
        For copying only a subset of the `data` use :func:`syncopy.selectdata` directly
        with the default `inplace=False` parameter.

        See also
        --------
        :func:`syncopy.save` : save to specific file path
        :func:`syncopy.selectdata` : creates copy of a selection with `inplace=False`

        """

        return spy.copy(self)

    # Attach trial-definition routine to not re-invent the wheel here
    definetrial = _definetrial

    # Wrapper that makes saving routine usable as class method
    def save(self, container=None, tag=None, filename=None, overwrite=False):
        r"""Save data object as new ``spy`` container to disk (:func:`syncopy.save_data`)

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
                raise SPYError(
                    "Cannot create spy container in temporary "
                    + "storage {} - please provide explicit path. ".format(__storage__)
                )
            overwrite = True
            filename = self.filename

        # Support `obj.save(tag="newtag")`
        if container is None and filename is None:
            if self.container is None:
                raise SPYError(
                    "Object is not associated to an existing spy container - "
                    + "please save object first using an explicit path. "
                )
            container = filename_parser(self.filename)["folder"]

        spy.save(
            self, filename=filename, container=container, tag=tag, overwrite=overwrite
        )

    # Helper function generating pseudo-random temp file-names
    def _gen_filename(self):

        fname_hsh = blake2b(
            digest_size=4, salt=os.urandom(blake2b.SALT_SIZE)
        ).hexdigest()
        fname = os.path.join(
            __storage__,
            "spy_{sess:s}_{hash:s}{ext:s}".format(
                sess=__sessionid__, hash=fname_hsh, ext=self._classname_to_extension()
            ),
        )
        return fname

    # Helper function converting object class-name to usable file extension
    def _classname_to_extension(self):
        return "." + self.__class__.__name__.split("Data")[0].lower()

    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make class contents readable from the command line
    @abstractmethod
    def __str__(self):
        pass

    # Destructor
    def __del__(self):
        if self.filename is not None:
            for propertyName in self._hdfFileDatasetProperties:
                prop = getattr(self, "_" + propertyName)
                try:
                    if isinstance(prop, h5py.Dataset):
                        try:
                            prop.file.close()
                        except (IOError, ValueError, TypeError, ImportError):
                            pass
                        except Exception as exc:
                            raise exc
                    else:
                        del prop
                except TypeError:
                    del prop
            if __storage__ in self.filename and os.path.exists(self.filename):
                os.unlink(self.filename)
                shutil.rmtree(os.path.splitext(self.filename)[0], ignore_errors=True)

    # Support for basic arithmetic operations (no in-place computations supported yet)
    def __add__(self, other):
        return _process_operator(self, other, "+")

    def __radd__(self, other):
        return _process_operator(self, other, "+")

    def __sub__(self, other):
        return _process_operator(self, other, "-")

    def __rsub__(self, other):
        return _process_operator(self, other, "-")

    def __mul__(self, other):
        return _process_operator(self, other, "*")

    def __rmul__(self, other):
        return _process_operator(self, other, "*")

    def __truediv__(self, other):
        return _process_operator(self, other, "/")

    def __rtruediv__(self, other):
        return _process_operator(self, other, "/")

    def __pow__(self, other):
        return _process_operator(self, other, "**")

    def __eq__(self, other):

        # If other object is not a Syncopy data-class, get out
        if not "BaseData" in str(other.__class__.__mro__):
            SPYInfo("Not a Syncopy object")
            return False

        # Check if two Syncopy objects of same type/dimord are present
        try:
            data_parser(other, dimord=self.dimord, dataclass=self.__class__.__name__)
        except Exception as exc:
            SPYInfo("Syncopy object of different type/dimord")
            return False

        # First, ensure we have something to compare here
        if self._is_empty():
            if not other._is_empty():
                SPYInfo("Empty and non-empty Syncopy object")
                return False
            return True
        elif not self._is_empty():
            if other._is_empty():
                SPYInfo("Non-empty and empty Syncopy object")
                return False

        # If in-place selections are present, abort
        if self.selection is not None or other.selection is not None:
            err = "Cannot perform object comparison with existing in-place selection"
            raise SPYError(err)

        # Use `_infoFileProperties` to fetch dimensional object props: remove `dimord`
        # (has already been checked by `data_parser` above) and remove `cfg` (two
        # objects might be identical even if their history deviates)
        dimProps = [
            prop for prop in self._infoFileProperties if not prop.startswith("_")
        ]
        dimProps = list(set(dimProps).difference(["dimord", "cfg"]))
        for prop in dimProps:
            val_this = getattr(self, prop)
            val_other = getattr(other, prop)
            if isinstance(val_this, np.ndarray) and isinstance(val_other, np.ndarray):
                isEqual = val_this.tolist() == val_other.tolist()
            # catch None
            elif val_this is None and val_other is not None:
                isEqual = False
            elif val_this is not None and val_other is None:
                isEqual = False
            else:
                isEqual = val_this == val_other
            if not isEqual:
                SPYInfo("Mismatch in {}".format(prop))
                return False

        # Check if trial setup is identical
        if not np.array_equal(self.trialdefinition, other.trialdefinition):
            SPYInfo("Mismatch in trial layouts")
            return False

        # If an object is compared to itself (or its shallow copy), don't bother
        # juggling NumPy arrays but simply perform a quick dataset/filename comparison
        both_hdfFileDatasetProperties = self._hdfFileDatasetProperties + other._hdfFileDatasetProperties

        isEqual = True
        if self.filename == other.filename:
            for dsetName in both_hdfFileDatasetProperties:
                if hasattr(self, "_" + dsetName) and hasattr(other, "_" + dsetName):
                    val_this = getattr(self, "_" + dsetName)
                    val_other = getattr(other, "_" + dsetName)
                    if isinstance(val_this, h5py.Dataset):
                        isEqual = val_this == val_other

                    if not isEqual:
                        SPYInfo(f"HDF dataset '{dsetName}' mismatch for types '{type(val_this)}' and '{type(val_other)}'")
                        return False
                else:
                    SPYInfo(f"HDF dataset mismatch: extra dataset '{dsetName}' in one instance")
                    return False
        else:
            for dsetName in both_hdfFileDatasetProperties:
                if dsetName != "data":
                    if hasattr(self, "_" + dsetName) and hasattr(other, "_" + dsetName):
                        val_this = getattr(self, "_" + dsetName)
                        val_other = getattr(other, "_" + dsetName)
                        if isinstance(val_this, h5py.Dataset):
                            #isEqual = True  # This case gets checked by trial below.
                            isEqual = val_this == val_other
                        elif val_this is None and val_other is None:
                            isEqual = True

                        if not isEqual:
                            SPYInfo(f"HDF dataset '{dsetName}' mismatch for types '{type(val_this)}' and '{type(val_other)}'")
                            return False
                    else:
                        SPYInfo(f"HDF dataset mismatch: extra dataset '{dsetName}' in one instance")
                        return False

            # The other object really is a standalone Syncopy class instance and
            # everything but the data itself aligns; now the most expensive part:
            # trial by trial data comparison
            for tk in range(len(self.trials)):
                if not np.allclose(self.trials[tk], other.trials[tk]):
                    SPYInfo("Mismatch in trial #{}".format(tk))
                    return False

        # If we made it this far, `self` and `other` really seem to be identical
        return True

    # Class "constructor"
    def __init__(self, filename=None, dimord=None, mode="r+", **kwargs):
        """
        Docstring

        1. filename + data = create HDF5 file at filename with data in it
        2. data only

        Keys of kwargs are the datasets from _hdfFileDatasetProperties, and
        kwargs must *only* include datasets for which a setter exists.
        """

        # each instance needs its own cfg!
        self._cfg = {}
        self._info = SerializableDict()

        # Initialize hidden attributes
        for propertyName in self._hdfFileDatasetProperties:
            setattr(self, "_" + propertyName, None)

        self._selector = None

        # Make instantiation persistent in all subclasses
        super().__init__()

        # Set mode
        self.mode = mode

        # If any dataset property contains data and no dimord is set, use the
        # default dimord
        if (
            any(
                [
                    key in self._hdfFileDatasetProperties and value is not None
                    for key, value in kwargs.items()
                ]
            )
            and dimord is None
        ):
            self.dimord = self._defaultDimord
        else:
            self.dimord = dimord

        # If a target filename is provided use it, otherwise generate random
        # filename in `syncopy.__storage__`
        if filename is not None:
            self.filename = filename
        else:
            self.filename = self._gen_filename()

        # Attach dataset properties and let set methods do error checking.
        for propertyName in self._hdfFileDatasetProperties:
            if propertyName in kwargs:
                setattr(self, propertyName, kwargs[propertyName])

        # Write initial log entry
        self.log = "created {clname:s} object".format(clname=self.__class__.__name__)

        # Write version
        self._version = __version__


class Indexer:

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
        if np.issubdtype(type(idx), np.number):
            try:
                scalar_parser(
                    idx, varname="idx", ntype="int_like", lims=[0, self._iterlen - 1]
                )
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
            if not (0 <= index.start < self._iterlen) or not (
                0 < index.stop <= self._iterlen
            ):
                err = "value between {lb:s} and {ub:s}"
                raise SPYValueError(
                    err.format(lb="0", ub=str(self._iterlen)),
                    varname="idx",
                    actual=str(index),
                )
            return np.hstack(islice(self._iterobj, index.start, index.stop, index.step))
        elif isinstance(idx, (list, np.ndarray)):
            try:
                array_parser(
                    idx,
                    varname="idx",
                    ntype="int_like",
                    hasnan=False,
                    hasinf=False,
                    lims=[0, self._iterlen],
                    dims=1,
                )
            except Exception as exc:
                raise exc
            return np.hstack(
                [next(islice(self._iterobj, int(ix), int(ix + 1))) for ix in idx]
            )
        else:
            raise SPYTypeError(idx, varname="idx", expected="int_like or slice")

    def __len__(self):
        return self._iterlen

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "{} element iterable".format(self._iterlen)


class SessionLogger:

    __slots__ = ["sessionfile", "_rm"]

    def __init__(self):

        # Create package-wide tmp directory if not already present
        if not os.path.exists(__storage__):
            try:
                os.mkdir(__storage__)
            except Exception as exc:
                err = (
                    "Syncopy core: cannot create temporary storage directory {}. "
                    + "Original error message below\n{}"
                )
                raise IOError(err.format(__storage__, str(exc)))

        # Check for upper bound of temp directory size
        with os.scandir(__storage__) as scan:
            st_size = 0.0
            st_fles = 0
            for fle in scan:
                try:
                    st_size += fle.stat().st_size / 1024 ** 3
                    st_fles += 1
                # this catches a cleanup by another process
                except FileNotFoundError:
                    continue

            if st_size > __storagelimit__:
                msg = (
                    "\nSyncopy <core> WARNING: Temporary storage folder {tmpdir:s} "
                    + "contains {nfs:d} files taking up a total of {sze:4.2f} GB on disk. \n"
                    + "Consider running `spy.cleanup()` to free up disk space."
                )
                print(msg.format(tmpdir=__storage__, nfs=st_fles, sze=st_size))

        # If we made it to this point, (attempt to) write the session file
        sess_log = "{user:s}@{host:s}: <{time:s}> started session {sess:s}"
        self.sessionfile = os.path.join(
            __storage__, "session_{}_log.id".format(__sessionid__)
        )
        try:
            with open(self.sessionfile, "w") as fid:
                fid.write(
                    sess_log.format(
                        user=getpass.getuser(),
                        host=socket.gethostname(),
                        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        sess=__sessionid__,
                    )
                )
        except Exception as exc:
            err = "Syncopy core: cannot access {}. Original error message below\n{}"
            raise IOError(err.format(self.sessionfile, str(exc)))

        # Workaround to prevent Python from garbage-collecting ``os.unlink``
        self._rm = os.unlink

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Session {}".format(__sessionid__)

    def __del__(self):
        try:
            self._rm(self.sessionfile)
        except FileNotFoundError:
            pass


class FauxTrial:
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
    dimord : list
        Dimensional order of source trial array

    Returns
    -------
    faux_trl : FauxTrial object
        An instance of `FauxTrial` that essentially parrots :class:`numpy.ndarray`
        objects and can, thus, be used to feed "fake" trials into a
        :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
        to get the `noCompute` runs out of the way w/o actually loading trials
        into memory.

    See also
    --------
    syncopy.continuous_data.ContinuousData._preview_trial : makes use of this class
    """

    def __init__(self, shape, idx, dtype, dimord):
        self.shape = tuple(shape)
        self.idx = tuple(idx)
        self.dtype = dtype
        self.dimord = dimord

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
        return FauxTrial(shp, self.idx, self.dtype, self.dimord)

    @property
    def T(self):
        """
        Return a new `FauxTrial` instance with reversed dimensions
        (parroting the NumPy original :func:`numpy.transpose`)
        """
        return FauxTrial(
            self.shape[::-1], self.idx[::-1], self.dtype, self.dimord[::-1]
        )


class Selector:
    """
    Auxiliary class for data selection

    Parameters
    ----------
    data : Syncopy data object
        A non-empty Syncopy data object
    select : dict or :class:`~syncopy.shared.tools.StructDict` or None or str
        Dictionary or :class:`~syncopy.shared.tools.StructDict` with keys
        specifying data selectors. **Note**: some keys are only valid for certain types
        of Syncopy objects, e.g., "freq" is not a valid selector for an
        :class:`~syncopy.AnalogData` object. Supported keys are (please see
        :func:`~syncopy.selectdata` for a detailed description of each selector)

        * 'trials' : list (integers)
        * 'channel' : list (integers or strings), slice or range
        * 'toi' : list (floats)
        * 'toilim' : list (floats [tmin, tmax])
        * 'foi' : list (floats)
        * 'foilim' : list (floats [fmin, fmax])
        * 'taper' : list (integers or strings), slice or range
        * 'unit' : list (integers or strings), slice or range
        * 'eventid' : list (integers), slice or range

        Any property of `data` that is not specifically accessed via one of
        the above keys is taken as is, e.g., ``select = {'trials': [1, 2]}``
        selects the entire contents of trials no. 2 and 3, while
        ``select = {'channel': range(0, 50)}`` selects the first 50 channels
        of `data` across all defined trials. Consequently, if `select` is
        `None` or if ``select = "all"`` the entire contents of `data` is selected.

    Returns
    -------
    selection : Syncopy :class:`Selector` object
        An instance of this class whose main properties are either lists or slices
        to be used as (fancy) indexing tuples. Note that the properties `time`,
        `unit` and `eventid` are **by-trial** selections, i.e., list of lists
        and/or slices encoding per-trial sample-indices, e.g., ``selection.time[0]``
        is intended to be used with ``data.trials[selection.trial_ids[0]]``.
        Addditional class attributes of note:

        * `_useFancy` : bool

          If `True`, selection requires "fancy" (or "advanced") array indexing

        * `_dataClass` : str

          Class name of `data`

        * `_samplerate` : float

          Samplerate of `data` (only relevant for objects supporting time-selections)

        * `_timeShuffle` : bool

          If `True`, time-selection contains unordered/repeated time-points.

        * `_allProps` : list

          List of all selection properties in class

        * `_byTrialProps` : list

          List off by-trial selection properties (see above)

        * `_dimProps` : list

          List off trial-independent selection properties (computed as
          `self._allProps` minus `self._byTrialProps`)

    Notes
    -----
    Whenever possible, this class performs extensive input parsing to ensure
    consistency of provided selectors. Some exceptions to this rule include
    `toi` and `toilim`: depending on the size of `data` and the number of
    defined trials, `data.time` might generate a list of arrays of substantial
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
        if isinstance(select, str):
            if select == "all":
                select = {}
            else:
                raise SPYValueError(
                    legal="'all' or `None` or dict", varname="select", actual=select
                )
        if not isinstance(select, dict):
            raise SPYTypeError(select, "select", expected="dict")

        # Keep list of supported selectors in sync w/supported keywords of `selectdata`
        supported = data._selectionKeyWords
        # `selectdata` already throws out not supported keywords
        # so this is just a hard check when setting a selection via assignment
        if not set(select.keys()).issubset(supported):
            lgl = (
                "dict with one or all of the following keys: '"
                + "'".join(opt + "', " for opt in supported)[:-2]
            )
            act = (
                "dict with keys '" + "'".join(key + "', " for key in select.keys())[:-2]
            )
            raise SPYValueError(legal=lgl, varname="select", actual=act)

        # Save class of input object for posterity
        self._dataClass = data.__class__.__name__

        # Set up lists of (a) all selectable properties (b) trial-dependent ones
        # and (c) selectors independent from trials
        self._allProps = [
            "channel",
            "channel_i",
            "channel_j",
            "time",
            "freq",
            "taper",
            "unit",
            "eventid",
        ]
        self._byTrialProps = ["time", "unit", "eventid"]
        self._dimProps = list(self._allProps)
        for prop in self._byTrialProps:
            self._dimProps.remove(prop)

        # Special adjustment for `CrossSpectralData`: remove (invalid) `channel` property
        # from `_dimProps` (avoid pitfalls in code-blocks iterating over `_dimProps`)
        if self._dataClass == "CrossSpectralData":
            self._dimProps.remove("channel")

        # Assign defaults (trials are not a "real" property, handle it separately,
        # same goes for `trialdefinition`)
        self._trials = None
        self._trial_ids = None
        self._trialdefinition = None
        for prop in self._allProps:
            setattr(self, "_{}".format(prop), None)
        self._useFancy = False  # flag indicating whether fancy indexing is necessary
        self._samplerate = None  # for objects supporting time-selections
        self._timeShuffle = (
            False  # flag indicating whether time-points are repeated/unordered
        )

        # We first need to know which trials are of interest here (assuming
        # that any valid input object *must* have a `trials` attribute)
        self.trial_ids = (data, select)

        # Now set any possible selection attribute (depending on type of `data`)
        # Note: `trialdefinition` is set *after* harmonizing indexing selections
        # in `_make_consistent`
        for prop in self._allProps:
            setattr(self, prop, (data, select))

        # Ensure correct indexing: harmonize selections for `DiscreteData`-children
        # or convert everything to lists for use w/`np.ix_` if we ended up w/more
        # than 2 list selectors for `ContinuousData`-offspring
        self._make_consistent(data)

        # store for later re-application/modification
        self.select = select

        # create the Selector._get_trial helper
        self.create_get_trial(data)

    @property
    def trial_ids(self):
        """Index list of selected trials"""
        return self._trial_ids

    @trial_ids.setter
    def trial_ids(self, dataselect):
        data, select = dataselect
        trlList = list(range(len(data.trials)))
        trials = select.get("trials", None)
        vname = "select: trials"

        if isinstance(trials, str):
            if trials == "all":
                trials = None
            else:
                raise SPYValueError(
                    legal="'all' or `None` or list/array", varname=vname, actual=trials
                )
        if trials is not None:
            if np.issubdtype(type(trials), np.number):
                trials = [trials]
            try:
                array_parser(
                    trials,
                    varname=vname,
                    ntype="int_like",
                    hasinf=False,
                    hasnan=False,
                    lims=[0, len(data.trials)],
                    dims=1,
                )
            except Exception as exc:
                raise exc
            if not set(trials).issubset(trlList):
                lgl = "list/array of values b/w 0 and {}".format(trlList[-1])
                act = "Values b/w {} and {}".format(min(trials), max(trials))
                raise SPYValueError(legal=lgl, varname=vname, actual=act)
        else:
            trials = trlList
        self._trial_ids = list(trials) # ensure `trials` is a list cf. #180

    @property
    def trials(self):
        """
        Returns an Indexer indexing single trial arrays respecting the selection
        Indices are RELATIVE with respect to existing trial selections:

        >>> selection.trials[2]

        indexes the 3rd trial of `selection.trial_ids`

        Selections must be "simple": ordered and without repetitions
        """

        return Indexer(map(self._get_trial, self.trial_ids),
                       len(self.trial_ids)) if self.trial_ids is not None else None

    def create_get_trial(self, data):
        """ Closure to allow emulation of BaseData._get_trial"""

        # trl_id has to be part of selection for coherence
        def _get_trial(trl_id):
            if trl_id not in self.trial_ids:
                lgl = "a trial part of the selection"
                act = trl_id
                raise SPYValueError(lgl, "Selector.trials", act)
            # extract the selection respecting FauxTrial idx tuple
            # which has length len(data.dimord) or 2 if `data` is a DiscreteData instance
            trl_idx = data._preview_trial(trl_id).idx

            # now massage/validate it such that we can use it to
            # directly index the hdf5 dataset
            # tuple elements can only be lists or ordered slices, see concrete
            # `_preview_trial` implementations which generate those idx tuples
            # maybe TODO: allow fancy indexing like in the CR
            for i, dim_idx in enumerate(trl_idx):
                if isinstance(dim_idx, list):
                    # no fancy indexing, no repetitions
                    if len(set(dim_idx)) != len(dim_idx):
                        lgl = "simple selections w/o repetitions"
                        act = f"fancy selection with repetitions for selector {data.dimord[i]}"
                        raise SPYValueError(lgl, "Selector.trials", act)

                    # DiscreteData selections inherently re-order the sample dim. idx
                    # so these we sort, all others we need ordered
                    if 'discrete_data' in str(data.__class__):
                        # sorts in place!
                        dim_idx.sort()
                    elif np.any(np.diff(dim_idx) < 0):
                        lgl = "simple selection in ascending order"
                        act = f"fancy non-ordered selection of selector {data.dimord[i]}"
                        raise SPYValueError(lgl, "Selector.trials", act)
            # if we landed here all is good and we take
            # a leap of faith into the hdf5 dataset
            return data.data[trl_idx]

        # finally bind it to the Selector instance
        self._get_trial = _get_trial


    @property
    def channel(self):
        """List or slice encoding channel-selection"""
        return self._channel

    @channel.setter
    def channel(self, dataselect):
        data, select = dataselect
        chanSpec = select.get("channel")
        if self._dataClass == "CrossSpectralData":
            if chanSpec is not None:
                lgl = "`channel_i` and/or `channel_j` selectors for `CrossSpectralData`"
                raise SPYValueError(
                    legal=lgl, varname="select: channel", actual=data.__class__.__name__
                )
            else:
                return
        self._selection_setter(data, select, "channel")

    @property
    def channel_i(self):
        """List or slice encoding principal channel-pair selection"""
        return self._channel_i

    @channel_i.setter
    def channel_i(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "channel_i")

    @property
    def channel_j(self):
        """List or slice encoding principal channel-pair selection"""
        return self._channel_j

    @channel_j.setter
    def channel_j(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "channel_j")

    @property
    def time(self):
        """len(self.trial_ids) list of lists/slices of by-trial time-selections"""
        return self._time

    @time.setter
    def time(self, dataselect):

        # Unpack input and perform error-checking
        data, select = dataselect
        timeSpec = select.get("latency", None)
        checkLim = True
        checkInf = None
        vname = "select: latency"

        hasTime = hasattr(data, "time") or hasattr(data, "trialtime")
        if timeSpec is not None and hasTime is False:
            lgl = "Syncopy data object with time-dimension"
            raise SPYValueError(
                legal=lgl, varname=vname, actual=data.__class__.__name__
            )

        # If `data` has a `time` property, fill up `self.time`
        if hasTime:
            if isinstance(timeSpec, str):
                if timeSpec == "all":
                    timeSpec = None
                    select["latency"] = None
                else:
                    raise SPYValueError(
                        legal="'all' or `None` or list/array",
                        varname=vname,
                        actual=timeSpec,
                    )
            if timeSpec is not None:
                if np.issubdtype(type(timeSpec), np.number):
                    timeSpec = [timeSpec]
                    array_parser(
                        timeSpec, varname=vname, hasinf=checkInf, hasnan=False, dims=1
                    )
                # can only be 2-sequence [start, end]
                else:
                    if len(timeSpec) != 2:
                        lgl = "`select: latency` selection with two components"
                        act = "`select: latency` with {} components".format(
                            len(timeSpec)
                        )
                        raise SPYValueError(legal=lgl, varname=vname, actual=act)
                    if timeSpec[0] >= timeSpec[1]:
                        lgl = (
                            "`select: latency` selection with `latency[0]` < `latency[1]`"
                        )
                        act = "selection range from {} to {}".format(
                            timeSpec[0], timeSpec[1]
                        )
                        raise SPYValueError(legal=lgl, varname=vname, actual=act)
            timing = data._get_time(self.trial_ids, toi=None, toilim=select.get("latency"))

            # ---------------------------------------------------------------------------
            # this is legacy, might be needed later if ppl really want to "time shuffle"
            # to destroy any correlations and produce white noise from their data..
            # .. which is questionable

            # Determine, whether time-selection is unordered/contains repetitions
            # and set `self._timeShuffle` accordingly
            if timeSpec is not None:  # saves time for `timeSpec = None` "selections"
                for tsel in timing:
                    if isinstance(tsel, list) and len(tsel) > 1:
                        if np.diff(tsel).min() <= 0:
                            self._timeShuffle = True
                            break
            # ---------------------------------------------------------------------------

            # Assign timing selection and copy over samplerate from source object
            self._time = timing
            self._samplerate = data.samplerate

        else:
            return

    @property
    def trialdefinition(self):
        """len(self.trial_ids)-by-(3+) :class:`numpy.ndarray` encoding trial-information of selection"""
        return self._trialdefinition

    @trialdefinition.setter
    def trialdefinition(self, data):

        # Get original `trialdefinition` array for reference
        trl = data.trialdefinition

        # `DiscreteData`: simply copy relevant sample-count -> trial assignments,
        # for other classes build new trialdefinition array using `t0`-offsets
        if self._dataClass in ["SpikeData", "EventData"]:
            trlDef = trl[self.trial_ids, :]
        else:
            trlDef = np.zeros((len(self.trial_ids), trl.shape[1]))
            counter = 0
            for tk, trlno in enumerate(self.trial_ids):
                tsel = self.time[tk]
                if isinstance(tsel, slice):
                    start, stop, step = tsel.start, tsel.stop, tsel.step
                    if start is None:
                        start = 0
                    if stop is None:
                        trlTime = data._get_time([trlno], toilim=[-np.inf, np.inf])[0]
                        if isinstance(trlTime, list):
                            stop = np.max(trlTime)
                            # Avoid creating empty arrays for "static" `SpectralData` objects
                            if stop == start == 0:
                                stop += 1
                        else:
                            stop = trlTime.stop
                    if step is None:
                        step = 1
                    nSamples = (stop - start) / step
                    endSample = stop + data._t0[trlno]
                    t0 = int(endSample - nSamples)
                else:
                    nSamples = len(tsel)
                    if nSamples == 0:
                        t0 = 0
                    else:
                        t0 = data._t0[trlno]
                trlDef[tk, :3] = [counter, counter + nSamples, t0]
                trlDef[tk, 3:] = trl[trlno, 3:]
                counter += nSamples
        self._trialdefinition = trlDef

    @property
    def sampleinfo(self):
        """nTrials x 2 :class:`numpy.ndarray` of [start, end] sample indices"""
        if self._trialdefinition is not None:
            return self._trialdefinition[:, :2]
        else:
            return None

    @sampleinfo.setter
    def sampleinfo(self, sinfo):
        raise SPYError("Cannot set sampleinfo. Use `Selector.trialdefinition` instead.")

    @property
    def trialintervals(self):
        """nTrials x 2 :class:`numpy.ndarray` of [start, end] times in seconds """
        if self._trialdefinition is not None and self._samplerate is not None:
            # trial lengths in samples
            start_end = self.sampleinfo - self.sampleinfo[:, 0][:, None]
            start_end[:, 1] -= 1  # account for last time point
            # add offset and convert to seconds
            start_end = (start_end + self.trialdefinition[:, 2][:, None]) / self._samplerate
            return start_end
        else:
            return None

    @property
    def timepoints(self):
        """len(self.trial_ids) list of lists encoding actual (not sample indices!)
        timing information of unordered `toi` selections"""
        if self._timeShuffle:
            return [
                [
                    (tvec[tp] + self.trialdefinition[tk, 2]) / self._samplerate
                    for tp in range(len(tvec))
                ]
                for tk, tvec in enumerate(self.time)
            ]

    @property
    def freq(self):
        """List or slice encoding frequency-selection"""
        return self._freq

    @freq.setter
    def freq(self, dataselect):

        # Unpack input and perform error-checking
        data, select = dataselect
        freqSpec = select.get("frequency")
        checkLim = True
        checkInf = None
        hasFreq = hasattr(data, "freq")
        if freqSpec is not None and hasFreq is False:
            lgl = "Syncopy data object with freq-dimension"
            raise SPYValueError(
                legal=lgl, varname="frequency", actual=data.__class__.__name__
            )

        # If `data` has a `freq` property, fill up `self.freq`
        if hasFreq:
            if isinstance(freqSpec, str):
                if freqSpec == "all":
                    freqSpec = None
                    select["frequency"] = None
                else:
                    raise SPYValueError(
                        legal="'all' or `None` or float or list/array",
                        varname="frequency",
                        actual=freqSpec,
                    )
            if freqSpec is None:
                # select all
                self._freq = data._get_freq()

            else:
                if np.issubdtype(type(freqSpec), np.number):
                    freqSpec = [freqSpec]

                    array_parser(
                        freqSpec,
                        varname="frequency",
                        hasinf=False,
                        hasnan=False,
                        lims=[data.freq.min(), data.freq.max()],
                        dims=(1,),
                    )
                    # single frequency
                    self._freq = data._get_freq(foi=freqSpec)
                # frequency range [fmin, fmax]
                else:
                    array_parser(
                        freqSpec,
                        ntype="numeric",
                        varname="frequency",
                        hasnan=False,
                        lims=[data.freq.min(), data.freq.max()],
                        dims=(2,),
                    )
                    if freqSpec[0] >= freqSpec[1]:
                        lgl = (
                            "`select: frequency` selection with `frequency[0]` < `frequency[1]`"
                        )
                        act = "selection range from {} to {}".format(
                            freqSpec[0], freqSpec[1]
                        )
                        raise SPYValueError(legal=lgl, varname='frequency', actual=act)

                    self._freq = data._get_freq(foi=None, foilim=freqSpec)

    @property
    def taper(self):
        """List or slice encoding taper-selection"""
        return self._taper

    @taper.setter
    def taper(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "taper")

    @property
    def unit(self):
        """len(self.trial_ids) list of lists/slices of by-trial unit-selections"""
        return self._unit

    @unit.setter
    def unit(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "unit")

    @property
    def eventid(self):
        """len(self.trials) list of lists/slices encoding by-trial event-id-selection"""
        return self._eventid

    @eventid.setter
    def eventid(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "eventid")

    # Helper function to process provided selections
    def _selection_setter(self, data, select, selectkey):
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
        selectkey : str
            Name of key in `select` holding selection pertinent to identically
            named property in `data`

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
        target = getattr(data, selectkey, None)
        selector = "_{}".format(selectkey)
        vname = "select: {}".format(selectkey)
        if selection is not None and target is None:
            lgl = "Syncopy data object with {}".format(selectkey)
            raise SPYValueError(
                legal=lgl, varname=vname, actual=data.__class__.__name__
            )

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

            # Convert 'all' selections to take-all `None` (see next if below) and
            # put single-string selections into a list; same for single-scalar selections
            if isinstance(selection, str):
                if selection == "all":
                    selection = None
                else:
                    selection = [selection]
            elif np.issubdtype(type(selection), np.number):
                selection = [selection]

            # Take entire inventory sitting in `selectkey`
            if selection is None:
                if selectkey in ["unit", "eventid"]:
                    setattr(self, selector, [slice(None, None, 1)] * len(self.trial_ids))
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
                if np.isfinite(selLims[0]) and (
                    selLims[0] < -slcLims[1] or selLims[0] >= slcLims[1]
                ):
                    lgl = "selection range with min >= {}".format(slcLims[0])
                    act = "selection range starting at {}".format(selLims[0])
                    raise SPYValueError(legal=lgl, varname=vname, actual=act)
                if np.isfinite(selLims[1]) and (
                    selLims[1] > slcLims[1] or selLims[1] < -slcLims[1]
                ):
                    lgl = "selection range with max <= {}".format(slcLims[1])
                    act = "selection range ending at {}".format(selLims[1])
                    raise SPYValueError(legal=lgl, varname=vname, actual=act)

                # The 2d-arrays in `DiscreteData` objects require some additional hand-holding
                # performed by the respective `_get_unit` and `_get_eventid` class methods
                if selectkey in ["unit", "eventid"]:
                    if selection.start is selection.stop is None:
                        setattr(self, selector, [slice(None, None, 1)] * len(self.trial_ids))
                    else:
                        if isinstance(selection, slice):
                            if np.issubdtype(target.dtype, np.dtype("str").type):
                                target = np.arange(target.size)
                            selection = list(target[selection])
                        else:
                            selection = list(selection)
                        setattr(self, selector, getattr(data, "_get_" + selectkey)(self.trial_ids, selection))

                else:
                    if selection.start is selection.stop is None:
                        setattr(self, selector, slice(None, None, 1))
                    else:
                        if selection.step is None:
                            step = 1
                        else:
                            step = selection.step
                        setattr(
                            self, selector, slice(selection.start, selection.stop, step)
                        )

            # Selection is either a valid list/array or bust
            else:
                try:
                    array_parser(
                        selection,
                        varname=vname,
                        hasinf=hasinf,
                        hasnan=hasnan,
                        lims=arrLims,
                        dims=1,
                    )
                except Exception as exc:
                    raise exc
                selection = np.array(selection)
                if np.issubdtype(selection.dtype, np.dtype("str").type):
                    targetArr = target
                else:
                    targetArr = np.arange(target.size)
                if not set(selection).issubset(targetArr):
                    lgl = "list/array of {} existing names or indices".format(selectkey)
                    raise SPYValueError(legal=lgl, varname=vname)

                # Preserve order and duplicates of selection - don't use `np.isin` here!
                idxList = []
                for sel in selection:
                    idxList += list(np.where(targetArr == sel)[0])

                if selectkey in ["unit", "eventid"]:
                    setattr(self, selector, getattr(data, "_get_" + selectkey)(self.trial_ids, idxList))
                else:
                    # if possible, convert range-arrays (`[0, 1, 2, 3]`) to slices for better performance
                    if len(idxList) > 1:
                        steps = np.diff(idxList)
                        if steps.min() == steps.max() == 1:
                            idxList = slice(idxList[0], idxList[-1] + 1, 1)

                    # be careful w/pairwise list-channel selections in `CrossSpectralData` objects
                    # (that could not be converted to slices above)
                    if isinstance(idxList, list) and selectkey in [
                        "channel_i",
                        "channel_j",
                    ]:
                        if len(idxList) > 1:
                            err = "Multi-channel-pair selections not supported"
                            raise NotImplementedError(err)
                        idxList = idxList[0]

                    setattr(self, selector, idxList)

        else:
            return

    # Local helper that converts slice selectors to lists (if necessary)
    def _make_consistent(self, data):
        """
        Consolidate multi-selections

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
        lists/slices.
        For instances of :class:`~syncopy.datatype.continuous_data.ContinuousData`
        child classes (i.e., :class:`~syncopy.AnalogData` and :class:`~syncopy.SpectralData`
        objects) the integrity of conjoint multi-dimensional selections
        is ensured.
        For instances of :class:`~syncopy.datatype.discrete_data.DiscreteData`
        child classes (i.e., :class:`~syncopy.SpikeData` and :class:`~syncopy.EventData`
        objects), any selection (`unit`, `eventid`, `time` and `channel`) operates
        on the rows of the object's underlying `data` array. Thus, multi-selections
        need to be synchronized (e.g., a `unit` selection pointing to rows `[0, 1, 2]`
        and a `time` selection filtering rows `[1, 2, 3]` are combined to `[1, 2]`).

        See also
        --------
        numpy.ix_ : Mesh-construction for array indexing
        """

        # Harmonize selections for `DiscreteData`-children: all selectors are row-
        # indices, go through each trial and combine them
        if self._dataClass in ["SpikeData", "EventData"]:

            # Get relevant selectors (e.g., `self.unit` is `None` for `EventData`)
            actualSelections = []
            for selection in ["time", "eventid", "unit"]:
                if getattr(self, selection) is not None:
                    actualSelections.append(selection)

            # Compute intersection of "time" x "{eventid|unit|channel}" row-indices
            # per trial. BONUS: in `SpikeData` objects, `channels` are **not**
            # the same in all trials - ensure that channel selection propagates
            # correctly. After this step, `self.time` == `self.{unit|eventid}`
            if self._dataClass == "SpikeData":
                chanIdx = data.dimord.index("channel")
                wantedChannels = np.unique(data.data[:, chanIdx])[self.channel]
                chanPerTrial = []

            for tk, trialno in enumerate(self.trial_ids):
                trialArr = np.arange(np.sum(data.trialid == trialno))
                byTrialSelections = []
                for selection in actualSelections:
                    byTrialSelections.append(trialArr[getattr(self, selection)[tk]])

                # (try to) preserve unordered selections by processing them first
                areShuffled = [(np.diff(sel) <= 0).any() for sel in byTrialSelections]
                combiOrder = np.argsort(areShuffled)[::-1]
                combinedSelect = byTrialSelections[combiOrder[0]]
                for combIdx in combiOrder:
                    combinedSelect = [
                        elem
                        for elem in combinedSelect
                        if elem in byTrialSelections[combIdx]
                    ]

                # Keep record of channels present in trials vs. selected channels
                if self._dataClass == "SpikeData":
                    rawChanInTrial = data.data[data.trialid == trialno, chanIdx]
                    chanTrlIdx = [
                        ck
                        for ck, chan in enumerate(rawChanInTrial)
                        if chan in wantedChannels
                    ]
                    combinedSelect = [
                        elem for elem in combinedSelect if elem in chanTrlIdx
                    ]
                    chanPerTrial.append(rawChanInTrial[combinedSelect])

                # The usual list -> slice conversion (if possible)
                if len(combinedSelect) > 1:
                    selSteps = np.diff(combinedSelect)
                    if selSteps.min() == selSteps.max() == 1:
                        combinedSelect = slice(
                            combinedSelect[0], combinedSelect[-1] + 1, 1
                        )

                # Update selector properties
                for selection in actualSelections:
                    getattr(self, "_{}".format(selection))[tk] = combinedSelect

            # Ensure that `self.channel` is compatible w/provided selections: harmonize
            # `self.channel` with what is actually available in selected trials
            if self._dataClass == "SpikeData":
                availChannels = reduce(np.union1d, chanPerTrial)
                chanSelection = [
                    chan for chan in wantedChannels if chan in availChannels
                ]
                if len(chanSelection) > 1:
                    selSteps = np.diff(chanSelection)
                    if selSteps.min() == selSteps.max() == 1:
                        chanSelection = slice(
                            chanSelection[0], chanSelection[-1] + 1, 1
                        )
                self._channel = chanSelection

            # Finally, prepare new `trialdefinition` array
            self.trialdefinition = data

            return

        # Count how many lists we got
        listCount = 0
        for prop in self._dimProps:
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

        # If (on a by-trial basis) we have two or more lists, we need fancy indexing
        if listCount >= 2:
            self._useFancy = True

        # Finally, prepare new `trialdefinition` array for objects with `time` dimensions
        if self.time is not None:
            self.trialdefinition = data

        return

    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make selection readable from the command line
    def __str__(self):

        # Get list of print-worthy attributes
        ppattrs = [attr for attr in self.__dir__() if not attr.startswith("_")]
        # legacy, we have proper `Selector.trials` now
        ppattrs.remove('trial_ids')
        ppattrs.sort()

        # Construct dict of pretty-printable property info
        ppdict = {}
        for attr in ppattrs:
            val = getattr(self, attr)
            if val is not None and attr in self._byTrialProps:
                val = val[0]
            if isinstance(val, slice):
                if val.start is val.stop is None:
                    ppdict[attr] = "all {}{}, ".format(
                        attr, "s" if not attr.endswith("s") else ""
                    )
                elif val.start is None or val.stop is None:
                    ppdict[attr] = "{}-range, ".format(attr)
                else:
                    ppdict[attr] = "{0:d} {1:s}{2:s}, ".format(
                        int(np.ceil((val.stop - val.start) / val.step)),
                        attr,
                        "s" if not attr.endswith("s") else "",
                    )
            elif isinstance(val, (list, Indexer)):
                ppdict[attr] = "{0:d} {1:s}{2:s}, ".format(
                    len(val), attr, "s" if not attr.endswith("s") else ""
                )
            elif np.issubdtype(type(val), np.number):
                ppdict[attr] = "one {0:s}, ".format(attr)
            else:
                ppdict[attr] = ""

        # Construct string for printing
        msg = "Syncopy {} selector with ".format(self._dataClass)
        for pout in ppdict.values():
            msg += pout

        return msg[:-2]
