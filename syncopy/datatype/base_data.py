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
from hashlib import blake2b
from itertools import chain
from types import GeneratorType
import shutil
import numpy as np
import h5py
import scipy as sp

# Local imports
import syncopy as spy
from .selector import Selector
from .util import TrialIndexer
from .methods.arithmetic import _process_operator
from .methods.selectdata import selectdata
from .methods.show import show
from syncopy.shared.tools import SerializableDict
from syncopy.shared.parsers import (
    array_parser,
    io_parser,
    filename_parser,
    data_parser,
)
from syncopy.shared.errors import SPYInfo, SPYTypeError, SPYValueError, SPYError, SPYWarning
from syncopy.datatype.methods.definetrial import definetrial as _definetrial
from syncopy import __version__, __storage__, __acme__, __sessionid__

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

    def _unregister_dataset(self, propertyName, del_from_file=True, del_attr=True):
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
                Whether to remove the dataset named 'propertyName' from the backing hdf5 file on disk.
            del_attr: bool
                Whether to remove the dataset attribute from the Syncopy data object.
        """
        if del_attr:
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
            dataIn : str, np.ndarray, or h5py.Dataset, list, generator
                Filename, array, list of arrays, list of syncopy objects,
                HDF5 dataset or generator object to be stored in property
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
            GeneratorType: self._set_dataset_property_with_generator,
            list: self._set_dataset_property_with_list,
            str: self._set_dataset_property_with_str,
            np.ndarray: self._set_dataset_property_with_ndarray,
            h5py.Dataset: self._set_dataset_property_with_dataset,
            type(None): self._set_dataset_property_with_none,
        }
        try:
            supportedSetters[type(inData)](inData, propertyName, ndim=ndim)
        except KeyError:
            msg = "filename of HDF5 file, HDF5 dataset, list, generator or NumPy array"
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
                raise SPYValueError(legal=lgl, varname=propertyName, actual=act)
            if prop.shape != inData.shape:
                lgl = "dataset with shape {}".format(str(prop.shape))
                act = "data with shape {}".format(str(inData.shape))
                raise SPYValueError(legal=lgl, varname=propertyName, actual=act)
            if prop.dtype != inData.dtype:
                lgl = "dataset of type {}".format(prop.dtype.name)
                act = "data of type {}".format(inData.dtype.name)
                raise SPYValueError(legal=lgl, varname=propertyName, actual=act)
            prop[...] = inData

        # or create backing file on disk
        else:
            if self.filename is None:
                self.filename = self._gen_filename()

            if propertyName not in self._hdfFileDatasetProperties:
                if getattr(self, "_" + propertyName) is not None and not isinstance(getattr(self, "_" + propertyName), h5py.Dataset):
                    raise SPYValueError(legal="propertyName that does not clash with existing attributes",
                                        varname='propertyName', actual=propertyName)

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

        if propertyName == "data":
            # Ensure dataset has right no. of dimensions
            if inData.ndim != ndim:
                lgl = "{}-dimensional data".format(ndim)
                act = "{}-dimensional HDF5 dataset".format(inData.ndim)
                raise SPYValueError(legal=lgl, varname="data", actual=act)

            self._check_dataset_property_discretedata(inData)
            self.filename = inData.file.filename
        else:
            # creates hidden attribute behind the property on the fly
            if not hasattr(self, "_" + propertyName):
                setattr(self, "_" + propertyName, None)

        self._mode = inData.file.mode
        setattr(self, "_" + propertyName, inData)

    def _set_dataset_property_with_list(self, inData, propertyName, ndim):
        """Set a dataset property with a list of NumPy arrays or syncopy
           data objects.

        Parameters
        ----------
            inData : list
                list of :class:`numpy.ndarray`s or syncopy data objects.
            propertyName : str
                Name of the property to be filled with the concatenated array
                Can only be ``data`` for syncopy objects to be concatenated.
            ndim : int
                Number of expected array dimensions.
        """

        # first catch empty lists
        if len(inData) == 0:
            msg = ("Trying to set syncopy data with empty list, "
                   f"setting `{propertyName}` dataset to `None`!")
            SPYWarning(msg)
            self._set_dataset_property_with_none(None, propertyName, ndim)
            return

        # check if we have consistent list entries
        check = np.sum([isinstance(val, np.ndarray) for val in inData])
        # check has to be either 0 (no arrays) or len(inData) (all arrays)
        if check != 0 and check != len(inData):
            lgl = "consistent data types"
            act = "mix of NumPy arrays and other data types"
            raise SPYValueError(lgl, "data", act)

        # as we catched empty lists above, and checked against inconsistent
        # types we can do a hard instance check on the 1st entry only
        if isinstance(inData[0], np.ndarray):
            self._set_dataset_property_with_array_list(inData,
                                                       propertyName,
                                                       ndim)
        # alternatively must be all syncopy data objects
        else:
            for val in inData:
                data_parser(val)

            # this should not happen, as all derived classes hardcoded this in their setters
            if propertyName != 'data':
                raise SPYError(f"Cannot concatenate syncopy objects for dataset {propertyName}")

            # if we landed here all is clear
            self._set_dataset_property_with_spy_list(inData, ndim)

    def _set_dataset_property_with_array_list(self, inData, propertyName, ndim):
        """Set a dataset property with a list of NumPy arrays.

        Parameters
        ----------
            inData : list
                list of :class:`numpy.ndarray`s
                Each array corresponds to a trial. Arrays are stacked
                together to fill dataset.
            propertyName : str
                Name of the property to be filled with the concatenated array
            ndim : int
                Number of expected array dimensions.
        """

        # Check list entries: must be numeric, finite NumPy arrays
        for val in inData:
            array_parser(val, varname="data", hasinf=False, dims=ndim)

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
                act = "NumPy arrays with mismatching shapes"
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

    def _set_dataset_property_with_spy_list(self, inData, ndim):
        """Set the `data` dataset property from a list of compatible
           syncopy data objects.
           This implements concatenation along trials of syncopy data objects.

        Parameters
        ----------
        inData : list
            Non empty list of syncopy data objects, e.g. :class:`~syncopy.AnalogData`.
            Trials are stacked together to fill dataset.
        ndim : int
            Number of expected array dimensions.
        """

        # --  dataset shape and object attribute inquiries --

        # take the 1st non-empty object as reference
        i_ref = 0   # to avoid "probably undefined loop variable" linter warning
        for i_ref, spy_obj in enumerate(inData):
            if spy_obj.data is None:
                SPYWarning(f"Skipping empty dataset {spy_obj.filename} for concatenation")
                continue
            else:
                spy_obj_ref = spy_obj
                shape_ref = np.array(spy_obj.data.shape)

                if len(shape_ref) != ndim:
                    lgl = f"dataset with dimension of {ndim}"
                    act = f"got dataset with dimension {len(shape_ref)}"
                    raise SPYValueError(lgl, 'data', act)

                stacking_dim_ref = spy_obj._stackingDim
                # collect remaining attribute names like channel, freq, etc.
                attr_ref = [attr for attr in spy_obj._hdfFileAttributeProperties
                            if not attr.startswith('_')]
                # boolean array to index non-stacking dimensions
                # for strict shape comparison
                bvec = np.ones(shape_ref.size, dtype=bool)
                bvec[stacking_dim_ref] = False
                break

        # now loop again and check against all others
        lgl = "compatible syncopy objects for concatenation"
        stack_count = 0
        for spy_obj in inData[i_ref:]:

            if spy_obj.selection is not None:
                SPYWarning("In place selections will be ignored for concatenation!")

            if spy_obj.data is None:
                SPYWarning(f"Skipping empty dataset {spy_obj.filename} for concatenation")
                continue

            if spy_obj._stackingDim != stacking_dim_ref:
                act = f"different stacking dimensions, {stacking_dim_ref} and {spy_obj._stackingDim}"
                raise SPYValueError(lgl, 'data', act)

            # catch mismatching dimensions (2d vs. 3d)
            if len(shape_ref) != len(spy_obj.data.shape):
                act = f"mismatching shapes, {tuple(shape_ref)} and {spy_obj.data.shape}"
                raise SPYValueError(lgl, 'data', act)

            # shape tuple gets casted by numpy for array subtraction
            if not np.all((shape_ref - spy_obj.data.shape)[bvec] == 0):
                act = f"mismatching shapes, {tuple(shape_ref)} and {spy_obj.data.shape}"
                raise SPYValueError(lgl, 'data', act)

            # check attributes like channel, freq, etc.
            # this also catches incompatible syncopy data types with same ndim,
            # e.g. SpectralData and CrossSpectralData
            for attr in spy_obj._hdfFileAttributeProperties:
                if attr.startswith('_'):
                    continue

                attr_val = getattr(spy_obj, attr, None)
                if attr_val is None or attr not in attr_ref:
                    act = f"missing attribute `{attr}` in {spy_obj.filename}"
                    raise SPYValueError(lgl, 'data', act)
                # now hard check values, should be all arrays/sequences
                # we want identical channel label, freq axis and so on..
                if not np.all(getattr(spy_obj_ref, attr) == attr_val):
                    act = f"different attribute values for `{attr}`"
                    raise SPYValueError(lgl, 'data', act)

            # finally increment stack count
            stack_count += spy_obj.data.shape[stacking_dim_ref]

        # now we have all we need to compute
        # the shape of the concatenated object
        res_shape = shape_ref
        res_shape[stacking_dim_ref] = stack_count

        # finally create the chained trial generator
        trl_gen = chain(*[spy_obj.trials for spy_obj in inData])

        # this setter is only valid for empty (new) syncopy objects
        # hence it should be fine to potentially re-define the dimord here
        self._stackingDimLabel = spy_obj_ref._stackingDimLabel

        # and route through the generator setter
        self._set_dataset_property_with_generator(trl_gen,
                                                  propertyName='data',
                                                  ndim=len(res_shape),
                                                  shape=res_shape)

        # -- set attribute properties --

        # attach dummy selection to reference object
        # for easy propagation of properties
        spy.selectdata(spy_obj_ref, inplace=True)

        # Get/set dimensional attributes
        for prop in spy_obj_ref.selection._dimProps:
            selection = getattr(spy_obj_ref.selection, prop)
            if selection is not None:
                if np.issubdtype(type(selection), np.number):
                    selection = [selection]
                setattr(self, prop, getattr(spy_obj_ref, prop)[selection])

        self.samplerate = spy_obj_ref.samplerate
        spy_obj_ref.selection = None

    def _set_dataset_property_with_generator(self, gen,
                                             propertyName,
                                             ndim,
                                             shape=None):
        """
        Create a dataset from a generator yielding (single trial) numpy arrays.
        If `shape` is not given fall back to HDF5 resizable datasets along
        the stacking dimension.

        Expects empty property - will not try to overwrite datasets with generators!

        Parameters
        ----------
        gen : generator
            Generator yielding (single trial) numpy arrays. Their shapes
            have to match except along the `stacking_dim`
        ndim : int
            Number of dimensions of the numpy arrays
        propertyName : str
            The name of the property which manages the dataset
        shape : tuple
            The final shape of the hdf5 dataset. If left at `None`,
            the dataset will be resized along the stacking dimension
            for every trial drawn from the generator
        """

        if propertyName not in self._hdfFileDatasetProperties:
            raise SPYValueError(legal=f"one of {self._hdfFileDatasetProperties}",
                                varname='propertyName', actual=propertyName)

        # If there is existing data, get out
        if isinstance(getattr(self, "_" + propertyName), h5py.Dataset):
            lgl = "empty syncopy object"
            act = "non-empty syncopy object"
            raise SPYValueError(lgl, 'data', act)

        # look at 1st trial to determine fixed dimensions
        try:
            trial1 = next(gen)
        except StopIteration:
            lgl = "non-exhausted generator"
            act = "exhausted generator"
            raise SPYValueError(lgl, 'data', act)

        shape1 = list(trial1.shape)  # initial shape

        # further generated arrays will be checked against shape1
        if len(shape1) != ndim:
            lgl = f"arrays of dimension {ndim}"
            act = f"got array with dimension {len(shape1)}"
            raise SPYValueError(lgl, 'data', act)

        # boolean array to index non-stacking dimensions
        # for strict shape comparison
        bvec = np.ones(len(shape1), dtype=bool)
        bvec[self._stackingDim] = False

        # prepare to resize hdf5
        if shape is None:
            shape = shape1
            maxshape = shape.copy()
            maxshape[self._stackingDim] = None
            resize = True
        else:
            maxshape = None
            resize = False

        # construct slicing index
        stack_idx = [np.s_[:] for _ in range(len(shape))]

        # -- write data --
        stack_count = 0
        trlSamples = []  # for constructing the trialdefinition
        with h5py.File(self.filename, "w") as h5f:
            dset = h5f.create_dataset(propertyName,
                                      shape=shape,
                                      maxshape=maxshape,
                                      dtype=trial1.dtype)

            # we have to plug in the 1st trial already generated
            stack_step = trial1.shape[self._stackingDim]
            stack_idx[self._stackingDim] = np.s_[0:stack_step]
            dset[tuple(stack_idx)] = trial1
            stack_count += stack_step
            trlSamples.append(stack_step)

            # now stream through the arrays from the generator
            for trial in gen:

                # check shape except stacking dim
                if not np.all((shape1 - np.array(trial.shape))[bvec] == 0):
                    lgl = "compatible trial shapes"
                    act = f"mismatching shapes, {tuple(shape1)} and {trial.shape}"
                    raise SPYValueError(lgl, 'data', act)

                stack_step = trial.shape[self._stackingDim]
                # we have to resize for every trial if no total shape was given
                if resize:
                    dset.resize(stack_count + stack_step, axis=self._stackingDim)
                stack_idx[self._stackingDim] = np.s_[stack_count:stack_count + stack_step]
                dset[tuple(stack_idx)] = trial
                stack_count += stack_step
                trlSamples.append(stack_step)

            setattr(self, '_' + propertyName, dset)

        self._reopen()

        # -- construct trialdefinition --

        if propertyName == 'data':
            si = np.r_[0, np.cumsum(trlSamples)]
            sampleinfo = np.column_stack([si[:-1], si[1:]])
            trialdefinition = np.column_stack([sampleinfo, np.zeros(len(sampleinfo))])
            if self.samplerate is not None:
                # set standard offset to -1s
                trialdefinition[:, 2] = -self.samplerate

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
            "Cannot set sampleinfo. Use `BaseData.trialdefinition` instead."
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

        if self.sampleinfo is not None:
            trial_ids = list(range(self.sampleinfo.shape[0]))
            # this is cheap as it just initializes a list-like object
            # with no real data and/or computation!
            return TrialIndexer(self, trial_ids)
        else:
            return None

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
            "Cannot set trialinfo. Use `BaseData.trialdefinition` or `syncopy.definetrial` instead."
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

        # keep all datasets alive and open
        if self._persistent_hdf5:
            return

        # close hdf5 file
        for propertyName in self._hdfFileDatasetProperties:
            prop = getattr(self, "_" + propertyName)
            if prop is not None:
                try:
                    prop.file.close()
                # can happen if the file was deleted elsewhere
                # or we exit un-gracefully from some undefined state
                except (ValueError, ImportError, TypeError, AttributeError):
                    pass

        # remove from file system
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
        Keys of kwargs are the datasets from _hdfFileDatasetProperties, and
        kwargs must *only* include datasets for which a property with a setter exists.

        1. filename + data = create HDF5 file at filename with data in it
        2. data only

        """

        # each instance needs its own cfg!
        self._cfg = {}
        self._info = SerializableDict()

        # set to `True` to keep backing hdf5 alive
        # when the destructor is hit
        self._persistent_hdf5 = False

        # Initialize hidden attributes
        for propertyName in self._hdfFileDatasetProperties:
            setattr(self, "_" + propertyName, None)

        self._selector = None

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

