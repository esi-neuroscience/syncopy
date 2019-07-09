# -*- coding: utf-8 -*-
#
# Base class for all computational kernels in Syncopy
#
# Created: 2019-05-13 09:18:55
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-09 16:31:18>

# Builtin/3rd party package imports
import os
import sys
import time
import h5py
import numpy as np
from abc import ABC, abstractmethod
from copy import copy
from tqdm import tqdm
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)

# Local imports
from .parsers import get_defaults
from syncopy import __storage__, __dask__, __path__
if __dask__:
    import dask
    import dask.distributed as dd
    import dask.array as da
    import dask.bag as db

__all__ = []


class ComputationalRoutine(ABC):
    """Abstract class for encapsulating sequential/parallel algorithms

    A Syncopy compute kernel consists of a
    :class:`ComputationalRoutine`-subclass that binds a static
    :func:`computeFunction` and provides the class method
    :meth:`process_metadata`.

    Requirements for :meth:`computeFunction`:

    * First positional argument is a :class:`numpy.ndarray`, the keywords
      `chunkShape` and `noCompute` are supported
    * Returns a :class:`numpy.ndarray` if `noCompute` is `False` and expected 
      shape and numerical type of output array otherwise.

    Requirements for :class:`ComputationalRoutine`:

    * Child of :class:`ComputationalRoutine`, binds :func:`computeFunction`
      as static method
    * Provides class method :func:`process_data`

    For details on developing compute kernels for Syncopy, please refer
    to :doc:`../compute_kernels`.
    """

    # Placeholder: the actual workhorse
    @staticmethod
    def computeFunction(arr, *argv, chunkShape=None, noCompute=None, **kwargs):
        """Computational core routine

        Parameters
        ----------
        arr : :class:`numpy.ndarray`
           Numerical data from a single trial
        *argv : list
           Arbitrary list of positional arguments
        chunkShape : None or tuple
           Mandatory keyword. If not `None`, represents global block-size of
           processed trial.
        noCompute : None or bool
           Preprocessing flag. If `True`, do not perform actual calculation but
           instead return expected shape and :class:`numpy.dtype` of output
           array.
        **kwargs: dict
           Other keyword arguments.

        Returns
        -------
        out Shape : tuple, if ``noCompute == True``
           expected shape of output array
        outDtype : :class:`numpy.dtype`, if ``noCompute == True``
           expected numerical type of output array
        res : :class:`numpy.ndarray`, if ``noCompute == False``
           Result of processing input `arr`

        Notes
        -----
        This concrete method is a placeholder that is intended to be
        overloaded.

        See also
        --------
        ComputationalRoutine : Developer documentation: :doc:`../compute_kernels`.
        """
        return None

    def __init__(self, *argv, **kwargs):
        """
        Instantiate a :class:`ComputationalRoutine` subclass

        Parameters
        ----------
        *argv : list
           List of positional arguments passed on to :meth:`computeFunction`
        **kwargs : dict
           Keyword arguments passed on to :meth:`computeFunction`

        Returns
        -------
        obj : instance of :class:`ComputationalRoutine`-subclass
           Usable class instance for processing Syncopy data objects. 
        """
        self.defaultCfg = get_defaults(self.computeFunction)
        self.cfg = copy(self.defaultCfg)
        for key in set(self.cfg.keys()).intersection(kwargs.keys()):
            self.cfg[key] = kwargs[key]
        self.keeptrials = kwargs.get("keeptrials", True)
        self.argv = argv
        self.outputShape = None
        self.dtype = None
        self.vdsdir = None
        self.dsetname = None
        self.datamode = None

    def initialize(self, data):
        """
        Perform dry-run of calculation to determine output shape

        Parameters
        ----------
        data : syncopy data object
           Syncopy data object to be processed (has to be the same object 
           that is passed to :meth:`compute` for the actual calculation). 
        
        Returns
        -------
        Nothing : None
        
        Notes
        -----
        This class method **has** to be called prior to performing the actual
        computation realized in :meth:`computeFunction`. 

        See also
        --------
        compute : core routine performing the actual computation
        """

        # Get output chunk-shape and dtype of first trial
        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        chunkShape, self.dtype = self.computeFunction(data.trials[0],
                                                      *self.argv,
                                                      **dryRunKwargs)

        # For trials of unequal length, compute output chunk-shape individually
        # to identify varying dimension(s). The aggregate shape is computed
        # as max across all chunks
        if np.any([data._shapes[0] != sh for sh in data._shapes]):
            chunkShape = list(chunkShape)
            chk_list = [chunkShape]
            for tk in range(1, len(data.trials)):
                chk_list.append(list(self.computeFunction(data.trials[tk],
                                                          *self.argv,
                                                          **dryRunKwargs)[0]))
            chk_arr = np.array(chk_list)
            if np.unique(chk_arr[:, 0]).size > 1 and not self.keeptrials:
                err = "Averaging trials of unequal lengths in output currently not supported!"
                raise NotImplementedError(err)
            chunkShape = tuple(chk_arr.max(axis=0))
            self.outputShape = (chk_arr[:, 0].sum(),) + chunkShape[1:]
        else:
            self.outputShape = (len(data.trials) * chunkShape[0],) + chunkShape[1:]

        # Assign computed chunkshape to cfg dict
        self.cfg["chunkShape"] = chunkShape

        # Get data access mode (only relevant for parallel reading access)
        self.datamode = data.mode

    def compute(self, data, out, parallel=False, parallel_store=None,
                method=None, log_dict=None):
        """
        Central management and processing method

        Parameters
        ----------
        data : syncopy data object
           Syncopy data object to be processed (has to be the same object
           that was used by :meth:`initialize` in the pre-calculation 
           dry-run). 
        out : syncopy data object
           Empty object for holding results
        parallel : bool
           If `True`, processing is performed in parallel (i.e., 
           :meth:`computeFunction` is executed concurrently across trials). 
           If `parallel` is `False`, :meth:`computeFunction` is executed 
           consecutively trial after trial (i.e., the calculation realized
           in :meth:`computeFunction` is performed sequentially). 
        parallel_store : None or bool
           Flag controlling saving mechanism. If `None`,
           ``parallel_store = parallel``, i.e., the compute-paradigm 
           dictates the employed writing method. Thus, in case of parallel
           processing, results are written in a fully concurrent
           manner (each worker saves its own local result segment on disk as 
           soon as it is done with its part of the computation). If `parallel_store`
           is `False` and `parallel` is `True` the processing result is saved 
           sequentially using a mutex. If both `parallel` and `parallel_store`
           are `False` standard single-process HDF5 writing is employed for
           saving the result of the (sequential) computation. 
        method : None or str
           If `None` the predefined methods :meth:`compute_parallel` or
           :meth:`compute_sequential` are used to control the actual computation
           (specifically, calling :meth:`computeFunction`) depending on whether
           `parallel` is `True` or `False`, respectively. If `method` is a 
           string, it has to specify the name of an alternative (provided) 
           class method that is invoked using `getattr`. 
        log_dict : None or dict
           If `None`, the `cfg` and `log` properties of `out` are populated
           with the employed keyword arguments used in :meth:`computeFunction`. 
           Otherwise, `out`'s `cfg` and `log` properties are filled  with 
           items taken from `log_dict`. 
        
        Returns
        -------
        Nothing : None
           The result of the computation is available in `out` once 
           :meth:`compute` terminated successfully. 

        Notes
        -----
        This routine calls several other class methods to perform all necessary
        pre- and post-processing steps in a fully automatic manner without
        requiring any user-input. Specifically, the following class methods
        are invoked consecutively (in the given order):
        
        1. :meth:`preallocate_output` allocates a (virtual) HDF5 dataset 
           of appropriate dimension for storing the result
        2. :meth:`compute_parallel` (or :meth:`compute_sequential`) performs
           the actual computation via concurrently (or sequentially) calling
           :meth:`computeFunction`
        3. :meth:`save_distributed` writes the result of a parallel calculation
           to disk. If `parallel_store` is `False` a queue  for all workers
           is constructed to consecutively save their piece of the result. 
           Conversely, (if `parallel_store` is `True`) a write command is 
           issued to participating workers to concurrently save all result 
           segments to disk which are subsequently consolidated in a single 
           virtual HDF5 dataset. 
        4. :meth:`process_metadata` attaches all relevant meta-information to
           the result `out` after successful termination of the calculation
        5. :meth:`write_log` stores employed input arguments in `out.cfg`
           and `out.log` to reproduce all relevant computational steps that 
           generated `out`. 
        
        See also
        --------
        initialize : pre-calculation preparations
        preallocate_output : storage provisioning
        compute_parallel : concurrent computation using :meth:`computeFunction`
        compute_sequential : sequential computation using :meth:`computeFunction`
        save_distributed : sequential/concurrent storage of parallel computation results
        process_metadata : management of meta-information
        write_log : log-entry organization
        """

        # By default, use VDS storage for parallel computing
        if parallel_store is None:
            parallel_store = parallel

        # In some cases distributed dask workers suffer from spontaneous
        # dementia and forget the `sys.path` of their parent process. Fun!
        if parallel:
            def init_syncopy(dask_worker):
                spy_path = os.path.abspath(os.path.split(__path__[0])[0])
                if spy_path not in sys.path:
                    sys.path.insert(0, spy_path)
            client = dd.get_client()
            client.register_worker_callbacks(init_syncopy)

        # Check if trials actually fit into memory before we start computing
        if parallel:
            # SLURM: client.cluster.worker_memory
            # LOCAL: client.cluster.workers[0].memory_limit
            # cc.cluster._count_active_and_pending_workers()
            # int: bytes of memory for worker to use

        # Create HDF5 dataset of appropriate dimension
        self.preallocate_output(out, parallel_store=parallel_store)

        # The `method` keyword can be used to override the `parallel` flag
        if method is None:
            if parallel:
                computeMethod = self.compute_parallel
            else:
                computeMethod = self.compute_sequential
        else:
            computeMethod = getattr(self, "compute_" + method, None)

        # Perform actual computation
        result = computeMethod(data, out)

        # If computing is done in parallel, save distributed array and
        # reset data access mode
        if parallel:
            self.save_distributed(result, out, parallel_store)
            data.mode = self.datamode

        # Attach computed results to output object
        out.data = h5py.File(out._filename, mode="r+")[self.dsetname]

        # Store meta-data, write log and get outta here
        self.process_metadata(data, out)
        self.write_log(data, out, log_dict)

    def preallocate_output(self, out, parallel_store=False):
        """
        Storage allocation and provisioning

        Parameters
        ----------
        out : syncopy data object
           Empty object for holding results
        parallel_store : bool
           If `True`, a directory for virtual source containers is created 
           in Syncopy's temporary on-disk storage (defined by `syncopy.__storage__`). 
           Otherwise, a dataset of appropriate type and shape is allocated 
           in a new regular HDF5 file created inside Syncopy's temporary 
           storage folder. 

        Returns
        -------
        Nothing : None

        See also
        --------
        compute : management routine controlling memory pre-allocation
        """

        # The output object's type determines dataset name for result
        self.dsetname = out.__class__.__name__

        # In case parallel writing via VDS storage is requested, prepare
        # directory for by-chunk HDF5 containers
        if parallel_store:
            vdsdir = os.path.splitext(os.path.basename(out._filename))[0]
            self.vdsdir = os.path.join(__storage__, vdsdir)
            os.mkdir(self.vdsdir)

        # Create regular HDF5 dataset for sequential writing
        else:
            if not self.keeptrials:
                shp = self.cfg["chunkShape"]
            else:
                shp = self.outputShape
            with h5py.File(out._filename, mode="w") as h5f:
                h5f.create_dataset(name=self.dsetname,
                                   dtype=self.dtype, shape=shp)

    def compute_parallel(self, data, out):
        """
        Concurrent computing kernel

        Parameters
        ----------
        data : syncopy data object
           Syncopy data object to be processed
        out : syncopy data object
           Empty object for holding results

        Returns
        -------
        da_arr : :class:`dask.array`
           Distributed array comprised of by-worker results spread across
           the provided cluster. The shape of `da_arr` corresponds to the 
           shape expected by `out`. 

        Notes
        -----
        This method is essentially divided into two segments depending on 
        whether trials in `data` have equal length or not. Equidistant trials
        can be efficiently processed leveraging array-stacking, whereas 
        trials of unequal lengths have to be collected in an unstructured 
        iterable object. Note that this routine first builds an entire 
        parallel instruction tree and only kicks off execution on the cluster
        at the very end of the calculation command assembly. 
        

        See also
        --------
        compute : management routine invoking parallel/sequential compute kernels
        compute_sequential : serial processing counterpart of this method
        """

        # Either stack trials along a new axis (inserted on "the left")
        # or stack along first dimension
        if len(out.dimord) > len(data.dimord):
            stacking = da.stack
        else:
            stacking = da.vstack

        # Ensure `data` is openend read-only to permit concurrent reading access
        data.mode = "r"

        # Depending on equidistance of trials use dask arrays directly...
        if np.all([data._shapes[0] == sh for sh in data._shapes]):

            # Point to trials on disk by using delayed **static** method calls
            lazy_trial = dask.delayed(data._copy_trial, traverse=False)
            lazy_trls = [lazy_trial(trialno,
                                    data._filename,
                                    data.dimord,
                                    data.sampleinfo,
                                    data.hdr)
                         for trialno in range(len(data.trials))]

            # Stack trials along new (3rd) axis inserted on the left
            trl_block = stacking([da.from_delayed(trl, shape=data._shapes[sk],
                                                  dtype=data.data.dtype)
                                  for sk, trl in enumerate(lazy_trls)])

            # If result of computation has diverging chunk dimension, account for that:
            # chunkdiff > 0 : result is higher-dimensional
            # chunkdiff < 0 : result is lower-dimensional (!!!COMPLETELY UNTESTED!!!)
            # chunkdiff = 0 : result has identical dimension
            chunkdiff = len(self.cfg["chunkShape"]) - len(trl_block.chunksize)
            if chunkdiff > 0:
                mbkwargs = {"new_axis": list(range(chunkdiff))}
            elif chunkdiff < 0:
                mbkwargs = {"drop_axis": list(range(chunkdiff))}
            else:
                mbkwargs = {}

            # Use `map_blocks` to perform computation for each trial in the
            # constructed dask array
            result = trl_block.map_blocks(self.computeFunction,
                                          *self.argv, **self.cfg,
                                          dtype=self.dtype,
                                          chunks=self.cfg["chunkShape"],
                                          **mbkwargs)

            # Re-arrange dimensional order
            result = result.reshape(self.outputShape)

            # If wanted, average across trials (AnalogData are the only objects
            # that do not stack trials along a distinct time dimension...)
            if not self.keeptrials:
                if self.dsetname == "AnalogData":
                    result = result.reshape(len(data.trials), *self.cfg["chunkShape"]).mean(axis=0)
                else:
                    result = result.mean(axis=0, keepdims=True)

        # ...or work w/bags to account for diverging trial dimensions
        else:

            # Construct bag of trials
            trl_bag = db.from_sequence([trialno for trialno in
                                        range(len(data.trials))]).map(data._copy_trial,
                                                                      data._filename,
                                                                      data.dimord,
                                                                      data.sampleinfo,
                                                                      data.hdr)

            # Map each element of the bag onto ``computeFunction`` to get a new bag
            res_bag = trl_bag.map(self.computeFunction, *self.argv, **self.cfg)

            # The "result bag" contains elements of appropriate dimension that
            # can be stacked into a dask array again
            result = stacking([rbag for rbag in res_bag])

            # Re-arrange dimensional order
            result = result.reshape(self.outputShape)

            # FIXME: Placeholder
            if not self.keeptrials:
                pass

        # Submit the array to be processed by the worker swarm
        result = result.persist()
        
        return result

    def compute_sequential(self, data, out):
        """
        Sequential computing kernel
        
        Parameters
        ----------
        data : syncopy data object
           Syncopy data object to be processed
        out : syncopy data object
           Empty object for holding results

        Returns
        -------
        Nothing : None
        
        Notes
        -----
        This method most closely reflects classic iterative process execution:
        trials in `data` are passed sequentially to :meth:`computeFunction`,
        results are stored consecutively in a regular HDF5 dataset (that was 
        pre-allocated by :meth:`preallocate_output`). Thus, in contrast to 
        the parallel computing case, this routine performs both tasks,
        processing and saving of results. Since the calculation result is 
        immediately stored on disk, propagation of arrays across routines
        is avoided and memory usage is kept to a minimum. 

        See also
        --------
        compute : management routine invoking parallel/sequential compute kernels
        compute_parallel : concurrent processing counterpart of this method
        """

        # Iterate across trials and write directly to HDF5 container (flush
        # after each trial to avoid memory leakage) - if trials are not to
        # be preserved, compute average across trials manually to avoid
        # allocation of unnecessarily large dataset
        with h5py.File(out._filename, "r+") as h5f:
            dset = h5f[self.dsetname]
            cnt = 0
            idx = [slice(None)] * len(dset.shape)
            if self.keeptrials:
                for tk, trl in enumerate(tqdm(data.trials,
                                              desc="Computing...")):
                    res = self.computeFunction(trl,
                                               *self.argv,
                                               **self.cfg)
                    idx[0] = slice(cnt, cnt + res.shape[0])
                    dset[tuple(idx)] = res
                    cnt += res.shape[0]
                    h5f.flush()
            else:
                for trl in tqdm(data.trials, desc="Computing..."):
                    dset[()] = np.nansum([dset, self.computeFunction(trl,
                                                                     *self.argv,
                                                                     **self.cfg)],
                                         axis=0)
                    h5f.flush()
                dset[()] /= len(data.trials)

        return

    def save_distributed(self, da_arr, out, parallel_store=True):
        """
        Store result of parallel computation

        Parameters
        ----------
        da_arr : dask array
           Distributed array returned by :meth:`compute_parallel`
        out : syncopy data object
           Empty object for holding results
        parallel_store : bool
           If `True`, results are written in a fully concurrent manner (each 
           worker saves its own local result segment on disk as soon as it 
           is done with its part of the computation). Otherwise, the processing 
           result is saved sequentially using a mutex. 

        Returns
        -------
        Nothing : None

        Notes
        -----
        In case of parallel writing (i.e., ``parallel_store = True``), the
        chunks of `da_arr` are mapped onto the helper routine 
        :meth:`_write_parallel` which is executed simultaneously by all workers. 
        This helper function creates an individual HDF5 container (in the
        directory that was previously created by :meth:`preallocate_output`)
        for each chunk of `da_arr`. After all blocks of `da_arr` have 
        been written concurrently by all participating workers, the generated
        by-chunk containers are consolidated into a single :class:`h5py.VirtualLayout` which 
        is subsequently used to allocate a virtual dataset inside a newly
        created HDF5 file (located in Syncopy's temporary storage folder). 

        Conversely, if `parallel_store` is `False`, `da_arr` is written 
        sequentially chunk by chunk (using the local helper method 
        :meth:`_write_sequential` and a distributed mutex for access control 
        to prevent write collisions) to an existing HDF5 container that was
        created by :meth:`preallocate_output`. 

        See also
        --------
        preallocate_output : provide storage prior to calculation
        _write_parallel : local helper for concurrent writing of `da_arr`'s chunks
        _write_sequential : local helper for serial writing of `da_arr`'s chunks
        """

        # Either write chunks fully parallel...
        nchk = len(da_arr.chunksize)
        if parallel_store:

            # Map `da_arr` chunk by chunk onto ``_write_parallel``
            writers = da_arr.map_blocks(self._write_parallel, nchk, self.vdsdir,
                                        dtype="int", chunks=(1,) * nchk)

            # Make sure that all futures are actually executed (i.e., data is written
            # to the container)
            futures = dd.client.futures_of(writers.persist())
            while any(f.status == 'pending' for f in futures):
                time.sleep(0.1)

            # Construct virtual layout from created HDF5 files
            layout = h5py.VirtualLayout(shape=da_arr.shape, dtype=da_arr.dtype)
            for k in range(len(futures)):
                fname = os.path.join(self.vdsdir, "{0:d}.h5".format(k))
                with h5py.File(fname, "r") as h5f:
                    idx = tuple([slice(*dim) for dim in h5f["idx"]])
                    shp = h5f["chk"].shape
                layout[idx] = h5py.VirtualSource(fname, "chk", shape=shp)

            # Use generated layout to create virtual dataset
            with h5py.File(out._filename, mode="w") as h5f:
                h5f.create_virtual_dataset(self.dsetname, layout)

        # ...or use a mutex to write to a single container sequentially
        else:

            # Initialize distributed lock
            lck = dd.lock.Lock(name='writer_lock')

            # Map `da_arr` chunk by chunk onto ``_write_sequential``
            writers = da_arr.map_blocks(self._write_sequential, nchk, out._filename,
                                        self.dsetname, lck, dtype="int",
                                        chunks=(1, 1))

            # Make sure that all futures are actually executed (i.e., data is written
            # to the container)
            futures = dd.client.futures_of(writers.persist())
            while any(f.status == 'pending' for f in futures):
                time.sleep(0.1)

    def write_log(self, data, out, log_dict=None):
        """
        Processing of output log

        Parameters
        ----------
        data : syncopy data object
           Syncopy data object that has been processed
        out : syncopy data object
           Syncopy data object holding calculation results
        log_dict : None or dict
           If `None`, the `cfg` and `log` properties of `out` are populated
           with the employed keyword arguments used in :meth:`computeFunction`. 
           Otherwise, `out`'s `cfg` and `log` properties are filled  with 
           items taken from `log_dict`. 
        
        Returns
        -------
        Nothing : None
 
        See also
        --------
        process_metadata : Management of meta-information
        """

        # Copy log from source object and write header
        out._log = str(data._log) + out._log
        logHead = "computed {name:s} with settings\n".format(name=self.computeFunction.__name__)

        # Either use `computeFunction`'s keywords (sans implementation-specific
        # stuff) or rely on provided `log_dict` dictionary for logging/`cfg`
        if log_dict is None:
            cfg = dict(self.cfg)
            for key in ["noCompute", "chunkShape"]:
                cfg.pop(key)
        else:
            cfg = log_dict

        # Write log and set `cfg` prop of `out`
        logOpts = ""
        for k, v in cfg.items():
            logOpts += "\t{key:s} = {value:s}\n".format(key=k,
                                                        value=str(v) if len(str(v)) < 80
                                                        else str(v)[:30] + ", ..., " + str(v)[-30:])
        out.log = logHead + logOpts
        out.cfg = cfg

    @staticmethod
    def _write_parallel(chk, nchk, vdsdir, block_info=None):
        """
        Creates a HDF5 container and saves a dask-array chunk in it
        
        Parameters
        ----------
        chk : :class:`numpy.ndarray`
           Chunk of parent distributed dask array
        nchk : int
           Running counter reflecting current chunk-number within the parent
           dask array. 
        vdsdir : str
           Full path to an existing (writable) folder for creating an HDF5
           container
        block_info : None or dict
           Special keyword used by :func:`dask.array.map_blocks` to propagate
           layout information of the current chunk with respect to the parent
           dask array. 

        Returns
        -------
        chunklocation : tuple
           Subscript index encoding the location of the processed chunk within
           its parent dask array

        Notes
        -----
        Using information gathered from `block_info`, this routine writes 
        `chk` to an identically named dataset in a HDF5 container that is 
        created in `vdsdir`. 
        In addition, the location of the current chunk within the parent 
        dask array is stored in a second dataset "idx". In this manner, 
        concurrently executing this routine spawns a family of HDF5 containers 
        that each contain not only a chunk of the original dask array but also
        the necessary topographic information to reconstruct the original 
        array solely from the generated HDF5 files. 

        See also
        --------
        save_distributed : management routine for storing result of concurrent calculation
        _write_sequential : sequential counterpart of this method
        """

        # Convert chunk-location tuple to 1D index for numbering current
        # HDF5 container and get index-location of current chunk in array
        cnt = np.ravel_multi_index(block_info[0]["chunk-location"],
                                   block_info[0]["num-chunks"])
        idx = block_info[0]["array-location"]

        # Save data and its original location within the array
        fname = os.path.join(vdsdir, "{0:d}.h5".format(cnt))
        with h5py.File(fname, "w") as h5f:
            h5f.create_dataset('chk', data=chk)
            h5f.create_dataset('idx', data=idx)
            h5f.flush()

        return (cnt,) * nchk

    @staticmethod
    def _write_sequential(chk, nchk, h5name, dsname, lck, block_info=None):
        """
        Saves chunk of dask-array in existing HDF5 container
        
        Parameters
        ----------
        chk : :class:`numpy.ndarray`
           Chunk of parent distributed dask array
        nchk : int
           Running counter reflecting current chunk-number within the parent
           dask array
        h5name : str
           Full path to existing HDF5 container
        dsname : str
           Name of existing dataset in container specified by `h5name`
        lck : :mod:`dask.distributed.lock`
           Distributed mutex controlling write-access to HDF5 container 
           specified by `h5name`
        block_info : None or dict
           Special keyword used by :func:`dask.array.map_blocks` to propagate
           layout information of the current chunk with respect to the parent
           dask array. 

        Returns
        -------
        chunklocation : tuple
           Subscript index encoding the location of the processed chunk within
           its parent dask array

        Notes
        -----
        This routine writes `chk` to the (already existing) dataset `dsname` 
        of the HDF5 container specified by `h5name` relying on location 
        information provided by `block_info`. Saving of `chk` is serialized 
        by acquiring the distributed mutex `lck` which locks the HDF5 container 
        until the write process is completed. 

        See also
        --------
        save_distributed : management routine for storing result of concurrent calculation
        _write_parallel : concurrent counterpart of this method
        """

        # Convert chunk-location tuple to 1D index for numbering current
        # HDF5 container and get index-location of current chunk in array
        cnt = np.ravel_multi_index(block_info[0]["chunk-location"],
                                   block_info[0]["num-chunks"])
        idx = tuple([slice(*dim) for dim in block_info[0]["array-location"]])

        # (Try to) acquire lock
        lck.acquire()
        while not lck.locked():
            time.sleep(0.05)

        # Open container and write current chunk
        with h5py.File(h5name, "r+") as h5f:
            h5f[dsname][idx] = chk

        # Release distributed lock
        lck.release()
        while lck.locked():
            time.sleep(0.05)

        return (cnt,) * nchk

    @abstractmethod
    def process_metadata(self, data, out):
        """
        Meta-information manager

        Parameters
        ----------
        data : syncopy data object
           Syncopy data object that has been processed
        out : syncopy data object
           Syncopy data object holding calculation results

        Returns
        -------
        Nothing : None

        Notes
        -----
        This routine is an abstract method and is thus intended to be overloaded. 
        Consult the developer documentation (:doc:`../compute_kernels`) for 
        further details. 

        See also
        --------
        write_log : Logging of calculation parameters
        """
        pass
