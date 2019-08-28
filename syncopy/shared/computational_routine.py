# -*- coding: utf-8 -*-
# 
# Base class for all computational kernels in Syncopy
# 
# Created: 2019-05-13 09:18:55
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-08-28 17:06:56>

# Builtin/3rd party package imports
import os
import sys
import time
import psutil
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
from syncopy.shared.errors import SPYIOError, SPYValueError
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
        self.sourceLayout = None
        self.targetLayout = None
        self.targetShapes = None
        self.chunkMem = None
        self.vdsdir = None
        self.dsetname = None
        self.datamode = None
        self.sleeptime = 0.1

    def initialize(self, data, chan_per_worker=None, timeout=300):
        """
        Perform dry-run of calculation to determine output shape

        Parameters
        ----------
        data : syncopy data object
           Syncopy data object to be processed (has to be the same object 
           that is passed to :meth:`compute` for the actual calculation). 
        timeout : int
           Number of seconds to wait for saving-mutex to be acquired (only 
           relevant in case of concurrent processing in combination with 
           sequential writing of results)
        
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
        
        # Prepare dryrun arguments and determine geometry of trials in output
        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        chk_list = []
        dtp_list = []
        # trials = []  # FIXME: FauxTrial preps
        for tk in range(len(data.trials)):
            trial = data.trials[tk]  # FIXME: this HAS to be replaced w/_preview_trial calls...                
            # trial = data._preview_trial(tk)
            chunkShape, dtype = self.computeFunction(trial, 
                                                     *self.argv, 
                                                     **dryRunKwargs)
            chk_list.append(list(chunkShape))
            dtp_list.append(dtype)
            # trials.append(trial)
            
        # The aggregate shape is computed as max across all chunks                    
        chk_arr = np.array(chk_list)
        if np.unique(chk_arr[:, 0]).size > 1 and not self.keeptrials:
            err = "Averaging trials of unequal lengths in output currently not supported!"
            raise NotImplementedError(err)
        if np.any([dtp_list[0] != dtp for dtp in dtp_list]):
            lgl = "unique output dtype"
            act = "{} different output dtypes".format(np.unique(dtp_list).size)
            raise SPYValueError(legal=lgl, varname="dtype", actual=act)
        chunkShape = tuple(chk_arr.max(axis=0))
        self.outputShape = (chk_arr[:, 0].sum(),) + chunkShape[1:]
        self.cfg["chunkShape"] = chunkShape
        self.dtype = np.dtype(dtp_list[0])

        # Ensure channel parallelization can be done at all
        if chan_per_worker is not None and "channel" not in data.dimord:
            print("Syncopy core - compute: WARNING >> input object does not " +\
                  "contain `channel` dimension for parallelization! <<")
            chan_per_worker = None
        if chan_per_worker is not None and self.keeptrials is False:
            print("Syncopy core - compute: WARNING >> trial-averaging does not " +\
                  "support channel-block parallelization! <<")
            chan_per_worker = None
            
        # Allocate control variables
        # trial = trials[0]  # FIXME: FauxTrial preps
        trial = data.trials[0]
        chunkShape0 = chk_arr[0, :]
        lyt = [slice(0, stop) for stop in chunkShape0]
        geo = list(chunkShape0) 
        sourceLayout = []
        targetLayout = []
        targetShapes = []

        # FIXME: will be obsolte w/FauxTrial
        # >>> START
        idx = [slice(None)] * len(data.dimord)  
        sid = data.dimord.index("time")
        trlslice = slice(int(data.sampleinfo[0, 0]), int(data.sampleinfo[0, 1]))
        # >>> STOP
        
        # If parallelization across channels is requested the first trial is 
        # split up into several chunks that need to be processed/allocated
        if chan_per_worker is not None:

            # Set up channel-chunking
            nChannels = data.channel.size
            rem = int(nChannels % chan_per_worker)        
            n_blocks = [chan_per_worker] * int(nChannels//chan_per_worker) + [rem] * int(rem > 0)        
            inchanidx = data.dimord.index("channel")
            
            # Perform dry-run w/first channel-block of first trial to identify 
            # changes in output shape w.r.t. full-trial output (`chunkShape`)
            idx[inchanidx] = slice(0, n_blocks[0])  # FIXME: will be obsolte w/FauxTrial
            # shp = list(trial.shape) # FIXME: FauxTrial preps
            # idx = list(trial.idx)
            # shp[chanidx] = n_blocks[0]
            # idx[chanidx] = slice(0, n_blocks[0])
            # trial.shape = tuple(shp)
            # trial.idx = tuple(idx)
            # res, _ = self.computeFunction(trial, *self.argv, **dryRunKwargs)
            res, _ = self.computeFunction(trial[tuple(idx)], *self.argv, **dryRunKwargs)
            outchan = [dim for dim in res if dim not in chunkShape0]
            if len(outchan) != 1:
                lgl = "exactly one output dimension to scale w/channel count"
                act = "{0:d} dimensions affected by varying channel count".format(len(outchan))
                raise SPYValueError(legal=lgl, varname="chan_per_worker", actual=act)
            outchanidx = res.index(outchan[0])            
            
            # Get output chunks and grid indices for first trial
            chanstack = 0
            blockstack = 0
            for block in n_blocks:
                idx[inchanidx] = slice(blockstack, blockstack + block)
                res, _ = self.computeFunction(trial[tuple(idx)], 
                                                *self.argv, **dryRunKwargs)
                # shp = list(trial.shape) # FIXME: FauxTrial preps
                # idx = list(trial.idx)
                # shp[inchanidx] = block
                # idx[inchanidx] = slice(blockstack, blockstack + block)
                # trial.shape = tuple(shp)
                # trial.idx = tuple(idx)
                # res, _ = self.computeFunction(trial, *self.argv, **dryRunKwargs) # FauxTrial
                refidx = list(idx)
                refidx[sid] = trlslice
                lyt[outchanidx] = slice(chanstack, chanstack + res[outchanidx])
                # geo[outchanidx] = block
                targetLayout.append(tuple(lyt))
                targetShapes.append(tuple([slc.stop - slc.start for slc in lyt]))
                sourceLayout.append(tuple(refidx))
                # sourceLayout.append(trial.idx) # FIXME: FauxTrial preps
                chanstack += res[outchanidx]
                blockstack += block

        # Simple: consume all channels simultaneously, i.e., just take the entire trial
        else:
            targetLayout.append(tuple(lyt))
            targetShapes.append(chunkShape0)
            # sourceLayout.append(trial.idx)  # FIXME: FauxTrial preps
            # FIXME: will be obsolte w/FauxTrial
            # >>> START
            refidx = list(idx)
            refidx[sid] = trlslice
            sourceLayout.append(tuple(refidx))
            # >>> STOP
            
        # Construct dimensional layout of output
        stacking = targetLayout[0][0].stop
        for tk in range(1, len(data.trials)):
            trial = data.trials[tk]
            trlslice = slice(int(data.sampleinfo[tk, 0]), int(data.sampleinfo[tk, 1])) # FIXME: will be obsolte w/FauxTrial
            # trial = data._preview_trial(tk)   # FIXME: FauxTrial preps
            chkshp, _ = self.computeFunction(trial, *self.argv, **dryRunKwargs)
            lyt = [slice(0, stop) for stop in chkshp]
            lyt[0] = slice(stacking, stacking + chkshp[0])
            stacking += chkshp[0]
            if chan_per_worker is None:
                targetLayout.append(tuple(lyt))
                targetShapes.append(tuple([slc.stop - slc.start for slc in lyt]))
                # sourceLayout.append(trial.idx)  # FIXME: FauxTrial preps
                # FIXME: will be obsolte w/FauxTrial
                # >>> START
                refidx = list(idx)
                refidx[sid] = trlslice
                sourceLayout.append(tuple(refidx))
                # >>> STOP
            else:
                chanstack = 0
                blockstack = 0
                for block in n_blocks:
                    idx[inchanidx] = slice(blockstack, blockstack + block)
                    res, _ = self.computeFunction(trial[tuple(idx)], 
                                                  *self.argv, **dryRunKwargs)
                    # shp = list(trial.shape) # FIXME: FauxTrial preps
                    # idx = list(trial.idx)
                    # shp[inchanidx] = block
                    # idx[inchanidx] = slice(blockstack, blockstack + block)
                    # trial.shape = tuple(shp)
                    # trial.idx = tuple(idx)
                    # res, _ = self.computeFunction(trial, *self.argv, **dryRunKwargs) # FauxTrial
                    refidx = list(idx)
                    refidx[sid] = trlslice
                    lyt[outchanidx] = slice(chanstack, chanstack + res[outchanidx])
                    # geo[outchanidx] = block
                    targetLayout.append(tuple(lyt))
                    targetShapes.append(tuple([slc.stop - slc.start for slc in lyt]))
                    sourceLayout.append(tuple(refidx))
                    # sourceLayout.append(trial.idx) # FIXME: FauxTrial preps
                    chanstack += res[outchanidx]
                    blockstack += block
        
        # Store determined shapes and grid layout
        self.sourceLayout = sourceLayout
        self.targetLayout = targetLayout
        self.targetShapes = targetShapes
        
        # Compute max. memory footprint of chunks
        if chan_per_worker is None:
            self.chunkMem = np.prod(self.cfg["chunkShape"]) * self.dtype.itemsize
        else:
            self.chunkMem = max([np.prod(shp) for shp in self.targetShapes]) * self.dtype.itemsize
        
        # Get data access mode (only relevant for parallel reading access)
        self.datamode = data.mode
        
        # Save timeout interval setting
        self.timeout = timeout

    def compute(self, data, out, parallel=False, parallel_store=None,
                method=None, mem_thresh=0.5, log_dict=None):
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
        mem_thresh : float
           Fraction of available memory required to perform computation. By
           default, the largest single trial result must not occupy more than
           50% (``mem_thresh = 0.5``) of available single-machine or worker
           memory (if `parallel` is `False` or `True`, respectively).
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
            
        # Do not spill trials on disk if they're supposed to be removed anyway
        if parallel_store and not self.keeptrials:
            print("Syncopy core - compute: WARNING >> trial-averaging only " +\
                  "supports sequential writing! <<")
            parallel_store = False

        # Concurrent processing requires some additional prep-work...
        if parallel:

            # First and foremost, make sure a dask client is accessible
            try:
                client = dd.get_client()
            except ValueError as exc:
                msg = "parallel computing client: {}"
                raise SPYIOError(msg.format(exc.args[0]))

            # Check if trials actually fit into memory before we start computation
            wrk_size = max(wrkr.memory_limit for wrkr in client.cluster.workers.values())
            if self.chunkMem >= mem_thresh * wrk_size:
                self.chunkMem /= 1024**3
                wrk_size /= 1024**3
                msg = "Single-trial result sizes ({0:2.2f} GB) larger than available " +\
                      "worker memory ({1:2.2f} GB) currently not supported"
                raise NotImplementedError(msg.format(self.chunkMem, wrk_size))

            # In some cases distributed dask workers suffer from spontaneous
            # dementia and forget the `sys.path` of their parent process. Fun!
            def init_syncopy(dask_worker):
                spy_path = os.path.abspath(os.path.split(__path__[0])[0])
                if spy_path not in sys.path:
                    sys.path.insert(0, spy_path)
            client = dd.get_client()
            client.register_worker_callbacks(init_syncopy)

        # For sequential processing, just ensure enough memory is available
        else:
            mem_size = psutil.virtual_memory().available
            if self.chunkMem >= mem_thresh * mem_size:
                self.chunkMem /= 1024**3
                mem_size /= 1024**3
                msg = "Single-trial result sizes ({0:2.2f} GB) larger than available " +\
                      "memory ({1:2.2f} GB) currently not supported"
                raise NotImplementedError(msg.format(self.chunkMem, mem_size))

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
        computeMethod(data, out)

        # Attach computed results to output object
        out.data = h5py.File(out.filename, mode="r+")[self.dsetname]

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
        # directory for by-chunk HDF5 containers and construct virutal HDF layout
        if parallel_store:
            vdsdir = os.path.splitext(os.path.basename(out.filename))[0]
            self.vdsdir = os.path.join(__storage__, vdsdir)
            os.mkdir(self.vdsdir)
            
            layout = h5py.VirtualLayout(shape=self.outputShape, dtype=self.dtype)
            for k, idx in enumerate(self.targetLayout):
                fname = os.path.join(self.vdsdir, "{0:d}.h5".format(k))
                layout[idx] = h5py.VirtualSource(fname, "chk", shape=self.targetShapes[k])
            self.vdslayout = layout

        # Create regular HDF5 dataset for sequential writing
        else:
            
            # The shape of the target depends on trial-averaging
            if not self.keeptrials:
                shp = self.cfg["chunkShape"]
            else:
                shp = self.outputShape
            with h5py.File(out.filename, mode="w") as h5f:
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
        Nothing : None

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

        # Ensure `data` is openend read-only to permit concurrent reading access
        data.mode = "r"
        
        # Depending on chosen processing paradigm, get settings or assign defaults
        hdr = None
        if hasattr(data, "hdr"):
            hdr = data.hdr

        # Prepare to write chunks concurrently
        if self.vdsdir is not None:
            outfilename = os.path.join(self.vdsdir, "{0:d}.h5")
            outdsetname = "chk"
            lock = None
            waitcount = None

        # Write chunks sequentially            
        else:
            outfilename = out.filename
            outdsetname = self.dsetname
            lock = dd.lock.Lock(name='writer_lock')
            waitcount = int(np.round(self.timeout/self.sleeptime))
            
        # Construct a dask bag with all necessary components for parallelization
        bag = db.from_sequence([{"hdr": hdr,
                                 "keeptrials": self.keeptrials, 
                                 "infile": data.filename,
                                 "indset": data.data.name,
                                 "ingrid": self.sourceLayout[chk],
                                 "vdsdir": self.vdsdir,
                                 "outfile": outfilename.format(chk),
                                 "outdset": outdsetname,
                                 "outgrid": self.targetLayout[chk],
                                 "lock": lock,
                                 "sleeptime": self.sleeptime,
                                 "waitcount": waitcount} for chk in range(len(self.sourceLayout))]) 
        
        # Map all components (channel-trial-blocks) onto `computeFunction`
        results = bag.map(self.computeFunction, *self.argv, **self.cfg)
        
        # Make sure that all futures are executed (i.e., data is actually written)
        futures = dd.client.futures_of(results.persist())
        while any(f.status == 'pending' for f in futures):
            time.sleep(self.sleeptime)
            
        # When writing concurrently, now's the time to finally create the virtual dataset
        if self.vdsdir is not None:
            with h5py.File(out.filename, mode="w") as h5f:
                h5f.create_virtual_dataset(self.dsetname, self.vdslayout)
                
        # If trials-averagin was requested, normalize computed sum to get mean
        if not self.keeptrials:
            with h5py.File(out.filename, mode="r+") as h5f:
                h5f[self.dsetname][()] /= len(data.trials)
                h5f.flush()
        
        # Reset data access mode
        data.mode = self.datamode
        
        return 
        
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
        with h5py.File(out.filename, "r+") as h5f:
            dset = h5f[self.dsetname]
            cnt = 0
            idx = [slice(None)] * len(dset.shape)
            if self.keeptrials:
                for trl in tqdm(data.trials, desc="Computing..."):
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

    def save_distributed(self, da_arr, out, parallel_store=True, timeout=300):
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
        timeout : int
           Number of seconds to wait for saving-mutex to be acquired (only relevant in case
           of sequential writing of results, i.e., ``parallel_store = False``)

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
                time.sleep(self.sleeptime)

            # Construct virtual layout from created HDF5 files
            layout = h5py.VirtualLayout(shape=da_arr.shape, dtype=da_arr.dtype)
            for k in range(len(futures)):
                fname = os.path.join(self.vdsdir, "{0:d}.h5".format(k))
                with h5py.File(fname, "r") as h5f:
                    idx = tuple([slice(*dim) for dim in h5f["idx"]])
                    shp = h5f["chk"].shape
                layout[idx] = h5py.VirtualSource(fname, "chk", shape=shp)

            # Use generated layout to create virtual dataset
            with h5py.File(out.filename, mode="w") as h5f:
                h5f.create_virtual_dataset(self.dsetname, layout)

        # ...or use a mutex to write to a single container sequentially
        else:

            # Initialize distributed lock
            lck = dd.lock.Lock(name='writer_lock')

            # Compute timeout counter given sleep timer and wait time
            waitcount = int(np.round(timeout/self.sleeptime))

            # Map `da_arr` chunk by chunk onto ``_write_sequential``
            writers = da_arr.map_blocks(self._write_sequential, nchk, out.filename,
                                        self.dsetname, lck, self.sleeptime, waitcount,
                                        dtype="int", chunks=(1, 1))
            
            # Make sure that all futures are actually executed (i.e., data is written
            # to the container)
            futures = dd.client.futures_of(writers.persist())
            while any(f.status == 'pending' for f in futures):
                time.sleep(self.sleeptime)

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
    def _write_sequential(chk, nchk, h5name, dsname, lck, sleeptime, waitcount, 
                          block_info=None):
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
        sleeptime : float
           Split-second waiting time before a mutex is reattempted to be acquired
           (derived from class attribute `self.sleeptime`)
        waitcount : int
           Maximal attempts to acquire/release mutex before a TimeoutError is 
           raised. 
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
        counter = 0
        lck.acquire()
        while not lck.locked() and counter < waitcount:
            time.sleep(sleeptime)
            counter += 1
            
        # Open container and write current chunk
        with h5py.File(h5name, "r+") as h5f:
            h5f[dsname][idx] = chk

        # Release distributed lock
        counter = 0
        lck.release()
        while lck.locked() and counter < waitcount:
            time.sleep(sleeptime)
            counter += 1

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
