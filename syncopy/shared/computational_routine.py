# -*- coding: utf-8 -*-
#
# Base class for all computational classes in Syncopy
#

# Builtin/3rd party package imports
import os
import sys
import psutil
import h5py
import numpy as np
from itertools import chain
from abc import ABC, abstractmethod
from copy import copy
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)

# Local imports
from .tools import get_defaults
from syncopy import __storage__, __acme__, __path__
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYParallelError, SPYWarning
if __acme__:
    from acme import ParallelMap
    import dask.distributed as dd
    import dask_jobqueue as dj
    # # In case of problems w/worker-stealing, uncomment the following lines
    # import dask
    # dask.config.set(distributed__scheduler__work_stealing=False)

__all__ = []


class ComputationalRoutine(ABC):
    """Abstract class for encapsulating sequential/parallel algorithms

    A Syncopy compute class consists of a
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
    * Provides class method :func:`process_metadata`

    For details on writing compute classes and metafunctions for Syncopy, please
    refer to :doc:`/developer/compute_kernels`.
    """

    # Placeholder: the actual workhorse
    @staticmethod
    def computeFunction(arr, *argv, chunkShape=None, noCompute=None, **kwargs):
        """Computational core routine

        Parameters
        ----------
        arr : :class:`numpy.ndarray`
           Numerical data from a single trial
        *argv : tuple
           Arbitrary tuple of positional arguments
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
        ComputationalRoutine : Developer documentation: :doc:`/developer/compute_kernels`.
        """
        return None

    def __init__(self, *argv, **kwargs):
        """
        Instantiate a :class:`ComputationalRoutine` subclass

        Parameters
        ----------
        *argv : tuple
           Tuple of positional arguments passed on to :meth:`computeFunction`
        **kwargs : dict
           Keyword arguments passed on to :meth:`computeFunction`

        Returns
        -------
        obj : instance of :class:`ComputationalRoutine`-subclass
           Usable class instance for processing Syncopy data objects.
        """

        # list of positional arguments to `computeFunction` for all workers, format:
        # ``self.argv = [3, [0, 1, 1], ('a', 'b', 'c')]``
        self.argv = list(argv)

        # dict of default keyword values accepted by `computeFunction`
        self.defaultCfg = get_defaults(self.computeFunction)

        # dict of actual keyword argument values to `computeFunction` provided by user
        self.cfg = copy(self.defaultCfg)
        for key in set(self.cfg.keys()).intersection(kwargs.keys()):
            self.cfg[key] = kwargs[key]

        # binary flag: if `True`, average across trials, do nothing otherwise
        self.keeptrials = None

        # full shape of final output dataset (all trials, all chunks, etc.)
        self.outputShape = None

        # numerical type of output dataset
        self.dtype = None

        # list of dicts encoding header info of raw binary input files (experimental!)
        self.hdr = None

        # list of trial numbers to process (either `data.trials` or `data._selection.trials`)
        self.trialList = None

        # number of trials to process (shortcut for `len(self.trialList)`)
        self.numTrials = None

        # number of channel blocks to process per Trial (1 by default, only > 1 if
        # `chan_per_worker` is not `None`)
        self.numBlocksPerTrial = None

        # actual number of parallel calls to `computeFunction` to perform
        # (if `chan_per_worker` is `None`, then `numCalls` = `numTrials`, otherwise
        # `numCalls` = `numBlocksPerTrial * numTrials`)
        self.numCalls = None

        # list of index-tuples for extracting trial-chunks from input HDF5 dataset
        # >>> MUST be ordered, no repetitions! <<<
        # indices are ABSOLUTE, i.e., wrt entire dataset, not just current trial!
        self.sourceLayout = None

        # list of index-tuples for re-ordering NumPy arrays extracted w/`self.sourceLayout`
        # >>> can be unordered w/repetitions <<<
        # indices are RELATIVE, i.e., wrt current trial!
        self.sourceSelectors = None

        # list of index-tuples for storing trial-chunk result in output dataset
        # >>> MUST be ordered, no repetitions! <<<
        # indices are ABSOLUTE, i.e., wrt entire dataset, not just current trial
        self.targetLayout = None

        # list of shape-tuples of trial-chunk results
        self.targetShapes = None

        # binary flag: if `True`, use fancy array indexing via `np.ix_` to extract
        # data from input via `self.sourceLayout` + `self.sourceSelectors`; if `False`,
        # only use `self.sourceLayout` (selections ordered, no reps)
        self.useFancyIdx = None

        # integer, max. memory footprint of largest input array piece (in bytes)
        self.chunkMem = None

        # instance of ACME's `ParallelMap` to handle actual parallel computing workload
        self.pmap = None

        # directory for storing source-HDF5 files making up virtual output dataset
        self.virtualDatasetDir = None

        # h5py layout encoding shape/geometry of file sources within virtual output dataset
        self.VirtualDatasetLayout = None

        # name of temporary datasets (only relevant for virtual output datasets)
        self.virtualDatasetNames = "chk"

        # placeholder name for (temporary) output datasets
        self.tmpDsetName = None

        # Name of output HDF5 file
        self.outFileName = None

        # name of target HDF5 dataset in output object
        self.outDatasetName = "data"

        # tmp holding var for preserving original access mode of `data`
        self.dataMode = None

        # time (in seconds) b/w querying state of futures ('pending' -> 'finished')
        self.sleepTime = 0.1

        # if `True`, enforces use of single-threaded scheduler in `compute_parallel`
        self.parallelDebug = False

        # format string for tqdm progress bars in sequential computation
        self.tqdmFormat = "{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

        # maximal acceptable size (in MB) of any provided positional argument
        self._maxArgSize = 100

        # counter and maximal recursion depth for calling `self._sizeof`
        self._callMax = 10000
        self._callCount = 0

    def initialize(self, data, out_stackingdim, chan_per_worker=None, keeptrials=True):
        """
        Perform dry-run of calculation to determine output shape

        Parameters
        ----------
        data : syncopy data object
           Syncopy data object to be processed (has to be the same object
           that is passed to :meth:`compute` for the actual calculation).
        out_stackingdim : int
           Index of data dimension for stacking trials in output object
        chan_per_worker : None or int
           Number of channels to be processed by each worker (only relevant in
           case of concurrent processing). If `chan_per_worker` is `None` (default)
           by-trial parallelism is used, i.e., each worker processes
           data corresponding to a full trial. If `chan_per_worker > 0`, trials
           are split into channel-groups of size `chan_per_worker` (+ rest if the
           number of channels is not divisible by `chan_per_worker` without
           remainder) and workers are assigned by-trial channel-groups for
           processing.
        keeptrials : bool
            Flag indicating whether to return individual trials or average

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

        # First store `keeptrial` keyword value (important for output shapes below)
        self.keeptrials = keeptrials

        # Determine if data-selection was provided; if so, extract trials and check
        # whether selection requires fancy array indexing
        if data._selection is not None:
            self.trialList = data._selection.trials
            self.useFancyIdx = data._selection._useFancy
        else:
            self.trialList = list(range(len(data.trials)))
            self.useFancyIdx = False
        self.numTrials = len(self.trialList)

        # Prepare dryrun arguments and determine geometry of trials in output
        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        chk_list = []
        dtp_list = []
        trials = []
        for tk, trialno in enumerate(self.trialList):
            trial = data._preview_trial(trialno)
            trlArg = tuple(arg[tk] if isinstance(arg, (list, tuple, np.ndarray)) and len(arg) == self.numTrials \
                else arg for arg in self.argv)
            chunkShape, dtype = self.computeFunction(trial,
                                                     *trlArg,
                                                     **dryRunKwargs)
            chk_list.append(list(chunkShape))
            dtp_list.append(dtype)
            trials.append(trial)

        # Determine trial stacking dimension and compute aggregate shape of output
        stackingDim = out_stackingdim
        totalSize = sum(cShape[stackingDim] for cShape in chk_list)
        outputShape = list(chunkShape)
        if stackingDim < 0 or stackingDim >= len(outputShape):
            msg = "valid trial stacking dimension"
            raise SPYTypeError(out_stackingdim, varname="out_stackingdim", expected=msg)
        outputShape[stackingDim] = totalSize

        # The aggregate shape is computed as max across all chunks
        chk_arr = np.array(chk_list)
        chunkShape = tuple(chk_arr.max(axis=0))
        if np.unique(chk_arr[:, stackingDim]).size > 1 and not self.keeptrials:
            err = "Averaging trials of unequal lengths in output currently not supported!"
            raise NotImplementedError(err)
        if np.any([dtp_list[0] != dtp for dtp in dtp_list]):
            lgl = "unique output dtype"
            act = "{} different output dtypes".format(np.unique(dtp_list).size)
            raise SPYValueError(legal=lgl, varname="dtype", actual=act)

        # Save determined shapes and data type
        self.outputShape = tuple(outputShape)
        self.cfg["chunkShape"] = chunkShape
        self.dtype = np.dtype(dtp_list[0])

        # Ensure channel parallelization can be done at all
        if chan_per_worker is not None and "channel" not in data.dimord:
            msg = "input object does not contain `channel` dimension for parallelization!"
            SPYWarning(msg)
            chan_per_worker = None
        if chan_per_worker is not None and self.keeptrials is False:
            msg = "trial-averaging does not support channel-block parallelization!"
            SPYWarning(msg)
            chan_per_worker = None
        if data._selection is not None:
            if chan_per_worker is not None and data._selection.channel != slice(None, None, 1):
                msg = "channel selection and simultaneous channel-block " +\
                    "parallelization not yet supported!"
                SPYWarning(msg)
                chan_per_worker = None

        # Allocate control variables
        trial = trials[0]
        trlArg0 = tuple(arg[0] if isinstance(arg, (list, tuple, np.ndarray)) and len(arg) == self.numTrials \
            else arg for arg in self.argv)
        chunkShape0 = chk_arr[0, :]
        lyt = [slice(0, stop) for stop in chunkShape0]
        sourceLayout = []
        targetLayout = []
        targetShapes = []
        c_blocks = [1]

        # If parallelization across channels is requested the first trial is
        # split up into several chunks that need to be processed/allocated
        if chan_per_worker is not None:

            # Set up channel-chunking: `c_blocks` holds channel blocks per trial
            nChannels = data.channel.size
            rem = int(nChannels % chan_per_worker)
            c_blocks = [chan_per_worker] * int(nChannels//chan_per_worker) + [rem] * int(rem > 0)
            inchanidx = data.dimord.index("channel")

            # Perform dry-run w/first channel-block of first trial to identify
            # changes in output shape w.r.t. full-trial output (`chunkShape`)
            shp = list(trial.shape)
            idx = list(trial.idx)
            shp[inchanidx] = c_blocks[0]
            idx[inchanidx] = slice(0, c_blocks[0])
            trial.shape = tuple(shp)
            trial.idx = tuple(idx)
            res, _ = self.computeFunction(trial, *trlArg0, **dryRunKwargs)
            outchan = [dim for dim in res if dim not in chunkShape0]
            if len(outchan) != 1:
                lgl = "exactly one output dimension to scale w/channel count"
                act = "{0:d} dimensions affected by varying channel count".format(len(outchan))
                raise SPYValueError(legal=lgl, varname="chan_per_worker", actual=act)
            outchanidx = res.index(outchan[0])

            # Get output chunks and grid indices for first trial
            chanstack = 0
            blockstack = 0
            for block in c_blocks:
                shp = list(trial.shape)
                idx = list(trial.idx)
                shp[inchanidx] = block
                idx[inchanidx] = slice(blockstack, blockstack + block)
                trial.shape = tuple(shp)
                trial.idx = tuple(idx)
                res, _ = self.computeFunction(trial, *trlArg0, **dryRunKwargs)
                lyt[outchanidx] = slice(chanstack, chanstack + res[outchanidx])
                targetLayout.append(tuple(lyt))
                targetShapes.append(tuple([slc.stop - slc.start for slc in lyt]))
                sourceLayout.append(trial.idx)
                chanstack += res[outchanidx]
                blockstack += block

        # Simple: consume all channels simultaneously, i.e., just take the entire trial
        else:
            targetLayout.append(tuple(lyt))
            targetShapes.append(chunkShape0)
            sourceLayout.append(trial.idx)

        # Construct dimensional layout of output
        stacking = targetLayout[0][stackingDim].stop
        for tk in range(1, self.numTrials):
            trial = trials[tk]
            trlArg = tuple(arg[tk] if isinstance(arg, (list, tuple, np.ndarray)) and len(arg) == self.numTrials \
                else arg for arg in self.argv)
            chkshp = chk_list[tk]
            lyt = [slice(0, stop) for stop in chkshp]
            lyt[stackingDim] = slice(stacking, stacking + chkshp[stackingDim])
            stacking += chkshp[stackingDim]
            if chan_per_worker is None:
                targetLayout.append(tuple(lyt))
                targetShapes.append(tuple([slc.stop - slc.start for slc in lyt]))
                sourceLayout.append(trial.idx)
            else:
                chanstack = 0
                blockstack = 0
                for block in c_blocks:
                    shp = list(trial.shape)
                    idx = list(trial.idx)
                    shp[inchanidx] = block
                    idx[inchanidx] = slice(blockstack, blockstack + block)
                    trial.shape = tuple(shp)
                    trial.idx = tuple(idx)
                    res, _ = self.computeFunction(trial, *trlArg, **dryRunKwargs) # FauxTrial
                    lyt[outchanidx] = slice(chanstack, chanstack + res[outchanidx])
                    targetLayout.append(tuple(lyt))
                    targetShapes.append(tuple([slc.stop - slc.start for slc in lyt]))
                    sourceLayout.append(trial.idx)
                    chanstack += res[outchanidx]
                    blockstack += block

        # Infer how many concurrent `computeFunction` calls we're about to execute
        self.numBlocksPerTrial = len(c_blocks)
        self.numCalls = self.numBlocksPerTrial * self.numTrials

        # If the determined source layout contains unordered lists and/or index
        # repetitions, set `self.useFancyIdx` to `True` and prepare a separate
        # `sourceSelectors` list that is used in addition to `sourceLayout` for
        # data extraction.
        # In this case `sourceLayout` uses ABSOLUTE indices (indices wrt to size
        # of ENTIRE DATASET) that are SORTED W/O REPS to extract a NumPy array
        # of appropriate size from HDF5.
        # Then `sourceLayout` uses RELATIVE indices (indices wrt to size of CURRENT
        # TRIAL) that can be UNSORTED W/REPS to actually perform the requested
        # selection on the NumPy array extracted w/`sourceLayout`.
        for grd in sourceLayout:
            if any([np.diff(sel).min() <= 0 if isinstance(sel, list)
                    and len(sel) > 1 else False for sel in grd]):
                self.useFancyIdx = True
                break
        if self.useFancyIdx:
            sourceSelectors = []
            for gk, grd in enumerate(sourceLayout):
                ingrid = list(grd)
                sigrid = []
                for sk, sel in enumerate(grd):
                    if isinstance(sel, list):
                        selarr = np.array(sel, dtype=np.intp)
                    else: # sel is a slice
                        step = sel.step
                        if sel.step is None:
                            step = 1
                        selarr = np.array(list(range(sel.start, sel.stop, step)), dtype=np.intp)
                    if selarr.size > 0:
                        sigrid.append(np.array(selarr) - selarr.min())
                        ingrid[sk] = slice(selarr.min(), selarr.max() + 1, 1)
                    else:
                        sigrid.append([])
                        ingrid[sk] = []
                sourceSelectors.append(tuple(sigrid))
                sourceLayout[gk] = tuple(ingrid)
        else:
            sourceSelectors = [Ellipsis] * self.numCalls

        # Store determined shapes and grid layout
        self.sourceLayout = sourceLayout
        self.sourceSelectors = sourceSelectors
        self.targetLayout = targetLayout
        self.targetShapes = targetShapes

        # Compute max. memory footprint of chunks
        if chan_per_worker is None:
            self.chunkMem = np.prod(self.cfg["chunkShape"]) * self.dtype.itemsize
        else:
            self.chunkMem = max([np.prod(shp) for shp in self.targetShapes]) * self.dtype.itemsize

        # Get data access mode (only relevant for parallel reading access)
        self.dataMode = data.mode

    def compute(self, data, out, parallel=False, parallel_store=None,
                method=None, mem_thresh=0.5, log_dict=None, parallel_debug=False):
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
           class method (starting with the word `"compute_"`).
        mem_thresh : float
           Fraction of available memory required to perform computation. By
           default, the largest single trial result must not occupy more than
           50% (``mem_thresh = 0.5``) of available single-machine or worker
           memory (if `parallel` is `False` or `True`, respectively).
        log_dict : None or dict
           If `None`, the `log` properties of `out` is populated with the employed
           keyword arguments used in :meth:`computeFunction`.
           Otherwise, `out`'s `log` properties are filled  with items taken
           from `log_dict`.
        parallel_debug : bool
           If `True`, concurrent processing is performed using a single-threaded
           scheduler, i.e., all parallel computing task are run in the current
           Python thread permitting usage of tools like `pdb`/`ipdb`, `cProfile`
           and the like in :meth:`computeFunction`.
           Note that enabling parallel debugging effectively runs the given computation
           on the calling machine locally thereby requiring sufficient memory and
           CPU capacity.

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
        3. :meth:`process_metadata` attaches all relevant meta-information to
           the result `out` after successful termination of the calculation
        4. :meth:`write_log` stores employed input arguments in `out.cfg`
           and `out.log` to reproduce all relevant computational steps that
           generated `out`.

        Parallel processing setup and input sanitization is performed by
        `ACME <https://github.com/esi-neuroscience/acme>`_. Specifically, all
        necessary information for parallel execution and storage of results is
        propagated to the :class:`~acme.ParallelMap` context manager.

        See also
        --------
        initialize : pre-calculation preparations
        preallocate_output : storage provisioning
        compute_parallel : concurrent computation using :meth:`computeFunction`
        compute_sequential : sequential computation using :meth:`computeFunction`
        process_metadata : management of meta-information
        write_log : log-entry organization
        acme.ParallelMap : concurrent execution of Python callables
        """

        # By default, use VDS storage for parallel computing
        if parallel_store is None:
            parallel_store = parallel

        # Do not spill trials on disk if they're supposed to be removed anyway
        if parallel_store and not self.keeptrials:
            msg = "trial-averaging only supports sequential writing!"
            SPYWarning(msg)
            parallel_store = False

        # Create HDF5 dataset of appropriate dimension
        self.preallocate_output(out, parallel_store=parallel_store)

        # Concurrent processing requires some additional prep-work...
        if parallel:

            # Construct list of dicts that will be passed on to workers: in the
            # parallel case, `trl_dat` is a dictionary!
            workerDicts = [{"hdr": self.hdr,
                            "keeptrials": self.keeptrials,
                            "infile": data.filename,
                            "indset": data.data.name,
                            "ingrid": self.sourceLayout[chk],
                            "sigrid": self.sourceSelectors[chk],
                            "fancy": self.useFancyIdx,
                            "vdsdir": self.virtualDatasetDir,
                            "outfile": self.outFileName.format(chk),
                            "outdset": self.tmpDsetName,
                            "outgrid": self.targetLayout[chk],
                            "outshape": self.targetShapes[chk],
                            "dtype": self.dtype} for chk in range(self.numCalls)]

            # If channel-block parallelization has been set up, positional args of
            # `computeFunction` need to be massaged: any list whose elements represent
            # trial-specific args, needs to be expanded (so that each channel-block
            # per trial receives the correct number of pos. args)
            ArgV = list(self.argv)
            if self.numBlocksPerTrial > 1:
                for ak, arg in enumerate(self.argv):
                    if isinstance(arg, (list, tuple)):
                        if len(arg) == self.numTrials:
                            unrolled = chain.from_iterable([[ag] * self.numBlocksPerTrial for ag in arg])
                            if isinstance(arg, list):
                                ArgV[ak] = list(unrolled)
                            else:
                                ArgV[ak] = tuple(unrolled)
                    elif isinstance(arg, np.ndarray):
                        if len(arg.squeeze().shape) == 1 and arg.squeeze().size == self.numTrials:
                            ArgV[ak] = np.array(chain.from_iterable([[ag] * self.numBlocksPerTrial for ag in arg]))

            # Positional args for computeFunctions consist of `trl_dat` + others
            # (stored in `ArgV`). Account for this when seeting up `ParallelMap`
            if len(ArgV) == 0:
                inargs = (workerDicts, )
            else:
                inargs = (workerDicts, *ArgV)

            # Let ACME take care of argument distribution and memory checks: note
            # that `cfg` is trial-independent, i.e., we can simply throw it in here!
            self.pmap = ParallelMap(self.computeFunction,
                                    *inargs,
                                    n_inputs=self.numCalls,
                                    write_worker_results=False,
                                    write_pickle=False,
                                    partition="auto",
                                    n_jobs="auto",
                                    mem_per_job="auto",
                                    setup_timeout=60,
                                    setup_interactive=False,
                                    stop_client="auto",
                                    verbose=None,
                                    logfile=None,
                                    **self.cfg)

            # Edge-case correction: if by chance, any array-like element `x` of `cfg`
            # satisfies `len(x) = numCalls`, `ParallelMap` attempts to tear open `x` and
            # distribute its elements across workers. Prevent this!
            if self.numCalls > 1:
                for key, value in self.pmap.kwargv.items():
                    if isinstance(value, (list, tuple)):
                        if len(value) == self.numCalls:
                            self.pmap.kwargv[key] = [value]
                    elif isinstance(value, np.ndarray):
                        if len(value.squeeze().shape) == 1 and value.squeeze().size == self.numCalls:
                            self.pmap.kwargv[key] = [value]

            # Check if trials actually fit into memory before we start computation
            client = self.pmap.daemon.client
            if isinstance(client.cluster, (dd.LocalCluster, dj.SLURMCluster)):
                workerMem = [w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()]
                if len(workerMem) == 0:
                    raise SPYParallelError("no online workers found", client=client)
                workerMemMax = max(workerMem)
                if self.chunkMem >= mem_thresh * workerMemMax:
                    self.chunkMem /= 1024**3
                    workerMemMax /= 1000**3
                    msg = "Single-trial result sizes ({0:2.2f} GB) larger than available " +\
                        "worker memory ({1:2.2f} GB) currently not supported"
                    raise NotImplementedError(msg.format(self.chunkMem, workerMemMax))
            else:
                msg = "`ComputationalRoutine` only supports `LocalCluster` and " +\
                    "`SLURMCluster` dask cluster objects. Proceed with caution. "
                SPYWarning(msg)

            # Store provided debugging state
            self.parallelDebug = parallel_debug

        # For sequential processing, just ensure enough memory is available
        else:
            memSize = psutil.virtual_memory().available
            if self.chunkMem >= mem_thresh * memSize:
                self.chunkMem /= 1024**3
                memSize /= 1024**3
                msg = "Single-trial result sizes ({0:2.2f} GB) larger than available " +\
                      "memory ({1:2.2f} GB) currently not supported"
                raise NotImplementedError(msg.format(self.chunkMem, memSize))

        # The `method` keyword can be used to override the `parallel` flag
        if method is None:
            if parallel:
                computeMethod = self.compute_parallel
            else:
                computeMethod = self.compute_sequential
        else:
            computeMethod = getattr(self, "compute_" + method, None)

        # Ensure `data` is openend read-only to permit (potentially concurrent)
        # reading access to backing device on disk
        data.mode = "r"

        # Take care of `VirtualData` objects
        self.hdr = getattr(data, "hdr", None)

        # Perform actual computation
        computeMethod(data, out)

        # Reset data access mode
        data.mode = self.dataMode

        # Attach computed results to output object
        out.data = h5py.File(out.filename, mode="r+")[self.outDatasetName]

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
           If `True`, a directory for virtual source files is created
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

        # In case parallel writing via VDS storage is requested, prepare
        # directory for by-chunk HDF5 files and construct virtual HDF layout
        if parallel_store:
            vdsdir = os.path.splitext(os.path.basename(out.filename))[0]
            self.virtualDatasetDir = os.path.join(__storage__, vdsdir)
            os.mkdir(self.virtualDatasetDir)

            layout = h5py.VirtualLayout(shape=self.outputShape, dtype=self.dtype)
            for k, idx in enumerate(self.targetLayout):
                fname = os.path.join(self.virtualDatasetDir, "{0:d}.h5".format(k))
                # Catch empty selections: don't map empty sources into the layout of the VDS
                if all([sel for sel in self.sourceLayout[k]]):
                    layout[idx] = h5py.VirtualSource(fname, self.virtualDatasetNames, shape=self.targetShapes[k])
            self.VirtualDatasetLayout = layout
            self.outFileName = os.path.join(self.virtualDatasetDir, "{0:d}.h5")
            self.tmpDsetName = self.virtualDatasetNames

        # Create regular HDF5 dataset for sequential writing
        else:

            # The shape of the target depends on trial-averaging
            if not self.keeptrials:
                shp = self.cfg["chunkShape"]
            else:
                shp = self.outputShape
            with h5py.File(out.filename, mode="w") as h5f:
                h5f.create_dataset(name=self.outDatasetName,
                                   dtype=self.dtype, shape=shp)
            self.outFileName = out.filename
            self.tmpDsetName = self.outDatasetName

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
        The actual reading of source data and writing of results is managed by
        the decorator :func:`syncopy.shared.parsers.unwrap_io`.

        See also
        --------
        compute : management routine invoking parallel/sequential compute kernels
        compute_sequential : serial processing counterpart of this method
        """

        # Let ACME do the heavy lifting
        with self.pmap as pm:
            pm.compute(debug=self.parallelDebug)

        # When writing concurrently, now's the time to finally create the virtual dataset
        if self.virtualDatasetDir is not None:
            with h5py.File(out.filename, mode="w") as h5f:
                h5f.create_virtual_dataset(self.outDatasetName, self.VirtualDatasetLayout)

        # If trial-averaging was requested, normalize computed sum to get mean
        if not self.keeptrials:
            with h5py.File(out.filename, mode="r+") as h5f:
                h5f[self.outDatasetName][()] /= self.numTrials
                h5f.flush()

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
        pre-allocated by :meth:`preallocate_output`). Since the calculation result
        is immediately stored on disk, propagation of arrays across routines
        is avoided and memory usage is kept to a minimum.

        See also
        --------
        compute : management routine invoking parallel/sequential compute kernels
        compute_parallel : concurrent processing counterpart of this method
        """

        # Initialize on-disk backing device (either HDF5 file or memmap)
        if self.hdr is None:
            try:
                sourceObj = h5py.File(data.filename, mode="r")[data.data.name]
                isHDF = True
            except OSError:
                sourceObj = open_memmap(data.filename, mode="c")
                isHDF = False
            except Exception as exc:
                raise exc

        # Iterate over (selected) trials and write directly to target HDF5 dataset
        with h5py.File(out.filename, "r+") as h5fout:
            target = h5fout[self.outDatasetName]

            for nblock in tqdm(range(self.numTrials), bar_format=self.tqdmFormat):

                # Extract respective indexing tuples from constructed lists
                ingrid = self.sourceLayout[nblock]
                sigrid = self.sourceSelectors[nblock]
                outgrid = self.targetLayout[nblock]
                argv = tuple(arg[nblock] \
                    if isinstance(arg, (list, tuple, np.ndarray)) and len(arg) == self.numTrials \
                        else arg for arg in self.argv)

                # Catch empty source-array selections; this workaround is not
                # necessary for h5py version 2.10+ (see https://github.com/h5py/h5py/pull/1174)
                if any([not sel for sel in ingrid]):
                    res = np.empty(self.targetShapes[nblock], dtype=self.dtype)
                else:
                    # Get source data as NumPy array
                    if self.hdr is None:
                        if isHDF:
                            if self.useFancyIdx:
                                arr = np.array(sourceObj[tuple(ingrid)])[np.ix_(*sigrid)]
                            else:
                                arr = np.array(sourceObj[tuple(ingrid)])
                        else:
                            if self.useFancyIdx:
                                arr = sourceObj[np.ix_(*ingrid)]
                            else:
                                arr = np.array(sourceObj[ingrid])
                        sourceObj.flush()
                    else:
                        idx = ingrid
                        if self.useFancyIdx:
                            idx = np.ix_(*ingrid)
                        stacks = []
                        for fk, fname in enumerate(data.filename):
                            stacks.append(np.memmap(fname, offset=int(self.hdr[fk]["length"]),
                                                    mode="r", dtype=self.hdr[fk]["dtype"],
                                                    shape=(self.hdr[fk]["M"], self.hdr[fk]["N"]))[idx])
                        arr = np.vstack(stacks)[ingrid]

                    # Perform computation
                    res = self.computeFunction(arr, *argv, **self.cfg)

                # Either write result to `outgrid` location in `target` or add it up
                if self.keeptrials:
                    target[outgrid] = res
                else:
                    target[()] = np.nansum([target, res], axis=0)

                # Flush every iteration to avoid memory leakage
                h5fout.flush()

            # If trial-averaging was requested, normalize computed sum to get mean
            if not self.keeptrials:
                target[()] /= self.numTrials

        # If source was HDF5 file, close it to prevent access errors
        if isHDF:
            sourceObj.file.close()

        return

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
           If `None`, the `log` properties of `out` is populated with the employed
           keyword arguments used in :meth:`computeFunction`.
           Otherwise, `out`'s `log` properties are filled  with items taken
           from `log_dict`.

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

        # Prepare keywords used by `computeFunction` (sans implementation-specific stuff)
        cfg = dict(self.cfg)
        for key in ["noCompute", "chunkShape"]:
            cfg.pop(key)

        # Write log and store `cfg` constructed above in corresponding prop of `out`
        if log_dict is None:
            log_dict = cfg
        logOpts = ""
        for k, v in log_dict.items():
            logOpts += "\t{key:s} = {value:s}\n".format(key=k,
                                                        value=str(v) if len(str(v)) < 80
                                                        else str(v)[:30] + ", ..., " + str(v)[-30:])
        out.log = logHead + logOpts
        out.cfg = cfg

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
        Consult the developer documentation (:doc:`/developer/compute_kernels`) for
        further details.

        See also
        --------
        write_log : Logging of calculation parameters
        """
        pass
