# -*- coding: utf-8 -*-
#
# ALREADY KNOW YOU THAT WHICH YOU NEED
#
# Created: 2019-05-13 09:18:55
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-03 17:45:26>

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
from syncopy import __storage__, __dask__
if __dask__:
    import dask
    import dask.distributed as dd
    import dask.array as da
    import dask.bag as db

__all__ = []


class ComputationalRoutine(ABC):
    """Abstract class for encapsulating sequential/parallel algorithms

    This class provides a blueprint for implementing algorithmic strategies
    in Syncopy. Every computational method in Syncopy consists of a core
    routine, the :func:`computeFunction`, which can be executed either
    sequentially or fully parallel. To unify common instruction sequences
    and minimize code redundancy, Syncopy's :class:`ComputationalRoutine`
    manages all pre- and post-processing steps necessary during preparation
    and after termination of a calculation. This permits developers to
    focus exclusively on the implementation of the actual algorithmic
    details when including a new computational method in Syncopy.

    Designing a :func:`computeFunction`
    -----------------------------------
    For enabling :class:`ComputationalRoutine` to perform all required
    computational management tasks, a :func:`computeFunction` has to
    satisfy a few basic requirements. Syncopy leverages a hierarchical
    parallelization paradigm whose low-level foundation is represented by
    trial-based parallelism (its open-ended higher levels may constitute
    by-object, by-experiment or by-session parallelization). Thus, with
    :func:`computeFunction` representing the computational core of an
    (arbitrarily complex) superseding algorithm, it has to be structured to
    support trial-based parallel computing.  Specifically, this means the
    scope of work of a :func:`computeFunction` is **a single trial**. Note
    that this also implies that any parallelism integrated in
    :func:`computeFunction` has to be designed with higher-level parallel
    execution in mind (e.g., concurrent processing of sessions on top of
    trials).

    Technically, a :func:`computeFunction` is a regular stand-alone Python
    function (**not** a class method) that accepts a :class:`numpy.ndarray`
    as its first positional argument and supports (at least) the two
    keyword arguments `chunkShape` and `noCompute`. The
    :class:`numpy.ndarray` represents aggregate data from one trial (only
    data, no meta-information). Any required meta-info (such as channel
    labels, trial definition records etc.) has to be passed to
    :func:`computeFunction` either as additional (2nd an onward) positional
    or named keyword arguments (`chunkShape` and `noCompute` are the only
    reserved keywords).

    The return values of :func:`computeFunction` are controlled by the
    `noCompute` keyword.  In general, :func:`computeFunction` returns
    exactly one :class:`numpy.ndarray` representing the result of
    processing data from a single trial. The `noCompute` keyword is used to
    perform a 'dry-run' of the processing operations to propagate the
    expected numerical type and memory footprint of the result to
    :class:`ComputationalRoutine` without actually performing any
    calculations. To optimize performance, :class:`ComputationalRoutine`
    uses the information gathered in the dry-runs for each trial to
    allocate identically-sized array-blocks accommodating the largest (by
    shape) result-array across all trials.  In this manner a global
    block-size is identified, which can subsequently be accessed inside
    :func:`computeFunction` via the `chunkShape` keyword during the actual
    computation.

    Summarized, a valid :func:`computeFunction`, `cfunc`, meets the 
    following basic requirements:

    * **Call signature** 

      >>> def cfunc(arr, arg1, arg2, ..., argN, chunkShape=None, noCompute=None, **kwargs)

      where `arr` is a :class:`numpy.ndarray` representing trial data,
      `arg1`, ..., `argN` are arbitrary positional arguments and
      `chunkShape` (a tuple if not `None`) as well as `noCompute` (bool if
      not `None`) are reserved keywords.

    * **Return values**

      During the dry-run phase, i.e., if `noCompute` is `True`, the expected
      output shape and its :class:`numpy.dtype` are returned, otherwise the
      result of the computation (a :class:`numpy.ndarray`) is returned:

      >>> def cfunc(arr, arg1, arg2, ..., argN, chunkShape=None, noCompute=None, **kwargs)
      >>> # determine expected output shape and numerical type...
      >>> if noCompute:
      >>>     return outShape, outdtype
      >>> # the actual computation is happening here...
      >>> return res

      Note that dtype and shape of `res` have to agree with `outShape` and 
      `outdtype` specified in the dry-run. 

    A simple example of a :func:`computeFunction` illustrating these concepts 
    is given in `Examples`. 

    The Algorithmic Layout of :class:`ComputationalRoutine`
    -------------------------------------------------------
    Technically, Syncopy's :class:`ComputationalRoutine` wraps an external
    :func:`computeFunction` by executing all necessary auxiliary routines
    leading up to and post termination of the actual computation (memory
    pre-allocation, generation of parallel/sequential instruction trees,
    processing and storage of results, etc.). Specifically,
    :class:`ComputationalRoutine` is an abstract base class that can
    represent any trial-concurrent computational tree. Thus, any
    arbitrarily complex algorithmic pattern satisfying this single
    criterion can be incorporated as a regular class into Syncopy with
    minimal implementation effort by simply inheriting from
    :class:`ComputationalRoutine`.

    Internally, the operational principle of a :class:`ComputationalRoutine` 
    is encapsulated in two class methods:

    1. :func:`initialize`

       The class is instantiated with (at least) the positional and keyword
       arguments of the associated :func:`computeFunction` minus the
       trial-data array (the the first positional argument of
       :func:`computeFunction`) and the reserved keywords `chunkShape` and
       `noCompute`. Further, an additional keyword is reserved at class
       instantiation time: `keeptrials` controls whether data is averaged
       across trials after calculation (``keeptrials = False``).  Thus, let
       `Algo` be a concrete subclass of :class:`ComputationalRoutine`, and
       let `cfunc`, defined akin to above

       >>> def cfunc(arr, arg1, arg2, argN, chunkShape=None, noCompute=None, kwarg1="this", kwarg2=False)
       
       be its corresponding :func:`computeFunction`. Then a valid 
       instantiation of `Algo` may look as follows:

       >>> algorithm = Algo(arg1, arg1, arg2, argN, kwarg1="this", kwarg2=False)

       Before the `algorithm` instance of `Algo` can be used, a dry-run of 
       the actual computation has to be performed to determine the expected 
       dimensionality and numerical type of the result,

       >>> algorithm.initialize(data)

       where `data` is a Syncopy data object representing the input quantity
       to be processed by `algorithm`. 

    2. :func:`compute`

       This management class method constitutes the functional core of
       :class:`ComputationalRoutine`.  It handles memory pre-allocation,
       storage provisioning, the actual computation and processing of
       meta-information. Theses tasks are encapsulated in distinct class
       methods which are designed to perform the respective operations
       independently from the concrete computational procedure.  Thus, most
       of these methods do not require any problem-specific adaptions and
       act as stand-alone administration routines. The only exception to
       this design-concept is :func:`process_metadata`, which is intended
       to attach meta-information to the final output object. Since
       modifications of meta-data are highly dependent on the nature of the
       performed calculation, :func:`process_metadata` is the only abstract
       method of :class:`ComputationalRoutine` that needs to be supplied in
       addition to :func:`computeFunction`.

       Several keywords control the workflow in :class:`ComputationalRoutine`:  

       * Depending on the `parallel` keyword, processing is done either
         sequentially trial by trial (``parallel = False``) or concurrently
         across all trials (if `parallel` is `True`). The two scenarios are
         handled by separate class methods, :func:`compute_sequential` and
         :func:`compute_parallel`, respectively, that use independent
         operational frameworks for processing. However, both
         :func:`compute_sequential` and :func:`compute_parallel` call an
         external :func:`computeFunction` to perform the actual
         calculation.

       * The `parallel_store` keyword controls the employed storage
         mechanism: if `True`, the result of the computation is stored in a
         fully concurrent manner where each worker saves its locally held
         data segment on disk leveraging the distributed access
         capabilities of virtual HDF5 datasets. If ``parallel_store =
         False``, a mutex is used to lock a single HDF5 file for sequential
         writing.

       * The `method` keyword can be used to override the default selection
         of the processing function (:func:`compute_parallel` if `parallel`
         is `True` or :func:`compute_sequential` otherwise). Refer to the
         docstrings of :func:`compute_parallel` or
         :func:`compute_sequential` for details on the required structure
         of a concurrent or serial processing function.

       * The keyword `log_dict` can be used to provide a dictionary of
         keyword-value pairs that are passed on to :func:`process_metadata`
         to be attached to the final output object.

       Going back to the exemplary `algorithm` instance of `Algo` discussed
       above, after initialization, the actual computation is kicked off
       with a single call of :func:`compute` with keywords pursuant to the
       intended computational workflow. For instance,

       >>> algorithm.compute(data, out, parallel=True)

       launches the parallel processing of `data` using the computational 
       scheme implemented in `cfunc` and stores the result in the Syncopy 
       object `out`. 
       
    To further clarify these concepts, `Examples` illustrates how to
    encapsulate a simple algorithmic scheme in a subclass of
    :class:`ComputationalRoutine` that calls a custom :func:`computeFunction`.

    Examples
    --------
    Coming soon.. 

    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FIXME: add slots
    """

    # Placeholder: the actual workhorse
    @staticmethod
    def computeFunction():
        return None

    # Placeholder: manager that calls `computeFunction` (sets up `dask` etc. )
    def computeMethod(self):
        return None

    def __init__(self, *argv, **kwargs):
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
        Coming soon...
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

        # By default, use VDS storage for parallel computing
        if parallel_store is None:
            parallel_store = parallel

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

        return result

    def compute_sequential(self, data, out):

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
                    # dset[tk, ...] = self.computeFunction(trl,
                    #                                      *self.argv,
                    #                                      **self.cfg)
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
    def process_metadata(self, *args):
        pass
