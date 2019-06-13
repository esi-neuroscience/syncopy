# -*- coding: utf-8 -*-
#
# ALREADY KNOW YOU THAT WHICH YOU NEED
#
# Created: 2019-05-13 09:18:55
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-06-13 15:17:21>

# Builtin/3rd party package imports
import os
import time
import h5py
import numpy as np
from abc import ABC, abstractmethod
from copy import copy

# Local imports
from .parsers import get_defaults
from syncopy import __storage__, __dask__
if __dask__:
    import dask.distributed as dd

__all__ = []


class ComputationalRoutine(ABC):

    # Placeholder: the actual workhorse
    @staticmethod
    def computeFunction():
        return None

    # Placeholder: manager that calls ``computeFunction`` (sets up `dask` etc. )
    def computeMethod(self):
        return None

    def __init__(self, *argv, **kwargs):
        self.defaultCfg = get_defaults(self.computeFunction)
        self.cfg = copy(self.defaultCfg)
        self.cfg.update(**kwargs)
        self.argv = argv
        self.outputShape = None
        self.dtype = None
        self.vdsdir = None

    def initialize(self, data):

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
            chunkShape = tuple(chk_arr.max(axis=0))
            self.outputShape = (chk_arr[:, 0].sum(),) + chunkShape[1:]
        else:
            self.outputShape = (len(data.trials),) + chunkShape[1:]

        # Assign computed chunkshape to cfg dict
        self.cfg["chunkShape"] = chunkShape

    def preallocate_output(self, out, parallel_store=False):

        # In case parallel writing via VDS storage is requested, prepare
        # directory for by-chunk HDF5 containers
        if parallel_store:
            vdsdir = os.path.splitext(os.path.basename(out._filename))[0]
            self.vdsdir = os.path.join(__storage__, vdsdir)
            os.mkdir(self.vdsdir)

        # Create regular HDF5 dataset for sequential writing
        else:
            with h5py.File(out._filename, mode="w") as h5f:
                h5f.create_dataset(name=out.__class__.__name__,
                                   dtype=self.dtype, shape=self.outputShape)

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

        # If computing is done in parallel, save distributed arraty
        if parallel:
            self.save_distributed(result, out, parallel_store)

        # Attach computed results to output object
        out.data = h5py.File(out._filename, mode="r+")[out.__class__.__name__]

        # Store meta-data, write log and get outta here
        self.handle_metadata(data, out)
        self.write_log(data, out, log_dict)

    def save_distributed(self, da_arr, out, parallel_store=True):

        # Either write chunks fully parallel
        if parallel_store:

            # Map `da_arr` chunk by chunk onto ``_write_parallel``
            nchk = len(da_arr.chunksize)
            writers = da_arr.map_blocks(self._write_parallel, nchk, self.vdsdir,
                                        dtype="int", chunks=(1,)*nchk)
            # res = result.persist()

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
                h5f.create_virtual_dataset(out.__class__.__name__, layout)

        # or use a semaphore to write to a single container sequentially
        else:

            # Initialize distributed lock
            lck = dd.lock.Lock(name='writer_lock')

            # Map `da_arr` chunk by chunk onto ``_write_sequential``
            writers = da_arr.map_blocks(self._write_sequential, nchk, out._filename,
                                        out.__class__.__name__, lck, dtype="int",
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
        # stuff) or rely on provided `log_dict` dictionary for logging
        if log_dict is None:
            cfg = dict(self.cfg)
            for key in ["noCompute", "chunkShape"]:
                cfg.pop(key)
        else:
            cfg = log_dict    
        logOpts = ""
        for k, v in cfg.items():
            logOpts += "\t{key:s} = {value:s}\n".format(key=k, value=str(v))
        out.log = logHead + logOpts

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
    def handle_metadata(self, *args):
        pass

    @abstractmethod
    def compute_sequential(self, *args):
        pass

    @abstractmethod
    def compute_parallel(self, *args):
        pass
