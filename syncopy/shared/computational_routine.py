# -*- coding: utf-8 -*-
#
# ALREADY KNOW YOU THAT WHICH YOU NEED
# 
# Created: 2019-05-13 09:18:55
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-21 21:10:23>

# Builtin/3rd party package imports
from abc import ABC, abstractmethod
from copy import copy

# Local imports
from .parsers import get_defaults
from syncopy import __storage__

__all__ = []


class ComputationalRoutine(ABC):

    # The actual workhorse 
    def computeFunction(x): return None

    # Manager that calls ``computeFunction`` (sets up `dask` etc. )
    def computeMethod(x): return None

    def __init__(self, *argv, **kwargs):
        self.defaultCfg = get_defaults(self.computeFunction)
        self.cfg = copy(self.defaultCfg)
        self.cfg.update(**kwargs)
        self.argv = argv
        self.chunkShape = None
        self.outputChunks = None
        self.dtype = None
        self.stackingdepth = None

    def initialize(self, data, stackingdepth):
        
        # FIXME: this only works for data with equal output trial lengths
        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        self.chunkShape, self.dtype = self.computeFunction(data.trials[0],
                                                            *self.argv,
                                                            **dryRunKwargs)

        # For trials of unequal length, compute output chunk-shape individually
        chk_list = [self.chunkShape]
        if np.any([data._shapes[0] != sh for sh in data._shapes]):
            for tk in range(1, len(data.trials)):
                chk_list.append(self.computeFunction(data.trials[tk],
                                                     *self.argv,
                                                     **dryRunKwargs)[0])
        else:
            chk_list += [chk] * (len(data.trials) -  1)
        self.outputChunks = chk_list

        # Finally compute aggregate shape of output data
        self.outputShape = (stackingdepth,) + self.chunkShape

    def compute(self, data, out, parallel=False, parallel_store=None, method=None):

        # By default, use VDS storage for parallel computing
        if parallel_store is None:
            parallel_store = parallel
        
        # Create HDF5 dataset of appropriate dimension
        self.preallocate_output(out, parallel_store=parallel_store)

        # The `method` keyword can be used to override the `parallel` flag
        if method is None:
            if parallel:
                computeMethod = compute_parallel
            else:
                computeMethod = compute_sequential
        else:
            computeMethod = getattr(self, "compute_" + method, None)

        # Perform actual computation
        computeMethod(data, out)

        # Store meta-data, write log and get outta here
        self.handle_metadata(data, out)
        self.write_log(data, out)
        return out

    def preallocate_output(self, out, parallel_store=False):

        # In case parallel writing via VDS storage is requested, prepare
        # directory for by-chunk HDF5 containers and create virtual layout
        if parallel_store:
            vds_dir = os.path.splitext(os.path.basename(out._filename))[0]
            vds_dir = os.path.join(__storage__, vds_dir)
            os.mkdir(vds_dir)

            layout = h5py.VirtualLayout(shape=self.outputShape, dtype=self.dtype)
            with h5py.File(out._filename, mode="w") as h5f:
                h5f.create_virtual_dataset(name=out.__class__.__name__, layout)

        # Create regular HDF5 dataset for sequential writing
        else:
            with h5py.File(out._filename, mode="w") as h5f:
                h5f.create_dataset(name=out.__class__.__name__,
                                   dtype=self.dtype, shape=self.outputShape)
    
    def write_log(self, data, out):
        # Write log
        out._log = str(data._log) + out._log
        logHead = "computed {name:s} with settings\n".format(name=self.computeFunction.__name__)

        logOpts = ""
        for k, v in self.cfg.items():
            logOpts += "\t{key:s} = {value:s}\n".format(key=k, value=str(v))

        out.log = logHead + logOpts

    # @abstractmethod
    # def preallocate_output(self, *args, parallel=False):
    #     pass

    @abstractmethod
    def handle_metadata(self, *args):
        pass

    @abstractmethod
    def compute_sequential(self, *args):
        pass

    @abstractmethod
    def compute_parallel(self, *args):
        pass
    
