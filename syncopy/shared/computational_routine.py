# -*- coding: utf-8 -*-
#
# ALREADY KNOW YOU THAT WHICH YOU NEED
# 
# Created: 2019-05-13 09:18:55
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-13 16:09:57>

# Builtin/3rd party package imports
from abc import ABC, abstractmethod
from copy import copy

# Local imports
from .parsers import get_defaults

__all__ = ["ComputationalRoutine"]


class ComputationalRoutine(ABC):

    def computeFunction(x): return None

    def computeMethod(x): return None

    def __init__(self, *argv, **kwargs):
        self.defaultCfg = get_defaults(self.computeFunction)
        self.cfg = copy(self.defaultCfg)
        self.cfg.update(**kwargs)
        self.argv = argv
        self.outputShape = None
        self.dtype = None

    # def __call__(self, data, out=None)

    def initialize(self, data):
        # FIXME: this only works for data with equal output trial lengths
        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        self.outputShape, self.dtype = self.computeFunction(data.trials[0],
                                                            *self.argv,
                                                            **dryRunKwargs)

    def compute(self, data, out, methodName="sequentially"):

        self.preallocate_output(data, out)
        result = None

        computeMethod = getattr(self, "compute_" + methodName, None)
        if computeMethod is None:
            raise AttributeError

        computeMethod(data, out)

        self.handle_metadata(data, out)
        self.write_log(data, out)
        return out

    def write_log(self, data, out):
        # Write log
        out._log = str(data._log) + out._log
        logHead = "computed {name:s} with settings\n".format(name=self.computeFunction.__name__)

        logOpts = ""
        for k, v in self.cfg.items():
            logOpts += "\t{key:s} = {value:s}\n".format(key=k, value=str(v))

        out.log = logHead + logOpts

    @abstractmethod
    def preallocate_output(self, *args):
        pass

    @abstractmethod
    def handle_metadata(self, *args):
        pass

    @abstractmethod
    def compute_sequentially(self, *args):
        pass

