# -*- coding: utf-8 -*-
# 
# Syncopy's abstract base class for statistical data + regular children
# 

"""Statistics data

"""
# Builtin/3rd party package imports

from abc import ABC
from collections.abc import Iterator
import inspect
import numpy as np

# Local imports
from .base_data import BaseData
from syncopy.shared.parsers import scalar_parser, array_parser, io_parser
from syncopy.shared.errors import SPYValueError, SPYIOError
import syncopy as spy

__all__ = ["TimelockData"]


class StatisticalData(BaseData, ABC):
    """StatisticalData
    """    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
class TimelockData(StatisticalData):
    
    # FIXME: set functions should check that avg, var, dof, cov and time have
    #        matching shapes
    # FIXME: selectdata is missing
    # FIXME: documentation is missing
    # FIXME: tests are missing    
    

    _hdfFileDatasetProperties = ("avg", "var", "dof", "cov", "time")
    _defaultDimord = ["time", "channel"]

    @property        
    def time(self):
        """:class:`numpy.ndarray`: trigger-relative time axis """
        return self._time
    
    @time.setter
    def time(self, time):        
        self._time = time
    
    @property
    def channel(self):
        """ :class:`numpy.ndarray` : list of recording channel names """
        # if data exists but no user-defined channel labels, create them on the fly
        if self._channel is None and self.avg is not None:
            nChannel = self.avg.shape[self.dimord.index("channel")]        
            return np.array(["channel" + str(i + 1).zfill(len(str(nChannel)))
                           for i in range(nChannel)])            
        return self._channel

    @channel.setter
    def channel(self, channel):                                
        
        if channel is None:
            self._channel = None
            return
        
        if self.avg is None:
            raise SPYValueError("Syncopy: Cannot assign `channels` without data. " +
                  "Please assign data first")     
                    
        try:
            array_parser(channel, varname="channel", ntype="str", 
                         dims=(self.avg.shape[self.dimord.index("channel")],))
        except Exception as exc:
            raise exc
        
        self._channel = np.array(channel)
    
    def __str__(self):
        # Get list of print-worthy attributes
        ppattrs = [attr for attr in self.__dir__()
                   if not (attr.startswith("_") or attr in ["log", "trialdefinition", "hdr"])]
        ppattrs = [attr for attr in ppattrs
                   if not (inspect.ismethod(getattr(self, attr))
                           or isinstance(getattr(self, attr), Iterator))]
        
        ppattrs.sort()

        # Construct string for pretty-printing class attributes
        dsep = "' x '"
        dinfo = ""
        hdstr = "Syncopy {clname:s} object with fields\n\n"
        ppstr = hdstr.format(diminfo=dinfo + "'"  + \
                             dsep.join(dim for dim in self.dimord) + "' " if self.dimord is not None else "Empty ",
                             clname=self.__class__.__name__)
        maxKeyLength = max([len(k) for k in ppattrs])
        printString = "{0:>" + str(maxKeyLength + 5) + "} : {1:}\n"
        for attr in ppattrs:
            value = getattr(self, attr)
            if hasattr(value, 'shape'):
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
            ppstr += printString.format(attr, valueString)
        ppstr += "\nUse `.log` to see object history"
        return ppstr
    
    def __init__(self, 
                 time=None,
                 avg=None, 
                 var=None, 
                 dof=None, 
                 cov=None,                 
                 channel=None,
                 dimord=None):
        
        self._time = None
        self._avg = None
        self._var = None
        self._dof = None
        self._cov = None
        self._channel = None
        
        super().__init__(time=time,
                         avg=avg, 
                         var=var, 
                         dof=dof, 
                         cov=cov,
                         trialdefinition=None,                         
                         channel=channel,
                         dimord=dimord,
                         )
        
        self.time = time
        self.avg = avg
        self.var = var
        self.dof = dof
        self.cov = cov
        self.channel = channel
            
    def _get_trial(self):
        pass
    
    def selectdata(self):
        pass
    
    @property
    def avg(self):
        return self._avg
    
    @avg.setter
    def avg(self, avg):
        self._avg = avg
        
    @property
    def var(self):
        return self._var
    
    @var.setter
    def var(self, var):
        self._var = var    
        

    @property
    def dof(self):
        return self._dof
    
    @dof.setter
    def dof(self, dof):
        self._dof = dof   
    
    @property
    def cov(self):
        return self._cov
    
    @cov.setter
    def cov(self, cov):
        self._cov = cov
    
    
        
            
        
