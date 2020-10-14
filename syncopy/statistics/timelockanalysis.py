# -*- coding: utf-8 -*-
# 
# Syncopy timelock-analysis methods
# 

import os
import numpy as np
import syncopy as spy
from tqdm.auto import tqdm
from syncopy.shared.parsers import data_parser
from syncopy.shared.tools import StructDict
import syncopy as spy

__all__ = ["timelockanalysis"]

def timelockanalysis(data, trials=None):
    """Prototype function for averaging :class:`~syncopy.AnalogData` across trials
    
    Parameters
    ----------
    data : Syncopy :class:`~syncopy.AnalogData` object
        Syncopy :class:`~syncopy.AnalogData` object to be averaged across trials
    trials : :class:`numpy.ndarray`
        Array of trial indices to be used for averaging
    
    Returns
    -------
    resdict : dict
        Dictionary with keys "avg", "var", "dof" "time", "channel" representing
        calculated average (across `trials`), variance (across `trials`), degrees 
        of freedom, time axis, and channel labels

    Notes
    -----
    This function is merely a proof of concept implementation for averaging data 
    across trials with using an efficient online algorithm. The final version 
    for release will change substantially.
    """
            
    # FIXME: There are currently no tests for this function.    
    # FIXME: Handle non-standard dimords    
    # FIXME: the output should be a "proper" Syncopy object
    
    try:
        data_parser(data, varname="data", empty=False, 
                    dataclass=spy.AnalogData, dimord=spy.AnalogData._defaultDimord)
    except Exception as exc:
        raise exc

    
    if trials is None:
        trials = np.arange(len(data.trials))

    intTimeAxes = [(np.arange(0, stop - start) + offset)
                for (start, stop, offset) in data.trialdefinition[trials, :3]]
    intervals = np.array([(x.min(), x.max()) for x in intTimeAxes])

    avgTimeAxis = np.arange(start=intervals.min(),
                            stop=intervals.max()+1)                        

    targetShape = (avgTimeAxis.size, len(data.channel))
    avg = np.zeros(targetShape)
    var = np.zeros(targetShape)
    dof = np.zeros(targetShape)
    oldAvg = np.zeros(targetShape)

    nTrial = len(data.trials)

    fullArray = np.empty((avgTimeAxis.size, len(data.channel), nTrial))
    fullArray[:] = np.nan

    # Welford's online method for computing-variance
    # http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/

    for n, iTrial in enumerate(tqdm(trials)):
        x = data.trials[iTrial]
        trialTimeAxis = intTimeAxes[n]
            
        targetIndex = np.in1d(avgTimeAxis, trialTimeAxis, assume_unique=True)
        dof[targetIndex, :] += 1
        oldAvg = avg.copy()
        avg[targetIndex, :] += (x-avg[targetIndex, :]) / (dof[targetIndex, :])
        var[targetIndex, :] += (x-avg[targetIndex, :]) * (x - oldAvg[targetIndex, :])    
        if np.mod(iTrial, 10) == 0:
            data.clear()    

    dof -= 1
    var /= dof+1
    
    result = StructDict()
    result.avg = avg
    result.var = var
    result.dof = dof
    result.channel = data.channel    
    result.time = avgTimeAxis / data.samplerate
    
    return result


if __name__ == "__main__":
    analogData = spy.load("~/testdata.spy")
    conditions = np.unique(analogData.trialinfo)
    tl = []
    for condition in conditions:
        selection = np.nonzero(analogData.trialinfo == condition)[0]
        tl.append(timelockanalysis(analogData, selection))
    chan = list(analogData.channel).index("vprobeMUA_020")

    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    for t in tl:
        plt.plot(t.time,  savgol_filter(t.avg[:, chan], 71, 3))    
        plt.fill_between(t.time, 
                        t.avg[:, chan] + np.sqrt(t.var[:, chan]/t.dof[:, chan]**2),
                        t.avg[:, chan] - np.sqrt(t.var[:, chan]/t.dof[:, chan]**2), 
                        alpha=0.5)
    plt.show()



