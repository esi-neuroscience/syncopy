
import os
import numpy as np
import syncopy as spy
import matplotlib.pyplot as plt

def timelockanalysis(data):

    intTimeAxes = [(np.arange(0, stop - start) + offset)
                for (start, stop, offset) in data.trialdefinition[:, :3]]
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

    for iTrial, trial in enumerate(data.trials):
        x = trial
        trialTimeAxis = intTimeAxes[iTrial]
            
        targetIndex = np.in1d(avgTimeAxis, trialTimeAxis, assume_unique=True)
        dof[targetIndex, :] += 1
        oldAvg = avg.copy()
        avg[targetIndex, :] += (x-avg[targetIndex, :]) / (dof[targetIndex, :])
        var[targetIndex, :] += (x-avg[targetIndex, :]) * (x - oldAvg[targetIndex, :])    
        if np.mod(iTrial, 10) == 0:
            data.clear()    

    dof -= 1
    var /= dof+1
    
    result = spy.StructDict()
    result.avg = avg
    result.var = var
    result.dof = dof
    result.time = avgTimeAxis
    
    return result


if __name__ == "__main__":
    analogData = spy.load("~/syncopy-testdata.spy")
    tl = timelockanalysis(analogData)
    
    chan = 26
    plt.plot(tl.time, tl.avg[:, chan], '-')    
    plt.fill_between(tl.time, 
                     tl.avg[:, chan] + np.sqrt(tl.var[:, chan]),
                     tl.avg[:, chan] - np.sqrt(tl.var[:, chan]), 
                     alpha=0.5)
    plt.show()



