# coding: utf-8
# ex_mtmfft.py - Example script illustrating usage of `BaseData` in
#                combination with Dask and spectral estimation
# 
# Author: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Created: January 24 2019
# Last modified: <2019-01-25 17:38:11>


import dask.bag
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
# import dask.array as da
# from dask import delayed
from dask.diagnostics import ProgressBar
import numpy as np
import scipy.signal.windows as windows
import os
import sys
import time

sys.path.append("/mnt/hpx/it/dev/SpykeWave")

spw_path = os.path.abspath(".." + os.sep + "..")
if spw_path not in sys.path:
    sys.path.insert(0, spw_path)


import spykewave as sw

slurmComputation = False
#%%
if slurmComputation:
    cluster = SLURMCluster(processes=8,
                           cores=8,
                           memory="48GB",                                              
                           queue="DEV")

    cluster.start_workers(1)    
else:
    import socket
    cluster = LocalCluster(ip=socket.gethostname(),
                           n_workers=8,
                           threads_per_worker=1,
                           memory_limit="8G",
                           processes=False)

print("Waiting for workers to start")    
while len(cluster.workers) == 0:
    time.sleep(0.5)
client = Client(cluster)

print(client)

#%%
def time2sample(t, dt=0.001):
    return (t/dt).astype(dtype=int)

def sample2time(s, dt=0.001):
    return s*dt

# load test dataset
datadir = "/mnt/hpx/it/dev/SpykeWave/testdata/"
basename = "MT_RFmapping_session-168a1"


#%% Define trial from photodiode onsets
pdFile = os.path.join(datadir, basename + ".dpd")
pdData = sw.BaseData(pdFile, filetype="esi")

# trials start 250 ms before stimulus onset
pdOnset = np.array(pdData._segments[:,:])
iStart = time2sample(sample2time(pdOnset[0, pdOnset[1,:] == 1], 
                                 dt=pdData.hdr["tSample"]/1E9) - 0.25,
                     dt=0.001)
iEnd = time2sample(sample2time(pdOnset[0, pdOnset[1,:] == 1],
                               dt=pdData.hdr["tSample"]/1E9) + 0.5,
                   dt=0.001)

# construct trial definition matrix
intervals = np.stack((iStart,iEnd, np.tile(250, [iStart.size]).T), axis=1)

# remove very short trials
intervals = intervals[intervals[:,1]-intervals[:,0] > 500]

dataFiles = [os.path.join(datadir, basename + ext) 
             for ext in ["_xWav.lfp", "_xWav.mua"]]
data = sw.BaseData(dataFiles, trialdefinition=intervals, filetype="esi")



#%% Prepare spectral analysis
cfg = sw.spw_get_defaults(sw.mtmfft)
cfg.pop("data")
print(cfg)
cfg["taper"] = windows.dpss
cfg["fftAxis"] = 1
cfg["pad"] = "nextpow2"
cfg["tapsmofrq"] = 5


freq, protospec, win = sw.mtmfft(data.get_segment(0), **cfg)


def get_trial_powerspectrum(trialno, files, intervals, cfg):
    data = np.array(sw.BaseData(files, 
                                trialdefinition=intervals, 
                                filetype="esi").get_segment(trialno))
    freq, spec, win = sw.mtmfft(data, **cfg)
    spec = np.mean(np.absolute(spec), axis=0)
    return spec

specs = dask.bag.from_sequence(range(0,len(data._sampleinfo)))\
                .map(get_trial_powerspectrum,
                         files=dataFiles, intervals=intervals, cfg=cfg)
                
#%% Compute spectra                
                
print("Computing single trial powerspectra with dask")
with ProgressBar():
    result = np.stack(specs.compute())


avgSpec = np.mean(result, axis=0)


erp = np.zeros(data.get_segment(0).shape) 
for segment in data.segments:
    erp += segment
erp /= len(data._trialinfo)
tAxis = np.arange(erp.shape[1]) * data.hdr["tSample"]/1E6 - 250

#%% Plot result
import matplotlib.pyplot as plt
plt.ion()
channels = np.arange(30, 60)
fig, ax = plt.subplots(2,2)
ax[0,0].plot(tAxis, erp[channels,:].T)
ax[0,1].plot(tAxis, erp[channels+256,:].T)
ax[0,0].set_ylabel('LFP (a.u.)')
ax[0,1].set_ylabel('MUA (a.u.)')
ax[0,0].set_xlabel('Time (ms)')
ax[1,0].semilogy(freq, avgSpec[channels,:].T)
ax[1,0].set_xlim([1, 100])
ax[1,1].semilogy(freq, avgSpec[channels+256,:].T)
ax[1,1].set_xlim([1, 100])
ax[1,0].set_xlabel('Frequency (Hz)')
ax[1,0].set_ylabel('Power (a.u.)')

fig.tight_layout()
plt.draw()

