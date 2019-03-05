# coding: utf-8
# ex_mtmfft.py - Example script illustrating usage of `BaseData` in
#                combination with Dask and spectral estimation
#
# Created: January 24 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-01 17:39:11>


from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster

import numpy as np
import scipy.signal.windows as windows
import os
import sys
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

spw_path = os.path.abspath(".." + os.sep + "..")
if spw_path not in sys.path:
    sys.path.insert(0, spw_path)

import spykewave as spy

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


if __name__ == "__main__":

    # Set path to data directory
    # datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
              # + os.sep + "testdata" + os.sep
    datadir = os.path.join(os.sep, "mnt", "hpx", "it",
                           "dev", "SpykeWave", "testdata")
    basename = "MT_RFmapping_session-168a1"
    data = spy.AnalogData(filename=os.path.join(datadir, basename + '.spy'),
                          mode='r')

    cfg = spy.spy_get_defaults(spy.mtmfft)
    cfg.pop("obj")
    print(cfg)
    cfg["taper"] = windows.hanning
    cfg["pad"] = "nextpow2"
    cfg["tapsmofrq"] = 5

    # run spectral analysis
    spec = spy.mtmfft(data, **cfg)
    avgSpec = np.zeros(spec._shapes[0])[0, ...]
    for trial in tqdm(spec.trials, desc="Averaging spectra..."):
        avgSpec += np.absolute(trial)
    avgSpec /= len(spec.trials)

    chanIdx = np.arange(35, 40)
    plt.ion()
    fig, ax = plt.subplots(1)
    ax.plot(spec.freq, avgSpec[0, chanIdx, :].T, '.-')
    ax.set_xlim([0.1, 100])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (a.u.)')
    ax.legend(np.array(spec.channel)[chanIdx])
    fig.tight_layout()
    plt.draw()
