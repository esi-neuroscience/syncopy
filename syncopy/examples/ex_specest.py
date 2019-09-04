# -*- coding: utf-8 -*-
#
#
#
# Created: 2019-02-25 13:08:56
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-04 09:06:06>

# # Builtin/3rd party package imports
# import dask.distributed as dd

# Add SynCoPy package to Python search path
import os
import sys
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)
import numpy as np
import matplotlib.pyplot as plt

from syncopy import *

# Import SynCoPy
import syncopy as spy

# Import artificial data generator
from syncopy.tests.misc import generate_artifical_data

# sys.exit()

if __name__ == "__main__":

    # nc = 10
    # ns = 30
    # data = np.arange(1, nc*ns + 1, dtype="float").reshape(ns, nc)
    # trl = np.vstack([np.arange(0, ns, 5),
    #                  np.arange(5, ns + 5, 5),
    #                  np.ones((int(ns/5), )),
    #                  np.ones((int(ns/5), )) * np.pi]).T

    from syncopy.datatype import AnalogData, SpectralData, StructDict, padding
    from syncopy.shared import esi_cluster_setup

    # create uniform `cfg` for testing on SLURM
    cfg = StructDict()
    cfg.method = "mtmfft"
    cfg.taper = "dpss"
    cfg.output = 'abs'
    cfg.tapsmofrq = 9.3
    cfg.keeptrials = True
    artdata = generate_artifical_data(nTrials=2, nChannels=16, equidistant=True, inmemory=True)
    
    # artdata.save('test', overwrite=True)
    # bdata = spy.load('test')
    spec1 = spy.freqanalysis(artdata, cfg)
    sys.exit()
    client = dd.Client()
    spec2 = spy.freqanalysis(artdata, cfg)
    
    cfg.chan_per_worker = 7
    spec3 = spy.freqanalysis(artdata, cfg)

    # # Constructe simple trigonometric signal to check FFT consistency: each
    # # channel is a sine wave of frequency `freqs[nchan]` with single unique
    # # amplitude `amp` and sampling frequency `fs`
    # nChannels = 32
    # nTrials = 8
    # fs = 1024
    # fband = np.linspace(0, fs/2, int(np.floor(fs/2) + 1))
    # freqs = np.random.choice(fband[1:-1], size=nChannels, replace=False)
    # amp = np.pi
    # phases = np.random.permutation(np.linspace(0, 2*np.pi, nChannels))
    # t = np.linspace(0, nTrials, nTrials*fs)
    # sig = np.zeros((t.size, nChannels), dtype="float32")
    # for nchan in range(nChannels):
    #     sig[:, nchan] = amp*np.sin(2*np.pi*freqs[nchan]*t + phases[nchan])
    # 
    # trialdefinition = np.zeros((nTrials, 3), dtype="int")
    # for ntrial in range(nTrials):
    #     trialdefinition[ntrial, :] = np.array([ntrial*fs, (ntrial + 1)*fs, 0])
    # 
    # adata = AnalogData(data=sig, samplerate=fs,
    #                    trialdefinition=trialdefinition)
    # 
    # 
    # 
    # sys.exit()
    # 
    # # client = dd.Client()
    # 
    # nChannels = 1
    # nTrials = 4
    # nSines = 8
    # fs = 1024
    # 
    # 
    # freqs = np.random.permutation(np.linspace(50, 250, nChannels))
    # amp = np.pi
    # phases = np.random.permutation(np.linspace(0, 2*np.pi, nChannels))
    # t = np.linspace(0, nTrials, nTrials*fs)
    # 
    # sig = np.zeros((t.size, nChannels), dtype="float32")
    # k = 0
    # for nchan in range(nChannels):
    #     sig[:, nchan] = amp*np.sin(2*np.pi*freqs[nchan]*t + phases[nchan])
    # 
    # trialdefinition = np.zeros((nTrials, 3), dtype="int")
    # for ntrial in range(nTrials):
    #     trialdefinition[ntrial, :] = np.array([ntrial*fs, (ntrial + 1)*fs, 0])
    # 
    # adata = spy.AnalogData(data=sig, samplerate=fs, trialdefinition=trialdefinition)
    # 
    # spec = spy.freqanalysis(adata, method="mtmfft", taper="hann", output="pow")
    # 
    # 
    # # plt.ion()
    # # ax = plt.subplot2grid((2, nTrials), (0, 0), colspan=nTrials)
    # # (fig, ax_arr) = plt.subplots(2, nTrials, tight_layout=True,
    # #                              gridspec_kw={'wspace':0.32,'left':0.01,'right':0.93,
    # #                                           'hspace':0.01},
    # #                              figsize=[5.6,4.3])
    # # ax = plt.subplot(ax_arr[0, :])
    # # ax.plot(t, sig.flatten())
    # # for ntrial in range(nTrials):
    # #     ax = plt.subplot2grid((2, nTrials), (1, ntrial))
    # #     ax.plot(spec.data[ntrial, ...].flatten())
    # 
    # sys.exit()
    # 
    # # FIXME: channel assignment is only temporarily necessary
    # adata = generate_artifical_data(nTrials=20, nChannels=256, equidistant=False, overlapping=True)        # ~50MB
    # adata.channel = ["channel" + str(i + 1) for i in range(256)]
    # # adata = generate_artifical_data(nTrials=100, nChannels=1024, equidistant=False)        # ~1.14GB
    # # adata.channel = ["channel" + str(i + 1) for i in range(1024)]
    # 
    # 
    # # ff = spy.SpectralData()
    # import numpy as np
    # spec = spy.freqanalysis(adata, method="mtmfft", taper="hann", keeptrials=False, keeptapers=True, foi=np.linspace(200,400,100))
    # 
    # sys.exit()
    # # 
    # # print("Save data")
    # # dat.save('example2.spy')
    # # dat = None
    # # del dat
    # # print("Load data")
    # # # data = spy.AnalogData(filename="~/Projects/SyNCoPy/example2.spy")
    # # data = spy.AnalogData(filename="example2.spy")
    # # 
    # # method = "with_dask"
    # # if method == "with_dask":
    # #     import socket
    # #     cluster = LocalCluster(ip=socket.gethostname(),
    # #                            n_workers=3,
    # #                            threads_per_worker=1,
    # #                            memory_limit="4G",
    # #                            processes=False)
    # #     client = Client(cluster)
    # # 
    # # print("Calculate tapered spectra")
    # # out = spy.SpectralData()
    # # mtmfft = spy.MultiTaperFFT(1 / data.samplerate, output="abs")
    # # mtmfft.initialize(data)
    # # result = mtmfft.compute(data, out, methodName=method)
    # # # out.save("mtmfft_spectrum")
    # # 
    # # # print("Calculate wavelet spectra")
    # # # outWavelet = spy.SpectralData()
    # # # wavelet = spy.WaveletTransform(1 / data.samplerate, stepsize=10, output="abs")
    # # # wavelet.initialize(data)
    # # # wavelet.compute(data, outWavelet, methodName="sequentially")
    # # 
    # # # #
    # # # fig, ax = plt.subplots(3, 1)
    # # # ax[0].pcolormesh(outWavelet.time[0], outWavelet.freq,
    # # #                  outWavelet.trials[0][:, 0, :, 0].T)
    # # # ax[0].set_ylim([0, 100])
    # # # ax[0].set_ylabel('Frequency (Hz)')
    # # 
    # # # ax[1].plot(data.time[0], np.mean(data.trials[0], axis=1))
    # # # ax[1].set_xlabel('Time (s)')
    # # 
    # # # ax[2].plot(out.freq, out.trials[0][0, 0, :, 0])
    # # # ax[2].plot(outWavelet.freq,
    # # #            np.squeeze(np.mean(outWavelet.trials[0][:, 0, :, 0], axis=0)))
    # # # ax[2].set_xlabel('Frequency (Hz)')
    # # # ax[2].set_ylabel('Power')
    # # # plt.show()
