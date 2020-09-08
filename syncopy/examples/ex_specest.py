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
# import matplotlib.pyplot as plt

from syncopy import *

# Import SynCoPy
import syncopy as spy

# Import artificial data generator
from syncopy.tests.misc import generate_artificial_data, figs_equal

import dask.distributed as dd
from time import time

# sys.exit()

if __name__ == "__main__":

    data = spy.load('/mnt/hpx/it/dev/testdata.spy/')
    # data = spy.load('~/Documents/job/SyNCoPy/Data/testdata.spy/')

    # from syncopy.plotting._plot_spectral import singlepanelplot, multipanelplot
    import matplotlib.pyplot as plt
    
    # data1 = generate_artificial_data()
    # data2 = data1
    # spy.singlepanelplot(data1, data2, channels=["channel1", "channel2"], toilim=[1.2, 1.5], overlay=False)
    
    # spy.singlepanelplot(data1, channels=["channel1", "channel2"], toilim=[0.2, 0.5])
    
    sys.exit()

    # cfg = spy.get_defaults(spy.freqanalysis)
    # cfg.method = 'mtmfft'
    # cfg.taper = "dpss"
    # cfg.tapsmofrq = 20
    # cfg.output = 'pow'
    # cfg.keeptrials = True
    # cfg.keeptapers = True
    # cfg.select = {"trials": [0, 10]}
    # overallSpectrum = spy.freqanalysis(cfg, data)
    
    # # spy.singlepanelplot(overallSpectrum, channels=[10, 50, 20], tapers=[3, 0], foilim=[30, 80],
    # #                 avg_channels=False, avg_tapers=True, grid=True)
    # # # multipanelplot(overallSpectrum, channels=[10, 50, 20], tapers=[3, 0], foilim=[30, 80],
    # # #                panels="tapers", avg_channels=True, avg_tapers=False, avg_trials=True)
    
    # fig1, fig2 = spy.singlepanelplot(overallSpectrum, overallSpectrum, tapers=[3, 0], 
    #                              foilim=[30, 80], avg_channels=False, avg_tapers=True, grid=True, overlay=True)
    
    # plt.show()
    # sys.exit()

    cfg = spy.get_defaults(spy.freqanalysis)
    cfg.method = 'mtmconvol'
    cfg.taper = 'dpss'
    cfg.output = 'pow'
    cfg.tapsmofrq = 20
    cfg.keeptrials = True
    cfg.keeptapers = True
    # cfg.foi = [30, 40, 50]
    # cfg.foilim = [30, 80]
    # cfg.toi = 0.25
    # cfg.toi = "all"
    # cfg.pad = 'nextpow2'
    # cfg.toi = [-0.1, 0.0, 0.5]
    # cfg.toi = [-0.1, 0.0, 0.2]
    cfg.toi = np.arange(-0.1, 0.5, 0.05) 
    # cfg.toi = np.arange(-0.1, 0.5, 0.1) 
    # cfg.toi = "all"
    cfg.t_ftimwin = 0.05
    # cfg.t_ftimwin = 0.75
    # cfg.pad = 'nextpow2'
    # cfg.select = {"toilim": [-0.25, 0]}
    cfg.select = {"trials": [0, 10]}
    # cfg.select = {"trials": [0, 10, 20]}
    # cfg.select = {"trials": [0, 10, 20], "toilim": [-0.001, 0.05]}
    # tfSpectrum = spy.freqanalysis(cfg, data)
    tfSpectrum = spy.freqanalysis(cfg, data)
    
    fig = tfSpectrum.singlepanelplot(toilim=[-0.1, 0.1]) 
    
    sys.exit()

    singlepanelplot(tfSpectrum, channels=[10, 50, 20], foilim=[30, 80],
                    avg_channels=True, avg_tapers=True, grid=True)
    # multipanelplot(tfSpectrum, channels=[10, 50, 20], foilim=[30, 80], panels="channels", 
    #                 avg_channels=False, avg_tapers=True, avg_trials=True)
    plt.show()
    sys.exit()

    cfg = spy.get_defaults(spy.freqanalysis)
    cfg.method = 'wavelet'
    cfg.wav = 'DOG'
    cfg.order = 4
    cfg.output = 'pow'
    cfg.keeptrials = True
    cfg.toi = "all"
    # cfg.foi = [30, 40, 50]
    cfg.foilim = [30, 80]
    # cfg.toi = np.arange(-0.1, 0.5, 0.05) 
    # cfg.toi = [-0.1, 0.0, 0.2]
    # cfg.select = {"trials": [0, 10]}
    cfg.select = {"trials": [0, 10], "toilim": [-0.001, 0.05]}
    tfSpectrum = spy.freqanalysis(cfg, data, select=dict(cfg.select))

    sys.exit()
    
    sys.exit()
    baselineSpectrum = spy.freqanalysis(cfg, data)
    
    bsc = baselineSpectrum.copy()


    
    
    cfg = spy.get_defaults(spy.freqanalysis)
    cfg.method = 'mtmfft'
    cfg.taper = "dpss"
    cfg.tapsmofrq = 20
    cfg.output = 'pow'
    cfg.keeptrials = False
    cfg.keeptapers = True
    overallSpectrum = spy.freqanalysis(cfg, data)
    sys.exit()
    
    # cfg = spy.get_defaults(spy.freqanalysis)
    # cfg.method = 'mtmfft'
    # cfg.taper = 'dpss'
    # cfg.output = 'pow'
    # cfg.tapsmofrq = 20
    # cfg.keeptrials = False
    # cfg.keeptapers = False
    # cfg.pad = 'nextpow2'
    # cfg.select = {"toilim": [-0.25, 0]}
    # cfg.data = data
    # baselineSpectrum = spy.freqanalysis(cfg=cfg)

    # sys.exit()

    # client = spy.esi_cluster_setup(n_jobs=10, partition="DEV", mem_per_job="2GB")
    
    
    # cfg = spy.get_defaults(spy.freqanalysis)
    # cfg.method = 'mtmfft'
    # cfg.taper = "dpss"
    # cfg.tapsmofrq = 20
    # cfg.output = 'pow'
    # cfg.keeptrials = False
    # cfg.keeptapers = True
    # overallSpectrum = spy.freqanalysis(cfg, data)
    # sys.exit()
    
    # # cfg = spy.get_defaults(spy.freqanalysis)
    # # cfg.method = 'mtmfft'
    # # cfg.output = 'pow'
    # # cfg.keeptrials = False
    # # cfg.foi = spy.np.arange(1,100)
    # # cfg.select = {"toilim": [-0.25, 0]}
    # # baselineSpectrum = spy.freqanalysis(cfg, data)
    # # plt.ion()
    # # plt.figure()
    # # plt.plot(baselineSpectrum.freq, baselineSpectrum.data[0, 0, :, 0])
    
    # # sys.exit()
    
    # try:
    #     client = dd.get_client()
    # except:
    #     client = spy.esi_cluster_setup(n_jobs=10, partition="DEV", mem_per_job="2GB")
    #     # client = spy.esi_cluster_setup(n_jobs=8, partition="DEV", mem_per_job="4GB")

    # cfg = spy.StructDict()
    # cfg.output = 'pow'        
    # cfg.taper = "dpss"
    # cfg.keeptrials = False
    # # cfg.keeptapers = False
    # # t0 = time()
    # spec = spy.freqanalysis(cfg, data)
    # # print("Elapsed time: ", time() - t0)
    
    # sys.exit()

    # nc = 10
    # ns = 30
    # data = np.arange(1, nc*ns + 1, dtype="float").reshape(ns, nc)
    # trl = np.vstack([np.arange(0, ns, 5),
    #                  np.arange(5, ns + 5, 5),
    #                  np.ones((int(ns/5), )),
    #                  np.ones((int(ns/5), )) * np.pi]).T

    from syncopy.shared.tools import StructDict
    from syncopy.datatype import AnalogData, SpectralData, padding
    from syncopy.shared import esi_cluster_setup

    # create uniform `cfg` for testing on SLURM
    cfg = StructDict()
    cfg.method = "mtmfft"
    cfg.taper = "dpss"
    cfg.output = 'abs'
    cfg.tapsmofrq = 9.3
    cfg.keeptrials = True
    artdata = generate_artificial_data(nTrials=2, nChannels=16, equidistant=True, inmemory=True)
    
    # artdata.save('test', overwrite=True)
    # bdata = spy.load('test')
    spec1 = spy.freqanalysis(artdata, cfg)
    

    # Set up "global" parameters for data objects to be tested
    nChannels = 10
    nSamples = 30
    nTrials = 5
    nFreqs = 15
    nSpikes = 50
    data = {}
    trl = {}
    
    # Generate 2D array simulating an AnalogData array
    data["AnalogData"] = np.arange(1, nChannels * nSamples + 1).reshape(nSamples, nChannels)
    trl["AnalogData"] = np.vstack([np.arange(0, nSamples, 5),
                                   np.arange(5, nSamples + 5, 5),
                                   np.ones((int(nSamples / 5), )),
                                   np.ones((int(nSamples / 5), )) * np.pi]).T

    # Generate a 4D array simulating a SpectralData array
    data["SpectralData"] = np.arange(1, nChannels * nSamples * nTrials * nFreqs + 1).reshape(nSamples, nTrials, nFreqs, nChannels)
    trl["SpectralData"] = trl["AnalogData"]

    # Use a fixed random number generator seed to simulate a 2D SpikeData array
    seed = np.random.RandomState(13)
    data["SpikeData"] = np.vstack([seed.choice(nSamples, size=nSpikes),
                                   seed.choice(np.arange(1, nChannels + 1), size=nSpikes), 
                                   seed.choice(int(nChannels/2), size=nSpikes)]).T
    trl["SpikeData"] = trl["AnalogData"]

    # Use a simple binary trigger pattern to simulate EventData
    data["EventData"] = np.vstack([np.arange(0, nSamples, 5),
                                   np.zeros((int(nSamples / 5), ))]).T
    
    data["EventData"] = np.vstack([np.arange(0, nSamples, 2), 
                                   np.zeros((int(nSamples / 2), ))]).T  
    data["EventData"][1::3, 1] = 1
    data["EventData"][2::3, 1] = 2
    trl["EventData"] = trl["AnalogData"]
    
    from syncopy.datatype.base_data import VirtualData, Selector
    
    spk = spy.SpikeData(data=data["SpikeData"], trialdefinition=trl["SpikeData"], samplerate=2) 

    sel=Selector(spk, select={"toi":[1.0, 1.5]})  
    
    evt = spy.EventData(data=data["EventData"], trialdefinition=trl["EventData"], samplerate=2)
       
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
    # adata = generate_artificial_data(nTrials=20, nChannels=256, equidistant=False, overlapping=True)        # ~50MB
    # adata.channel = ["channel" + str(i + 1) for i in range(256)]
    # # adata = generate_artificial_data(nTrials=100, nChannels=1024, equidistant=False)        # ~1.14GB
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
