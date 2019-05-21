# -*- coding: utf-8 -*-
# 
# 
# 
# Created: 2019-02-25 13:08:56
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-21 09:59:34>

# Add SynCoPy package to Python search path
import os
import sys
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)

# Import SynCoPy
import syncopy as spy

# Import artificial data generator
from syncopy.tests.misc import generate_artifical_data

if __name__ == "__main__":
    
    adata = generate_artifical_data(nTrials=20, nChannels=256)

    spec = spy.freqanalysis(adata, method="mtmfft")

    sys.exit()
    
    print("Save data")
    dat.save('example2.spy')
    dat = None
    del dat
    print("Load data")
    # data = spy.AnalogData(filename="~/Projects/SyNCoPy/example2.spy")
    data = spy.AnalogData(filename="example2.spy")

    method = "with_dask"
    if method == "with_dask":
        import socket
        cluster = LocalCluster(ip=socket.gethostname(),
                               n_workers=3,
                               threads_per_worker=1,
                               memory_limit="4G",
                               processes=False)
        client = Client(cluster)

    print("Calculate tapered spectra")
    out = spy.SpectralData()
    mtmfft = spy.MultiTaperFFT(1 / data.samplerate, output="abs")
    mtmfft.initialize(data)
    result = mtmfft.compute(data, out, methodName=method)
    # out.save("mtmfft_spectrum")

    # print("Calculate wavelet spectra")
    # outWavelet = spy.SpectralData()
    # wavelet = spy.WaveletTransform(1 / data.samplerate, stepsize=10, output="abs")
    # wavelet.initialize(data)
    # wavelet.compute(data, outWavelet, methodName="sequentially")

    # #
    # fig, ax = plt.subplots(3, 1)
    # ax[0].pcolormesh(outWavelet.time[0], outWavelet.freq,
    #                  outWavelet.trials[0][:, 0, :, 0].T)
    # ax[0].set_ylim([0, 100])
    # ax[0].set_ylabel('Frequency (Hz)')

    # ax[1].plot(data.time[0], np.mean(data.trials[0], axis=1))
    # ax[1].set_xlabel('Time (s)')

    # ax[2].plot(out.freq, out.trials[0][0, 0, :, 0])
    # ax[2].plot(outWavelet.freq,
    #            np.squeeze(np.mean(outWavelet.trials[0][:, 0, :, 0], axis=0)))
    # ax[2].set_xlabel('Frequency (Hz)')
    # ax[2].set_ylabel('Power')
    # plt.show()
