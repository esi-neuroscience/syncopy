# coding: utf-8
# ex_mtmfft.py - Example script illustrating usage of `BaseData` in
#                combination with Dask and spectral estimation
#
# Created: January 24 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-07 12:53:09>

# Add SynCoPy package to Python search path
import os
import sys
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from syncopy import *

# Import SynCoPy
import syncopy as spy
from syncopy.specest.wavelets import cwt, Morlet

# Import artificial data generator
from syncopy.tests.misc import generate_artificial_data, figs_equal


if __name__ == "__main__":

    # fs = 1e3
    # N = 1e5
    # amp = 2 * np.sqrt(2)
    # noise_power = 0.01 * fs / 2
    # time = np.arange(N) / float(fs)
    
    # tstart = -29.5
    # tstop = 70.5
    # time = (np.arange(0, (tstop - tstart) * fs) + tstart * fs) / fs
    # time1 = np.arange(time.size) / fs
    
    # mod = 500*np.cos(2*np.pi*0.0625*time)
    # mod = 500*np.cos(2*np.pi*0.125*time)
    # carrier = amp * np.sin(2*np.pi*3e2*time + mod)
    # noise = np.random.normal(scale=np.sqrt(noise_power),
    #                         size=time.shape)
    # # noise *= np.exp(-time1/5)
    # noise *= np.exp(-np.arange(time.size) / (5*fs))
    # x = carrier + noise

    # # Trials: stitch together [x, x, x]
    # # channels: 
    # # 1, 3, 5, 7: mod -> 0.0625 * time
    # # 0, 2, 4, 6: mod -> 0.125 * time
    
    # Construct high-frequency signal modulated by slow oscillating cosine and 
    # add time-decaying noise
    nChannels = 8
    nChan2 = int(nChannels / 2)
    nTrials = 3
    fs = 1000
    seed = 151120
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    numType = "float32"
    modPeriods = [0.125, 0.0625]
    rng = np.random.default_rng(seed)
    tStart = -29.5
    tStop = 70.5
    t0 = -np.abs(tStart * fs).astype(np.intp)
    time = (np.arange(0, (tStop - tStart) * fs, dtype=numType) + tStart * fs) / fs
    N = time.size
    carriers = np.zeros((N, 2), dtype=numType)
    modulators = np.zeros((N, 2), dtype=numType)
    noise_decay = np.exp(-np.arange(N) / (5*fs))
    
    fadeIn = -9.5
    fadeOut = 50.5
    fader = np.ones((N,), dtype=numType)
    if fadeIn is None:
        fadeIn = tStart
    if fadeOut is None:
        fadeOut = tStop
    fadeIn = np.arange(0, (fadeIn - tStart) * fs, dtype=np.intp)
    fadeOut = np.arange((fadeOut - tStart) * fs, 100 * fs, dtype=np.intp)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    fader[fadeIn] = sigmoid(np.linspace(-2 * np.pi, 2 * np.pi, fadeIn.size))
    fader[fadeOut] = sigmoid(-np.linspace(-2 * np.pi, 2 * np.pi, fadeOut.size))
    sys.exit()
    for k, period in enumerate(modPeriods):
        modulators[:, k] = 500 * np.cos(2 * np.pi * period * time)
        carriers[:, k] = fader * amp * np.sin(2 * np.pi * 3e2 * time + modulators[:, k])
        
    # For trials: stitch together carrier + noise, each trial gets its own (fixed 
    # but randomized) noise term, channels differ by period in modulator, stratified
    # by trials, i.e., 
    # Trial #0, channels 0, 2, 4, 6, ...: mod -> 0.125 * time
    #                    1, 3, 5, 7, ...: mod -> 0.0625 * time
    # Trial #1, channels 0, 2, 4, 6, ...: mod -> 0.0625 * time
    #                    1, 3, 5, 7, ...: mod -> 0.125 * time
    even = [None, 0, 1]
    odd = [None, 1, 0]
    sig = np.zeros((N * nTrials, nChannels), dtype=numType)
    trialdefinition = np.zeros((nTrials, 3), dtype=np.intp)
    for ntrial in range(nTrials):
        noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape).astype(numType)
        noise *= noise_decay
        nt1 = ntrial * N
        nt2 = (ntrial + 1) * N
        sig[nt1 : nt2, ::2] = np.tile(carriers[:, even[(-1)**ntrial]] + noise, (nChan2, 1)).T
        sig[nt1 : nt2, 1::2] = np.tile(carriers[:, odd[(-1)**ntrial]] + noise, (nChan2, 1)).T
        trialdefinition[ntrial, :] = np.array([nt1, nt2, t0])

    # Finally allocate `AnalogData` object that makes use of all this
    tfData = AnalogData(data=sig, samplerate=fs, trialdefinition=trialdefinition)

    cfg = get_defaults(freqanalysis)
    cfg.method = "wavelet"
    cfg.wav = "Morlet"
    trlNo = 0
    # cfg.toi = np.unique(np.floor(tfData.time[trlNo]))
    cfg.toi = np.arange(tfData.time[trlNo][0], tfData.time[trlNo][-1] + 1)   
    cfg.output = "pow"
    # cfg.foi = np.arange(0, 501)  
    tfSpec = freqanalysis(cfg, tfData)
    
    plt.figure(); plt.imshow(tfSpec.trials[0][..., 0].squeeze().T, aspect="auto")
    
       
    x = sig[:N, 1]
    carrier = carriers[:, 1]

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, squeeze=True)    
    axes[0].plot(time, carrier)
    axes[0].set_title('Carrier')
    axes[0].set_xlabel('Time [sec]')
    axes[0].set_ylabel('Signal')
    axes[1].plot(time, x)
    axes[1].set_title('Carrier + Noise')
    axes[1].set_xlabel('Time [sec]')
    plt.show()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, squeeze=True)    
    axes[0].plot(time, modulators[:, 0])
    axes[0].set_title('Modulator #1')
    axes[0].set_xlabel('Time [sec]')
    axes[0].set_ylabel('Signal')
    axes[1].plot(time, modulators[:, 1])
    axes[1].set_title('Modulator #2')
    axes[1].set_xlabel('Time [sec]')
    plt.show()
    
    f, t, Zxx = signal.stft(x, fs, nperseg=1000)
    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    sys.exit()
    
    # max ~= amp / 2.1
    
# Ensure peak count in modulator corresponds to detected tf-waveform
# np.where(mod == mod.max())[0].size                                                                                                                                   
# Out[82]: 13

# In [83]: np.where(mod == mod.min())[0].size                                                                                                                                   
# Out[83]: 12

# In [85]: freqIdx, timeIdx = np.where(np.abs(Zxx) >= (np.abs(Zxx).max() - 0.1*np.abs(Zxx).max()))

# # Only max and min frequencies are detected by this criterion
# In [88]: np.unique(freqIdx)                                                                                                                                                   
# Out[88]: array([238, 362])  -> assert np.unique(freqIdx).size == 2

# In [86]: sum(freqIdx == freqIdx.min())                                                                                                                                        
# Out[86]: 13

# In [87]: sum(freqIdx == freqIdx.max())                                                                                                                                        
# Out[87]: 12

# assert np.abs(sum(freqIdx == freqIdx.max()) - np.where(mod == mod.max())[0].size) < 2
# peak-count can be off by one (depending on windowing, initial down-ward slope
# is reflected or not), but not more

    
    foi = np.arange(501)
    wav = Morlet()
    widths = wav.scale_from_period(1/foi[foi > 0])  
    tf = cwt(x, axis=0, wavelet=wav, widths=widths, dt=1/fs)
    plt.figure()
    plt.pcolormesh(time, foi[foi > 0], np.abs(tf))
    plt.title('Wavelet Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
    cut = slice(int(20*fs), int(40*fs + 1))
    tf_cut = cwt(x[cut], axis=0, wavelet=wav, widths=widths, dt=1/fs)
    plt.figure()
    plt.pcolormesh(time[cut], foi[foi > 0], np.abs(tf_cut))
    plt.title('Trimmed Wavelet Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    