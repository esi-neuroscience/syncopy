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

    fs = 1e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 500*np.cos(2*np.pi*0.0625*time)
    # mod = 500*np.cos(2*np.pi*0.125*time)
    carrier = amp * np.sin(2*np.pi*3e2*time + mod)
    noise = np.random.normal(scale=np.sqrt(noise_power),
                            size=time.shape)
    noise *= np.exp(-time/5)
    x = carrier + noise    
    
    # Trials: stitch together [x, x, x]
    # channels: 
    # 1, 3, 5, 7: mod -> 0.0625 * time
    # 0, 2, 4, 6: mod -> 0.125 * time

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, squeeze=True)    
    axes[0].plot(time, carrier)
    axes[0].set_title('Carrier')
    axes[0].set_xlabel('Time [sec]')
    axes[0].set_ylabel('Signal')
    axes[1].plot(time, x)
    axes[1].set_title('Carrier + Noise')
    axes[1].set_xlabel('Time [sec]')
    plt.show()
    
    plt.figure()
    plt.plot(time, mod)
    plt.xlabel('Time [sec]')
    plt.title('Modulator')
    
    f, t, Zxx = signal.stft(x, fs, nperseg=1000)
    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
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

    
    sys.exit()
    
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
    