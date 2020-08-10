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
    mod = 500*np.cos(2*np.pi*0.125*time)
    carrier = amp * np.sin(2*np.pi*3e2*time + mod)
    noise = np.random.normal(scale=np.sqrt(noise_power),
                            size=time.shape)
    noise *= np.exp(-time/5)
    x = carrier + noise    
    
    f, t, Zxx = signal.stft(x, fs, nperseg=1000)
    plt.figure()
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
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
    