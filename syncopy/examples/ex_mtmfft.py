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
    
# In [11]: widths[:10]                                                                                                                                                               
# Out[11]: 
# array([0.968 , 0.484 , 0.3227, 0.242 , 0.1936, 0.1613, 0.1383, 0.121 , 0.1076,
#        0.0968])

# In [12]: wav.fourier_period(widths[:10])                                                                                                                                           
# Out[12]: 
# array([1.    , 0.5   , 0.3333, 0.25  , 0.2   , 0.1667, 0.1429, 0.125 , 0.1111,
#        0.1   ])

# In [13]: foi[1:11]                                                                                                                                                                 
# Out[13]: array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

# In [14]: 1/foi[1:11]                                                                                                                                                               
# Out[14]: 
# array([1.    , 0.5   , 0.3333, 0.25  , 0.2   , 0.1667, 0.1429, 0.125 , 0.1111,
#        0.1   ])    
# In [15]: 1/wav.fourier_period(widths[:10])                                                                                                                                         
# Out[15]: array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
    