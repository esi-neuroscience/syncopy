from syncopy.specest import mtmfft
import numpy as np
import matplotlib.pyplot as ppl
from scipy.signal import windows

fs = 1000
# superposition 40Hz and 100Hz oscillations A1:A2 for 1s
f1, f2 = 40, 100
A1, A2 = 5, 3
tvec = np.arange(0, 2, 1 / 1000)

signal = A1 * np.cos(2 * np.pi * f1 * tvec)
signal += A2 * np.cos(2 * np.pi * f2 * tvec)

# the transforms have shape (nTaper, nFreq, nChannel)    
ftr, freqs = mtmfft.mtmfft(signal, fs, taper=None)

# average over potential tapers (only 1 here)
spec = np.real(ftr * ftr.conj()).mean(axis=0)
amplitudes = np.sqrt(spec)[:, 0] # only 1 channel

N = len(tvec)
minBw = 2 * fs / N
Bw = 10
W = Bw / 2 # Hz
NW = W * N / fs
Kmax = int(2 * NW - 1)

taperopt = {'Kmax' : Kmax, 'NW' : NW}
ftr, freqs = mtmfft.mtmfft(signal, fs, taper="dpss", taperopt=taperopt)
# average over tapers 
dpss_spec = np.real(ftr * ftr.conj()).mean(axis=0)
dpss_amplitudes = np.sqrt(dpss_spec)[:, 0] # only 1 channel
# check for amplitudes (and taper normalisation)


fig, ax = ppl.subplots()
ax.set_title(f"Amplitude spectrum {A1} x {f1}Hz + {A2} x {f2}Hz, {Bw}Hz smoothing")
ax.plot(freqs[:250], amplitudes[:250], label="No taper", lw=2)
ax.set_xlabel('frequency (Hz)')
ax.set_ylabel('amplitude (a.u.)')
    
ax.plot(freqs[:250], dpss_amplitudes[:250], label="Slepian", lw=2)
ax.legend()
