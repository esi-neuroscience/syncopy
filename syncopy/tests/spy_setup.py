# -*- coding: utf-8 -*-
#
# Simple script for testing Syncopy w/o pip-installing it
#

# Builtin/3rd party package imports
import numpy as np

# Add SynCoPy package to Python search path
import os
import sys
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)

# Import package
import syncopy as spy

# Import artificial data generator
from syncopy.tests.misc import generate_artificial_data

# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    # Construct simple trigonometric signal to check FFT consistency: each
    # channel is a sine wave of frequency `freqs[nchan]` with single unique
    # amplitude `amp` and sampling frequency `fs`
    nChannels = 32
    nTrials = 8
    fs = 1024
    fband = np.linspace(1, fs/2, int(np.floor(fs/2)))
    freqs = [88.,  35., 278., 104., 405., 314., 271., 441., 343., 374., 428.,
             367., 75., 118., 289., 310., 510., 102., 123., 417., 273., 449.,
             416.,  32., 438., 111., 140., 304., 327., 494.,  23., 493.]
    freqs = freqs[:nChannels]
    # freqs = np.random.choice(fband[:-2], size=nChannels, replace=False)
    amp = np.pi
    phases = np.random.permutation(np.linspace(0, 2 * np.pi, nChannels))
    t = np.linspace(0, nTrials, nTrials * fs)
    sig = np.zeros((t.size, nChannels), dtype="float32")
    for nchan in range(nChannels):
        sig[:, nchan] = amp * \
            np.sin(2 * np.pi * freqs[nchan] * t + phases[nchan])

    trialdefinition = np.zeros((nTrials, 3), dtype="int")
    for ntrial in range(nTrials):
        trialdefinition[ntrial, :] = np.array(
            [ntrial * fs, (ntrial + 1) * fs, 0])

    adata = spy.AnalogData(data=sig, samplerate=fs,
                       trialdefinition=trialdefinition)

    spec = spy.freqanalysis(adata, method="mtmfft", taper="hann", output="fourier", parallel=True)
