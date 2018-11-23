# fft_slalom.py - Torture-test sFFT vs. NumPy's FFT implementation
# 
# Author: Stefan Fuertinger [stefan@fuertinger.science]
# Created: November  8 2018
# Last modified: <2018-11-09 17:56:02>

# Account for Python 2.x
from __future__ import division, print_function
import pyfftw
import timeit

# Set number of repitiions and no. of loops to run + output message template
reps = 5
nmbr = int(1e2)
msg = "{nmbr:d} loops, best of {reps:d}: {t:3.2f} sec per loop"

# The setup part common to all profiling runs: setup code is executed only once and not timed
setup = """
import os
import sys
import numpy as np

# Set size of signal and sparsity level (taken from sFFT v.2 manual)
n = 16777216
k = 50

# To get comparable results, fix random number generator seed
rnd = np.random.RandomState(13082010)

# Set up a sparse dummy signal
freq = rnd.randint(0, n - 1, k)
sigl = np.zeros((n,), dtype=np.complex)
sigl[freq] = 1.0
"""

# sFFT-specific setup
setup_sfft = setup + """
# Include "python" directory in search path, so we don't have to pip install sfft
sys.path.insert(0, os.path.abspath("python" + os.sep))
import sfft.sfft as sfft

# Instance and configure sFFT class
myfft = sfft.sfft(n, k, 2)
"""

# Time sFFT
timings = timeit.repeat(setup=setup_sfft,
                        stmt="myfft.execute(sigl)",
                        repeat=reps, number=nmbr)
print("sFFT: " + msg.format(nmbr=nmbr, reps=reps, t=min(timings)))

# pyFFTW-specific setup
setup_pyfftw = setup + """
import pyfftw

# Accessing `pyfftw.FFTW` is probably a little faster due to a-priori memory
# allocation, but pyfftw's `builders` package should be better comparable to
# the way the sFFT Python wrapper is doing things
fft_object = pyfftw.builders.fft(sigl, planner_effort='FFTW_ESTIMATE')
"""

# Now pyFFTW
timings = timeit.repeat(setup=setup_pyfftw,
                        stmt="fft_object()",
                        repeat=reps, number=nmbr)
print("pyFFTW: " + msg.format(nmbr=nmbr, reps=reps, t=min(timings)))

# Finally, NumPy's turn
timings = timeit.repeat(setup=setup,
                        stmt="np.fft.fft(sigl)",
                        repeat=reps, number=nmbr)
print("NumPy: " + msg.format(nmbr=nmbr, reps=reps, t=min(timings)))
