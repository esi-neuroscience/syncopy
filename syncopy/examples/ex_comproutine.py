# -*- coding: utf-8 -*-
#
# Exemplary implementation of an algorithmic strategy in Syncopy
#
# Created: 2019-07-02 14:25:52
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-07-04 12:01:00>

# Add SynCoPy package to Python search path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Import SynCoPy
spy_path = os.path.abspath(".." + os.sep + "..")
if spy_path not in sys.path:
    sys.path.insert(0, spy_path)
import syncopy as spy
from syncopy.shared.computational_routine import ComputationalRoutine
if spy.__dask__:
    import dask.distributed as dd


# Dummy function for generating a docstring and function links
def ex_datatype():
    """
    Exemplary implementation of a filtering routine in Syncopy to illustrate
    the concept of :class:`ComputationalRoutine` and its use in practice.

    See also
    --------
    ComputationalRoutine : abstract parent class for implementing algorithmic strategies in Syncopy
    """


if __name__ == "__main__":

    # Set number of channels and trial-count
    nChannels = 32
    nTrials = 8

    # Frequency of data and (additively imposed) noise
    fData = 2
    fNoise = 64

    # Sample frequency and base interval of signal
    fs = 1000
    t = np.linspace(-1, 1, fs)

    # The "actual" signal to be reconstructed is a sine wave of frequency `fData`
    orig = np.sin(2 * np.pi * fData * t)
    sig = orig + np.sin(2 * np.pi * fNoise * t)

    # Construct a Butterworth low-pass filter with 50 Hz cutoff frequency
    cutoff = 50
    b, a = signal.butter(8, 2 * cutoff / fs)

    # Low-pass filter the signal using the constructed filter
    filtered = signal.filtfilt(b, a, sig, padlen=200)

    # Perform visual inspection of result
    msg = "Max. absolute error source - filtered: {}"
    plt.ion()
    plt.figure(figsize=[9, 6.4])
    ax = plt.gca()
    ax.plot(t, sig, "b", label="noisy signal")
    ax.plot(t, orig, "r", label="clean source")
    ax.plot(t, filtered, "k--", label="filtered signal")
    ax.legend(loc="best")
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.annotate(msg.format(np.abs(filtered - orig).max()),
                (t[0], -2.1), xycoords="data")

    # Duplicate and inflate signal to artificially generate "channels" and "trials"
    # (the signal is concatendated to itself `nTrial` times to simulate trials
    # and copied `nChannel` times to mimic channels)
    sig = np.repeat(sig.reshape(-1, 1), axis=1, repeats=nChannels)
    sig = np.tile(sig, (nTrials, 1))

    # Construct artificial trial-definition array (each trial has the same length
    # as the original signal)
    trl = np.zeros((nTrials, 3), dtype="int")
    for ntrial in range(nTrials):
        trl[ntrial, :] = np.array([ntrial * fs, (ntrial + 1) * fs, 0])

    # Use the generated arrays to create an artificial `AnalogData object`
    data = spy.AnalogData(data=sig, samplerate=fs, trialdefinition=trl,
                          dimord=["time", "channel"])

    # The `computeFunction` that performs the actual filtering
    def lowpass(arr, b, a, noCompute=None, chunkShape=None):
        if noCompute:
            return arr.shape, arr.dtype
        res = signal.filtfilt(b, a, arr.T, padlen=200).T
        return res

    # A subclass of `ComputationalRoutine` that binds `lowpass` (our `computeFunction`)
    # as static method and provides a class method for processing meta-information
    # for the result of the filtering
    class LowPassFilter(ComputationalRoutine):
        computeFunction = staticmethod(lowpass)

        def process_metadata(self, data, out):
            if self.keeptrials:
                out.sampleinfo = np.array(data.sampleinfo)
                out.trialinfo = np.array(data.trialinfo)
                out._t0 = np.zeros((len(data.trials),))
            else:
                trl = np.array([[0, data.sampleinfo[0, 1], 0]])
                out.sampleinfo = trl[:, :2]
                out._t0 = trl[:, 2]
                out.trialinfo = trl[:, 3:]
            out.samplerate = data.samplerate
            out.channel = np.array(data.channel)

    # Allocate an empty object for storing the result
    out = spy.AnalogData()

    # Instantiate our new filtering class (allocates class attributes)
    myfilter = LowPassFilter(b, a)

    # Initialize computation with input data (perform pre-allocation dry-run)
    myfilter.initialize(data)

    # Perform the actual filtering, save the result in `out`
    myfilter.compute(data, out)

    # To verify the result, we apply the same inflation transform (channel-wise
    # duplication/by-trial concatenation) to the clean source
    orig = np.repeat(orig.reshape(-1, 1), axis=1, repeats=nChannels)
    orig = np.tile(orig, (nTrials, 1))

    # Review reconstruction error
    print(msg.format(np.abs(out.data - orig).max()))

    # If available, perform the same computation in parallel (note that
    # re-instantiation is not necessary as long as `b` and `a` remain unchanged)
    if spy.__dask__:
        client = dd.Client()
        out_parallel = spy.AnalogData()
        myfilter.initialize(data)
        myfilter.compute(data, out_parallel, parallel=True)
        print(msg.format(np.abs(out_parallel.data - orig).max()))

    # Filter `data` again, this time average result across trials (re-instantiation
    # is necessary to propagate `keeptrials` keyword)
    out = spy.AnalogData()
    myfilter = LowPassFilter(b, a, keeptrials=False)
    myfilter.initialize(data)
    myfilter.compute(data, out)
    print(msg.format(np.abs(out.data - orig[:t.size, :]).max()))

    # If available, perform the same computation in parallel (note that
    # re-instantiation is not necessary as long as `b`, `a` and `keeptrials`
    # remain unchanged)
    if spy.__dask__:
        client = dd.Client()
        out_parallel = spy.AnalogData()
        myfilter.initialize(data)
        myfilter.compute(data, out_parallel, parallel=True)
        print(msg.format(np.abs(out_parallel.data - orig[:t.size, :]).max()))
