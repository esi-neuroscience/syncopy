# -*- coding: utf-8 -*-
#
# SynCoPy spectral estimation methods
#
# Created: 2019-01-22 09:07:47
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-03-15 15:11:06>

# Builtin/3rd party package imports
import sys
import numpy as np
from scipy.signal.windows import hann, dpss
from numpy.lib.format import open_memmap
from tqdm import tqdm
from abc import ABC, abstractmethod
import pdb
from memory_profiler import profile

# Local imports
import syncopy as spy
from syncopy import data_parser
from syncopy import SpectralData
from syncopy.specest.wavelets import cwt, Morlet
from copy import copy

from syncopy import __dask__
if __dask__:
    import dask
    import dask.array as da
    from dask.distributed import get_client


__all__ = ["freqanalysis", "mtmfft", "wavelet", "MultiTaperFFT", "WaveletTransform"]


def freqanalysis(data, method='mtmfft', output='fourier',
                 keeptrials=True, keeptapers=True,
                 pad='nextpow2', polyremoval=0, padtype='zero',
                 taper=hann, tapsmofrq=None,
                 foi=None, toi=None,
                 width=6, outputfile=None, out=None):
    # FIXME: parse remaining input arguments
    if polyremoval:
        raise NotImplementedError("Detrending has not been implemented yet.")

    # Make sure input object can be processed
    try:
        data_parser(data, varname="data", dataclass="AnalogData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc

    # If provided, make sure output object is appropriate
    if out is not None:
        try:
            data_parser(out, varname="out", writable=True,
                        dataclass="SpectralData")
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = SpectralData()
        new_out = True

    methods = {
        "mtmfft": (MultiTaperFFT(taper=hann, tapsmofrq=tapsmofrq,
                                 pad=pad, padtype=padtype,
                                 polyorder=polyremoval),
                   1 / data.samplerate)
    }

    specestMethod = methods[method][0]
    argv = methods[method][1]
    specestMethod.initialize(data, argv)
    specestMethod.compute(data, out)

    # if output == "power":
    #     powerOut = out.copy(deep=T)
    #     for trial in out.trials:
    #         trial = np.absolute(trial)

    return out if newOut else None


def mtmfft(dat, dt,
           taper=hann, taperopt={}, tapsmofrq=None,
           pad="nextpow2", padtype="zero",
           polyorder=None, fftAxis=1,
           noCompute=False):

    # move fft/samples dimension into first place
    dat = np.moveaxis(np.atleast_2d(dat), fftAxis, 1)
    nSamples = dat.shape[1]
    nChannels = dat.shape[0]

    # padding
    if pad:
        padWidth = np.zeros((dat.ndim, 2), dtype=int)
        if pad == "nextpow2":
            padWidth[1, 0] = _nextpow2(nSamples) - nSamples
        else:
            raise NotImplementedError('padding not implemented')

        if padtype == "zero":
            dat = np.pad(dat, pad_width=padWidth,
                         mode="constant", constant_values=0)

        # update number of samples
        nSamples = dat.shape[1]

    if taper == dpss and (not taperopt):
        nTaper = np.int(np.floor(tapsmofrq * nSamples * dt))
        taperopt = {"NW": tapsmofrq, "Kmax": nTaper}
    win = np.atleast_2d(taper(nSamples, **taperopt))

    nFreq = int(np.floor(nSamples / 2) + 1)
    outputShape = (win.shape[0], nChannels, nFreq)
    if noCompute:
        return outputShape

    # taper x chan x freq
    spec = np.zeros((win.shape[0], nChannels, nFreq), dtype=complex)
    for taperIdx, taper in enumerate(win):
        if dat.ndim > 1:
            taper = np.tile(taper, (nChannels, 1))
        spec[taperIdx, ...] = np.fft.rfft(dat * taper, axis=1)

    return spec


class ComputationalRoutine(ABC):

    computeFunction = None
    dtype = None

    def __init__(self, *argv, **kwargs):
        self.defaultCfg = spy.get_defaults(self.computeFunction)
        self.cfg = copy(self.defaultCfg)
        self.cfg.update(**kwargs)
        self.argv = argv
        self.outputShape = None

    def initialize(self, data):
        # FIXME: this only works for data with equal output trial lengths
        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        self.outputShape = self.computeFunction(data.trials[0],
                                                *self.argv,
                                                **dryRunKwargs)

    @profile
    def compute(self, data, out, useDask=False):

        self.preallocate_output(data, out)
        result = None
        if useDask:
            self.compute_dask(data, out)
        else:
            self.compute_sequential(data, out)

        self.handle_metadata(data, out)
        self.write_log(data, out)
        return out

    def write_log(self, data, out):
        # Write log
        out._log = str(data._log) + out._log
        logHead = "computed {name:s} with settings\n".format(name=self.computeFunction.__name__)

        logOpts = ""
        for k, v in self.cfg.items():
            logOpts += "\t{key:s} = {value:s}\n".format(key=k, value=str(v))

        out.log = logHead + logOpts

    @abstractmethod
    def preallocate_output(self):
        pass

    @abstractmethod
    def handle_metadata(self):
        pass

    @abstractmethod
    def compute_dask(self):
        pass

    @abstractmethod
    def compute_sequential(self):
        pass


class MultiTaperFFT(ComputationalRoutine):
    computeFunction = staticmethod(mtmfft)
    dtype = np.complex128

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    def preallocate_output(self, data, out):
        res = open_memmap(out._filename,
                          shape=(len(data.trials),) + self.outputShape,
                          dtype=self.dtype,
                          mode="w+")
        del res

    def handle_metadata(self, data, out):
        out.data = open_memmap(out._filename, mode="r+")

        # We can't simply use ``redefinetrial`` here, prep things by hand
        time = np.arange(len(data.trials))
        time = time.reshape((time.size, 1))
        out.sampleinfo = np.hstack([time, time + 1])
        out.trialinfo = np.array(data.trialinfo)
        out._t0 = np.zeros((len(data.trials),))

        # Attach meta-data
        out.samplerate = data.samplerate
        out.channel = np.array(data.channel)
        out.taper = np.array([self.cfg["taper"].__name__] * self.outputShape[0])
        out.freq = np.linspace(0, 1, self.outputShape[2]) * (data.samplerate / 2)
        out.cfg = self.cfg

    @profile
    def compute_dask(self, data, out):
        print('Dask!')
        # Point to trials on disk by using delayed **static** method calls
        lazy_trial = dask.delayed(data._copy_trial, traverse=False)
        lazy_trls = [lazy_trial(trialno,
                                data._filename,
                                data.dimord,
                                data.sampleinfo,
                                data.hdr)
                     for trialno in range(data.sampleinfo.shape[0])]

        # Construct a distributed dask array block by stacking delayed trials
        trl_block = da.hstack([da.from_delayed(trl, shape=data._shapes[sk],
                                               dtype=data.data.dtype)
                               for sk, trl in enumerate(lazy_trls)])

        # Use `map_blocks` to compute spectra for each trial in the
        # constructed dask array
        specs = trl_block.map_blocks(self.computeFunction,
                                     *self.argv, **self.cfg,
                                     dtype="complex",
                                     chunks=self.outputShape,
                                     new_axis=[0])

        # # Write computed spectra in pre-allocated memmap
        # if out is not None:
        if True:
            daskResult = specs.map_blocks(self.write_block, out._filename,
                                          dtype=self.dtype, drop_axis=[0, 1],
                                          chunks=(1,))
        return daskResult.compute()

    def compute_sequential(self, data, out):
        for tk, trl in enumerate(tqdm(data.trials, desc="Computing MTMFFT...")):
            res = open_memmap(out._filename, mode="r+")[tk, :, :, :]
            res[...] = self.computeFunction(trl, 1 / data.samplerate,
                                            **self.cfg)
            del res
            data.clear()

    @staticmethod
    def write_block(blk, filename, block_info=None):
        idx = block_info[0]["chunk-location"][-1]

        res = open_memmap(filename, mode="r+")[idx, :, :, :]
        res[...] = blk
        res = None
        del res
        return idx


def _nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n


# def mtmfft(dat, dt,
#            taper=hann, taperopt={}, tapsmofrq=None,
#            pad="nextpow2", padtype="zero",
#            polyorder=None, fftAxis=1,
#            noCompute=False):


def wavelet(dat, dt,
            freqoi=None, polyorder=None, wav=Morlet(w0=6), axis=1,
            stepsize=100, noCompute=False):

    dat = np.atleast_2d(dat)
    scales = wav.scale_from_period(np.reciprocal(freqoi))

    if noCompute:
        # time x taper x chan x freq
        return (np.floor(dat.shape[1] / stepsize), 1, dat.shape[0], len(scales))

    # cwt returns (len(data), len(widths)).
    transformed = cwt(dat, axis=axis, wavelet=wav, widths=scales, dt=dt)
    transformed = transformed[:, :, np.newaxis, ...].transpose()

    # time x taper x chan x freq
    return np.complex64(transformed[0:-1:int(stepsize), ...])


class WaveletTransform(ComputationalRoutine):
    computeFunction = staticmethod(wavelet)
    dtype = np.complex64

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    def initialize(self, data):
        minTrialLength = np.array(data._shapes)[:, 1].min()
        if self.cfg["freqoi"] is None:
            self.cfg["freqoi"] = 1 / _get_optimal_wavelet_scales(minTrialLength,
                                                                 1 / data.samplerate,
                                                                 dj=0.25)

        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        self.outputShape = self.computeFunction(data.trials[0], *self.argv, **dryRunKwargs)

    def preallocate_output(self, data, out):
        totalTriallength = np.int(np.sum(np.floor(np.array(data._shapes)[:, 1] /
                                                  self.cfg["stepsize"])))

        result = open_memmap(out._filename,
                             shape=(totalTriallength,) + self.outputShape[1:],
                             dtype=self.dtype,
                             mode="w+")
        # pdb.set_trace()
        del result

    def compute_sequential(self, data, out):
        outIndex = 0
        for idx, trial in enumerate(tqdm(data.trials,
                                         desc="Computing Wavelet spectrum...")):
            tmp = self.computeFunction(trial, *self.argv, **self.cfg)
            selector = slice(outIndex, outIndex + tmp.shape[0])
            res = open_memmap(out._filename, mode="r+")[selector, :, :, :]
            res[...] = tmp
            del res
            data.clear()

    def compute_dask(self, data, out):
        raise NotImplementedError('Dask computation of wavelet transform is not yet implemented')

    def handle_metadata(self, data, out):
        out.data = open_memmap(out._filename, mode="r+")
        # We can't simply use ``redefinetrial`` here, prep things by hand
        out.sampleinfo = np.floor(data.sampleinfo / self.cfg["stepsize"]).astype(np.int)
        out.trialinfo = np.array(data.trialinfo)
        out._t0 = data._t0 / self.cfg["stepsize"]

        # Attach meta-data
        out.samplerate = data.samplerate / self.cfg["stepsize"]
        out.channel = np.array(data.channel)
        out.freq = self.cfg["freqoi"]
        return out


def _get_optimal_wavelet_scales(nSamples, dt, dj=0.25, s0=1):
    """Form a set of scales to use in the wavelet transform.

    For non-orthogonal wavelet analysis, one can use an
    arbitrary set of scales.

    It is convenient to write the scales as fractional powers of
    two:

        s_j = s_0 * 2 ** (j * dj), j = 0, 1, ..., J

        J = (1 / dj) * log2(N * dt / s_0)

    s0 - smallest resolvable scale
    J - largest scale

    choose s0 so that the equivalent Fourier period is 2 * dt.

    The choice of dj depends on the width in spectral space of
    the wavelet function. For the Morlet, dj=0.5 is the largest
    that still adequately samples scale. Smaller dj gives finer
    scale resolution.
    """
    s0 = 2 * dt
    # Largest scale
    J = int((1 / dj) * np.log2(nSamples * dt / s0))
    return s0 * 2 ** (dj * np.arange(0, J + 1))
