# -*- coding: utf-8 -*-
#
# SynCoPy spectral estimation methods
#
# Created: 2019-01-22 09:07:47
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-03-13 16:32:34>

# Builtin/3rd party package imports
import sys
import numpy as np
from scipy.signal.windows import hann, dpss
from numpy.lib.format import open_memmap
from tqdm import tqdm
from abc import ABC, abstractmethod
import pdb

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


__all__ = ["freqanalysis", "mtmfft", "wavelet", "MultiTaperFFT"]


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
        "mtmfft": MultiTaperFFT(taper=hann, tapsmofrq=tapsmofrq,
                                pad=pad, padtype=padtype,
                                polyorder=polyremoval)
    }

    specestMethod = methods[method]
    specestMethod.initialize(data)
    specestMethod.calculate(data, out)

    if output == "power":
        powerOut = out.copy(deep=T)
        for trial in out.trials:
            trial = np.absolute(trial)

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


class MultiTaperFFT():
    computeFunction = staticmethod(mtmfft)
    dtype = np.complex128

    def __init__(self, **kwargs):
        self.defaultCfg = spy.get_defaults(self.computeFunction)
        self.cfg = copy(self.defaultCfg)
        self.cfg.update(**kwargs)

        # self.outputfile = None

    def initialize(self, data):
        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        self.outputShape = self.computeFunction(data.trials[0],
                                                1 / data.samplerate,
                                                **dryRunKwargs)

    def calculate(self, data, out):
        res = open_memmap(out._filename,
                          shape=(len(data.trials),) + self.outputShape,
                          dtype=self.dtype,
                          mode="w+")
        del res

        useDask = True
        result = None
        if useDask:

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
                                         1 / data.samplerate,
                                         **self.cfg,
                                         dtype="complex",
                                         chunks=self.outputShape,
                                         new_axis=[0])

            # # Write computed spectra in pre-allocated memmap
            if out is not None:
                daskResult = specs.map_blocks(write_block, out._filename,
                                              dtype=self.dtype, drop_axis=[0, 1],
                                              chunks=(1,))

        else:
            for tk, trl in enumerate(tqdm(data.trials, desc="Computing MTMFFT...")):
                res = open_memmap(out._filename, mode="r+")[tk, :, :, :]
                res[...] = self.computeFunction(trl, 1 / data.samplerate,
                                                **self.cfg)
                del res
                data.clear()

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

        # return out
        # # Write log
        # out._log = str(obj._log) + out._log
        # log = "computed multi-taper FFT with settings\n" +\
        #       "\ttaper = {tpr:s}\n" +\
        #       "\tpadding = {pad:s}\n" +\
        #       "\tpadtype = {pat:s}\n" +\
        #       "\tpolyorder = {pol:s}\n" +\
        #       "\ttaperopt = {topt:s}\n" +\
        #       "\ttapsmofrq = {tfr:s}\n"
        # out.log = log.format(tpr=cfg["taper"],
        #                      pad=cfg["padding"],
        #                      pat=cfg["padtype"],
        #                      pol=str(cfg["polyorder"]),
        #                      topt=str(cfg["taperopt"]),
        #                      tfr=str(cfg["tapsmofrq"]))


def write_block(blk, filename, block_info=None):
    # print(block_info)
    idx = block_info[0]["chunk-location"][-1]
    print(blk.shape)
    print(idx)
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


def wavelet(data=None, pad=None, padtype=None, freqoi=None, timeoi=None,
            width=6, polyorder=None, out=None, wavelet=Morlet):
    if out is None:
        out = spy.SpectralData()
    wav = Morlet(w0=width)

    minTrialLength = np.array(data._shapes)[:, 1].min()
    totalTriallength = np.sum(np.array(data._shapes)[:, 1])

    if freqoi is None:
        scales = _get_optimal_wavelet_scales(minTrialLength,
                                             1 / data.samplerate,
                                             dj=0.25)
    else:
        scales = wav.scale_from_period(np.reciprocal(freqoi))

    result = open_memmap(out._filename,
                         shape=(totalTriallength, 1, len(data.channel), len(scales)),
                         dtype="complex",
                         mode="w+")
    del result

    for idx, trial in enumerate(tqdm(data.trials, desc="Computing Wavelet spectrum...")):
        selector = slice(data.sampleinfo[idx, 0], data.sampleinfo[idx, 1])
        res = open_memmap(out._filename, mode="r+")[selector, :, :, :]
        tmp = _wavelet_compute(dat=trial,
                               wavelet=wav,
                               dt=1 / data.samplerate,
                               scales=scales,
                               axis=1)[:, :, np.newaxis, ...].transpose()
        res[...] = tmp
        # del res
        # data.clear()

    out._data = open_memmap(out._filename, mode="r+")
    # We can't simply use ``redefinetrial`` here, prep things by hand
    time = np.arange(len(data.trials))
    time = time.reshape((time.size, 1))
    out.sampleinfo = data.sampleinfo
    out.trialinfo = np.array(data.trialinfo)
    out._t0 = np.zeros((len(data.trials),))

    # Attach meta-data
    out.samplerate = data.samplerate
    out.channel = np.array(data.channel)
    out.freq = np.reciprocal(wav.fourier_period(scales))

    return out


def _wavelet_compute(dat, wavelet, dt, scales, axis=1):
    # def compute_wavelet
    return cwt(dat, axis=axis, wavelet=wavelet, widths=scales, dt=dt)


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
