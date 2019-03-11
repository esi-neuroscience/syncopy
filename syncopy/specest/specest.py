# -*- coding: utf-8 -*-
#
# SynCoPy spectral estimation methods
#
# Created: 2019-01-22 09:07:47
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-03-11 16:39:29>

# Builtin/3rd party package imports
import sys
import numpy as np
import scipy.signal.windows as windows
from numpy.lib.format import open_memmap
from tqdm import tqdm
from abc import ABC, abstractmethod

# Local imports
import syncopy as spy
from syncopy import spy_data_parser
from syncopy import SpectralData
from syncopy.specest.wavelets import cwt, Morlet
from copy import copy

from syncopy import __dask__
if __dask__:
    import dask
    import dask.array as da
    from dask.distributed import get_client


__all__ = ["freqanalysis", "mtmfft", "wavelet"]


def freqanalysis(data, method='mtmfft', output='fourier',
                 keeptrials=True, keeptapers=True,
                 pad='nextpow2', polyremoval=0, padtype='zero',
                 taper=windows.hann, tapsmofrq=None,
                 foi=None, toi=None,
                 width=6, outputfile=None, delayed=False, out=None):
    # FIXME: parse remaining input arguments
    if polyremoval:
        raise NotImplementedError("Detrending has not been implemented yet.")

    # Make sure input object can be processed
    try:
        spy_data_parser(data, varname="data", dataclass="AnalogData",
                        writable=None, empty=False)
    except Exception as exc:
        raise exc

    # If provided, make sure output object is appropriate
    if out is not None:
        try:
            spy_data_parser(out, varname="out", writable=True,
                            dataclass="SpectralData")
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = SpectralData()
        new_out = True

    methods = {
        "mtmfft": MultiTaperFFT(pad=pad, polyorder=polyremoval,
                                padtype=padtype, taper=taper,
                                tapsmofrq=tapsmofrq)
    }

    specestMethod = methods[method]

    specestMethod.initialize(data)
    result = specestMethod.compute(data)
    return result


class SpecestMethod(ABC):

    computeFunction = None

    outputDType = {
        "fourier", np.complex128,
        "pow", np.float64,
    }

    def __init__(self, **kwargs):
        self.cfg = kwargs
        self.output = None

    def __call__(self, data):
        self.initialize(data)
        self.allocate_output()
        self.compute(data)
        self.finalize(data)

    def allocate_output(self):
        if self.outputfile:
            # Allocate memory map for results
            res = open_memmap(self.outputfile,
                              shape=self.outputshape,
                              dtype=self.outputDType[self.method],
                              mode="w+")
            del res

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def compute():
        pass

    # @abstractmethod
    # def finalize():
    #     pass

    @staticmethod
    def write_block(blk, resname, block_info=None):
        idx = block_info[0]["chunk-location"][-1]
        res = open_memmap(resname, mode="r+")[idx, :, :, :]
        res[...] = blk
        res = None
        del res
        return idx


def mtmfft(dat, win,
           tapsmofrq=None, dimord=None, freqoi=None, pad=None, padtype=None,
           polyorder=None, fftAxis=1, outputfile=None):

    # move fft/samples dimension into first place
    dat = np.moveaxis(np.atleast_2d(dat), fftAxis, 1)
    nSamples = dat.shape[1]
    nChannels = dat.shape[0]

    # # padding
    # if pad:
    #     padWidth = np.zeros((dat.ndim, 2), dtype=int)
    #     if pad == "nextpow2":
    #         padWidth[1, 0] = _nextpow2(nSamples) - nSamples
    #     else:
    #         raise NotImplementedError('padding not implemented')

    #     if padtype == "zero":
    #         dat = np.pad(dat, pad_width=padWidth,
    #                      mode="constant", constant_values=0)

    #     # update number of samples
    #     nSamples = dat.shape[1]

    nFreq = int(np.floor(nSamples / 2) + 1)
    # taper x chan x freq
    spec = np.zeros((win.shape[0], nChannels, nFreq), dtype=complex)
    for taperIdx, taper in enumerate(win):
        if dat.ndim > 1:
            taper = np.tile(taper, (nChannels, 1))
        spec[taperIdx, ...] = np.fft.rfft(dat * taper, axis=1)

    return spec


class MultiTaperFFT():
    computeFunction = mtmfft
    arguments = None

    defaultCfg = spy.spy_get_defaults(computeFunction)
    # argv = None * 2

    def __init__(self, **kwargs):
        self.__dict__ = copy(self.defaultCfg)
        self.__dict__.update(**kwargs)
        self._taperopt = None
        # self.outputfile = None

    def initialize(self, data):

        if self.pad == "nextpow2":
            nSamples = _nextpow2(data._shapes[0][self.fftAxis])
        else:
            raise NotImplementedError("Coming soon...")

        if self.taper == windows.dpss and (not self._taperopt):
            self._nTaper = np.int(np.floor(self.tapsmofrq * nSamples / data.samplerate))
            self._taperopt = {"NW": self.tapsmofrq, "Kmax": self._nTaper}

        # Compute taper in shape nTaper x nSamples and determine size of freq. axis
        self._win = np.atleast_2d(self.taper(nSamples)) #, **self._taperopt))
        self._nFreq = int(np.floor(nSamples / 2) + 1)
        self._outputShape = (len(data.trials),
                             self._win.shape[0],
                             data._shapes[0][0],
                             self._nFreq)

    def compute(self, data):
        # self.computeFunction(data, )
        useDask = True
        specs = None
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
                                         self._win,
                                         dtype="complex",
                                         chunks=(self._win.shape[0],
                                                 data.data.shape[0],
                                                 self._nFreq),
                                         new_axis=[0])

            # # Write computed spectra in pre-allocated memmap
            # TODO: Maybe do freqoi and timeoi selection here?
            # result = specs.map_blocks(self.write_block, self._filename,
            #                           dtype="complex",
            #                           chunks=(1,),
            #                           drop_axis=[0, 1])

        return specs
        # # Perform actual computation
        # result.compute()

    @staticmethod
    def write_block(blk, resname, block_info=None):
        idx = block_info[0]["chunk-location"][-1]
        res = open_memmap(resname, mode="r+")[idx, :, :, :]
        res[...] = blk
        res = None
        del res
        return idx


# def mtmfft_old(obj, taper=windows.hann, pad="nextpow2", padtype="zero",
#                polyorder=None, taperopt={}, fftAxis=1, tapsmofrq=None, out=None):

#     # FIXME: parse remaining input arguments
#     if polyorder:
#         raise NotImplementedError("Detrending has not been implemented yet.")

#     # Make sure input object can be processed
#     try:
#         spy_data_parser(obj, varname="obj", dataclass="AnalogData",
#                         writable=None, empty=False)
#     except Exception as exc:
#         raise exc

#     # If provided, make sure output object is appropriate
#     if out is not None:
#         try:
#             spy_data_parser(out, varname="out", writable=True,
#                             dataclass="SpectralData")
#         except Exception as exc:
#             raise exc
#         new_out = False
#     else:
#         out = SpectralData()
#         new_out = True

#     # Set parameters applying to all trials: FIXME: make sure trials
#     # are consistent, i.e., padding results in same no. of freqs across all trials
#     fftAxis = obj.dimord.index("time")
#     if pad == "nextpow2":
#         nSamples = _nextpow2(obj._shapes[0][fftAxis])
#     else:
#         raise NotImplementedError("Coming soon...")
#     if taper == windows.dpss and (not taperopt):
#         nTaper = np.int(np.floor(tapsmofrq * nSamples / obj.samplerate))
#         taperopt = {"NW": tapsmofrq, "Kmax": nTaper}

#     # Compute taper in shape nTaper x nSamples and determine size of freq. axis
#     win = np.atleast_2d(taper(nSamples, **taperopt))
#     nFreq = int(np.floor(nSamples / 2) + 1)
#     freq = np.arange(0, np.floor(nSamples / 2) + 1) * obj.samplerate / nSamples

#     # Allocate memory map for results
#     res = open_memmap(out._filename,
#                       shape=(len(obj.trials), win.shape[0], obj._shapes[0][0], nFreq),
#                       dtype="complex",
#                       mode="w+")
#     del res

#     # See if a dask client is running
#     try:
#         use_dask = bool(get_client())
#     except:
#         use_dask = False

#     # Perform parallel computation
#     if use_dask:

#         # Point to trials on disk by using delayed **static** method calls
#         lazy_trial = dask.delayed(obj._copy_trial, traverse=False)
#         lazy_trls = [lazy_trial(trialno,
#                                 obj._filename,
#                                 obj.dimord,
#                                 obj.sampleinfo,
#                                 obj.hdr)
#                      for trialno in range(obj.sampleinfo.shape[0])]

#         # Construct a distributed dask array block by stacking delayed trials
#         trl_block = da.hstack([da.from_delayed(trl,
#                                                shape=obj._shapes[sk],
#                                                dtype=obj.data.dtype) for sk, trl in enumerate(lazy_trls)])

#         # Use `map_blocks` to compute spectra for each trial in the constructred dask array
#         specs = trl_block.map_blocks(_mtmfft_bytrl, win, nFreq, pad, padtype, fftAxis, use_dask,
#                                      dtype="complex",
#                                      chunks=(win.shape[0], obj.data.shape[0], nFreq),
#                                      new_axis=[0])

#         # Write computed spectra in pre-allocated memmap
#         result = specs.map_blocks(_mtmfft_writer, out._filename,
#                                   dtype="complex",
#                                   chunks=(1,),
#                                   drop_axis=[0, 1])

#         # Perform actual computation
#         result.compute()

#     # Serial calculation solely relying on NumPy
#     else:
#         for tk, trl in enumerate(tqdm(obj.trials, desc="Computing MTMFFT...")):
#             res = open_memmap(out._filename, mode="r+")[tk, :, :, :]
#             res[...] = _mtmfft_bytrl(trl, win, nFreq, pad, padtype, fftAxis, use_dask)
#             del res
#             obj.clear()

#     # First things first: attach data to output object
#     out._data = open_memmap(out._filename, mode="r+")

#     # We can't simply use ``redefinetrial`` here, prep things by hand
#     time = np.arange(len(obj.trials))
#     time = time.reshape((time.size, 1))
#     out.sampleinfo = np.hstack([time, time + 1])
#     out.trialinfo = np.array(obj.trialinfo)
#     out._t0 = np.zeros((len(obj.trials),))

#     # Attach meta-data
#     out.samplerate = obj.samplerate
#     out.channel = np.array(obj.channel)
#     out.taper = np.array([taper.__name__] * win.shape[0])
#     out.freq = freq
#     cfg = {"method": sys._getframe().f_code.co_name,
#            "taper": taper.__name__,
#            "padding": pad,
#            "padtype": padtype,
#            "polyorder": polyorder,
#            "taperopt": taperopt,
#            "tapsmofrq": tapsmofrq}
#     out.cfg = cfg
#     out.cfg = dict(obj.cfg)

#     # Write log
#     out._log = str(obj._log) + out._log
#     log = "computed multi-taper FFT with settings\n" +\
#           "\ttaper = {tpr:s}\n" +\
#           "\tpadding = {pad:s}\n" +\
#           "\tpadtype = {pat:s}\n" +\
#           "\tpolyorder = {pol:s}\n" +\
#           "\ttaperopt = {topt:s}\n" +\
#           "\ttapsmofrq = {tfr:s}\n"
#     out.log = log.format(tpr=cfg["taper"],
#                          pad=cfg["padding"],
#                          pat=cfg["padtype"],
#                          pol=str(cfg["polyorder"]),
#                          topt=str(cfg["taperopt"]),
#                          tfr=str(cfg["tapsmofrq"]))

#     # Happy breakdown
#     return out if new_out else None


# def _mtmfft_writer(blk, resname, block_info=None):
#     """
#     Pumps computed spectra into target memmap
#     """
#     idx = block_info[0]["chunk-location"][-1]
#     res = open_memmap(resname, mode="r+")[idx, :, :, :]
#     res[...] = blk
#     res = None
#     del res
#     return idx


# def _mtmfft_bytrl(trl, win, nFreq, pad, padtype, fftAxis, use_dask):
#     """
#     Performs the actual heavy-lifting
#     """

#     # move fft/samples dimension into first place
#     trl = np.moveaxis(np.atleast_2d(trl), fftAxis, 1)
#     nSamples = trl.shape[1]
#     nChannels = trl.shape[0]

#     # padding
#     if pad:
#         padWidth = np.zeros((trl.ndim, 2), dtype=int)
#         if pad == "nextpow2":
#             padWidth[1, 0] = _nextpow2(nSamples) - nSamples
#         else:
#             padWidth[1, 0] = np.ceil((pad - T) / dt).astype(int)
#         if padtype == "zero":
#             trl = np.pad(trl, pad_width=padWidth,
#                          mode="constant", constant_values=0)

#         # update number of samples
#         nSamples = trl.shape[1]

#     # Decide whether to further parallelize or plow through entire chunk
#     if use_dask and trl.size * trl.dtype.itemsize * 1024**(-2) > 1000:
#         spex = []
#         for tap in win:
#             if trl.ndim > 1:
#                 tap = np.tile(tap, (nChannels, 1))
#             prod = da.from_array(trl * tap, chunks=(1, trl.shape[1]))
#             spex.append(da.fft.rfft(prod))
#         spec = da.stack(spex)
#     else:
#         # taper x chan x freq
#         spec = np.zeros((win.shape[0],) + (nChannels,) + (nFreq,), dtype=complex)
#         for wIdx, tap in enumerate(win):
#             if trl.ndim > 1:
#                 tap = np.tile(tap, (nChannels, 1))
#             spec[wIdx, ...] = np.fft.rfft(trl * tap, axis=1)

#     return spec


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
