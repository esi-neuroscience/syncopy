# -*- coding: utf-8 -*-
# 
# SyNCoPy spectral estimation methods
# 
# Created: 2019-01-22 09:07:47
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-16 16:48:18>

# Builtin/3rd party package imports
import sys
import numpy as np
from scipy.signal.windows import hann, dpss
from tqdm import tqdm
import h5py
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)
from copy import copy

# Local imports
from syncopy.shared import data_parser, scalar_parser, array_parser, get_defaults
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.datatype import SpectralData
from syncopy.specest.wavelets import cwt, Morlet
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.shared.parsers import unwrap_cfg
from syncopy import __dask__
if __dask__:
    import dask
    import dask.array as da

# Module-wide output specs
spectralDTypes = {"pow": np.float32,
                  "fourier": np.complex128,
                  "abs": np.float32}
spectralConversions = {"pow": lambda x: np.float32(x * np.conj(x)),
                       "fourier": lambda x: np.complex128(x),
                       "abs": lambda x: np.float32(np.absolute(x))}

__all__ = ["freqanalysis"]


@unwrap_cfg
def freqanalysis(data, method='mtmfft', output='fourier',
                 keeptrials=True, keeptapers=True,
                 pad='nextpow2', polyremoval=0, padtype='zero',
                 taper=hann, taperopt={}, tapsmofrq=None,
                 foi=None, toi=None,
                 width=6, out=None, cfg=None):
    """
    Explain taperopt...
    """
    
    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", dataclass="AnalogData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc

    # Get everything in local namespace
    lcls = locals()

    # Ensure a valid computational method was selected
    avail_methods = ["mtmfft", "wavelet"]
    if method not in avail_methods:
        raise SPYValueError(legal="".join(opt + ", " for opt in avail_methods)[:-2],
                            varname="method", actual=method)

    # Ensure a valid output format was selected
    options = ["pow", "fourier", "abs"]
    if output not in options:
        raise SPYValueError(legal="".join(opt + ", " for opt in options)[:-2],
                            varname="output", actual=output)

    # Parse all Boolean keyword arguments
    for vname in ["keeptrials", "keeptapers"]:
        if not isinstance(lcls[vname], bool):
            raise SPYTypeError(lcls[vname], varname=vname, expected="Bool")

    # Ensure padding selection makes sense
    options = [None, "nextpow2"]
    if pad not in options:
        raise SPYValueError(legal="".join(opt + ", " for opt in options)[:-2],
                            varname="pad", actual=pad)
    options = ["zero"]
    if padtype not in options:
        raise SPYValueError(legal="".join(opt + ", " for opt in options)[:-2],
                            varname="padtype", actual=padtype)

    # Ensure frequency-of-interest is below Nyquist
    if foi is not None:
        try:
            array_parser(foi, varname="foi", hasinf=False, hasnan=False,
                         lims=[0.1, data.samplerate/2], dims=(None,))
        except Exception as exc:
            raise exc

    # Ensure time-of-interest is within data bounds
    if toi is not None:
        timing = np.array([[-data.t0[tk]/data.samplerate,
                            (end - start - data.t0[tk])/data.samplerate] \
                           for tk, (start, end) in enumerate(data.sampleinfo)])
        try:
            array_parser(toi, varname="toi", hasinf=False, hasnan=False,
                         lims=[timing.min(), timing.max()], dims=(None,))
        except Exception as exc:
            raise exc
    
    # FIXME: implement detrending
    if polyremoval:
        raise NotImplementedError("Detrending has not been implemented yet.")

    # Ensure consistency of method-specific options
    defaults = get_defaults(freqanalysis)
    if method == "mtmfft":
        options = [hann, dpss]
        if taper not in options:
            raise SPYValueError(legal="".join(opt + ", " for opt in options)[:-2],
                                varname="taper", actual=taper)
        if not isinstance(taperopt, dict):
            raise SPYTypeError(taperopt, varname="taperopt", expected="dictionary")
        try:
            scalar_parse(tapsmofrq, varname="tapsmofrq", lims=[0.1, np.inf])
        except Exception as exc:
            raise exc

        # if width != defaults["width"]:
        #     wrng = "<freqanalysis> WARNING: `width` keyword ignored by mtmfft method"
        #     print(wrng)
        
    elif method == "wavelet":
        try:
            scalar_parse(width, varname="width", lims=[1, np.inf])
        except Exception as exc:
            raise exc
        for vname in ["taper", "tapsmofrq"]:
            if lcls[vname] != defaults[vname]:
                wrng = "<freqanalysis> WARNING: `{}` keyword ignored by wavelet method"
                print(wrng.format(vname))

    # Warn the user in case other method-specifc options are set
    other = list(avail_methods)
    other.pop(other.index(method))
    settings = set()
    mth_defaults = {}
    glbls = globals()
    for mth_str in other:
        mth_defaults.update(get_defaults(glbls[mth_str]))
    settings = set(mth_defaults.keys()).difference(defaults.keys())
    for key in settings:
        if lcls[key] != mth_defaults[key]:
            wrng = "<freqanalysis> WARNING: `{kw:s}` keyword has no effect in " +\
                   "chosen method {m:s}"
            print(wrng.format(kw=key, m=method))

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

    # Prepare dict of optional keywords for computational class constructor
    kws = get_defaults(glbls[method]).keys()
    for kw in ["output", "noCompute"]:
        kws.pop(kws.index(kw))
    mth_input = {}
    for kw in kws:
        mth_input[kw] = lcls[kw]

    import ipdb; ipdb.set_trace()

    # Construct dict for classes of available methods
    methods = {
        "mtmfft": (MultiTaperFFT(**kws), 1/data.samplerate)
    }
    # methods = {
    #     "mtmfft": (MultiTaperFFT(taper=hann, tapsmofrq=tapsmofrq,
    #                              pad=pad, padtype=padtype,
    #                              polyorder=polyremoval),
    #                1 / data.samplerate)
    # }

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
           polyorder=None, output="pow",
           noCompute=False):

    nSamples = dat.shape[0]
    nChannels = dat.shape[1]

    # padding
    if pad:
        padWidth = np.zeros((dat.ndim, 2), dtype=int)
        if pad == "nextpow2":
            padWidth[0, 0] = _nextpow2(nSamples) - nSamples

        if padtype == "zero":
            dat = np.pad(dat, pad_width=padWidth,
                         mode="constant", constant_values=0)

        # update number of samples
        nSamples = dat.shape[0]

    if taper == dpss and (not taperopt):
        nTaper = np.int(np.floor(tapsmofrq * nSamples * dt))
        taperopt = {"NW": tapsmofrq, "Kmax": nTaper}
    win = np.atleast_2d(taper(nSamples, **taperopt))
    nTaper = win.shape[0]

    nFreq = int(np.floor(nSamples / 2) + 1)
    # taper x freq x channel
    outputShape = (nTaper, nFreq, nChannels)

    if noCompute:
        return outputShape, spectralDTypes[output]

    spec = np.zeros(outputShape, dtype=spectralDTypes[output])
    for taperIdx, taper in enumerate(win):
        if dat.ndim > 1:
            taper = np.tile(taper, (nChannels, 1)).T
        spec[taperIdx, ...] = spectralConversions[output](np.fft.rfft(dat * taper, axis=0))

    return spec


class MultiTaperFFT(ComputationalRoutine):
    computeFunction = staticmethod(mtmfft)

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    def preallocate_output(self, data, out):
        with h5py.File(out._filename, mode="w") as h5f:
            h5f.create_dataset(name="SpectralData",
                               dtype=self.dtype,
                               shape=(len(data.trials),) + self.outputShape)

    def handle_metadata(self, data, out):
        h5f = h5py.File(out._filename, mode="r")
        out.data = h5f["SpectralData"]

        time = np.arange(len(data.trials))
        time = time.reshape((time.size, 1))
        out.sampleinfo = np.hstack([time, time + 1])
        out.trialinfo = np.array(data.trialinfo)
        out._t0 = np.zeros((len(data.trials),))

        # Attach meta-data
        out.samplerate = data.samplerate
        out.channel = np.array(data.channel)
        out.taper = np.array([self.cfg["taper"].__name__] * self.outputShape[0])
        nFreqs = self.outputShape[out.dimord.index("freq") - 1]
        out.freq = np.linspace(0, 1, nFreqs) * (data.samplerate / 2)
        out.cfg = self.cfg

    def compute_with_dask(self, data, out):
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

    def compute_sequentially(self, data, out):
        with h5py.File(out._filename, "r+") as h5f:
            for tk, trl in enumerate(tqdm(data.trials, desc="Computing MTMFFT...")):                
                h5f["SpectralData"][tk, :,:,:] = self.computeFunction(trl, 1 / data.samplerate,
                                                                      **self.cfg)                          
                h5f.flush()

    @staticmethod
    def write_block(blk, filename, block_info=None):
        idx = block_info[0]["chunk-location"][-1]
        with h5py.File(filename, "r+") as h5f:
             h5f["SpectralData"][idx, :, :, :] = blk
        return idx


def _nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n


def wavelet(dat, dt,
            freqoi=None, polyorder=None, wav=Morlet(w0=6),
            stepsize=100, output="pow", noCompute=False):
    """ dat = samples x channel
    """

    dat = np.atleast_2d(dat)
    scales = wav.scale_from_period(np.reciprocal(freqoi))

    # time x taper=1 x freq x channel
    outputShape = (int(np.floor(dat.shape[0] / stepsize)),
                   1,
                   len(scales),
                   dat.shape[1])
    if noCompute:
        return outputShape, spectralDTypes[output]

    # cwt returns (len(scales),) + dat.shape
    transformed = cwt(dat, axis=0, wavelet=wav, widths=scales, dt=dt)
    transformed = transformed[:, 0:-1:int(stepsize), :, np.newaxis].transpose([1, 3, 0, 2])

    return spectralConversions[output](transformed)


class WaveletTransform(ComputationalRoutine):
    computeFunction = staticmethod(wavelet)

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    def initialize(self, data):
        timeAxis = data.dimord.index("time")
        minTrialLength = np.array(data._shapes)[:, timeAxis].min()
        if self.cfg["freqoi"] is None:
            self.cfg["freqoi"] = 1 / _get_optimal_wavelet_scales(minTrialLength,
                                                                 1 / data.samplerate,
                                                                 dj=0.25)

        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        self.outputShape, self.dtype = self.computeFunction(data.trials[0],
                                                            *self.argv, **dryRunKwargs)

    def preallocate_output(self, data, out):
        totalTriallength = np.int(np.sum(np.floor(np.array(data._shapes)[:, 0] /
                                                  self.cfg["stepsize"])))

        result = open_memmap(out._filename,
                             shape=(totalTriallength,) + self.outputShape[1:],
                             dtype=self.dtype,
                             mode="w+")
        del result

    def compute_sequentially(self, data, out):
        outIndex = 0
        for idx, trial in enumerate(tqdm(data.trials,
                                         desc="Computing Wavelet spectrum...")):
            tmp = self.computeFunction(trial, *self.argv, **self.cfg)
            selector = slice(outIndex, outIndex + tmp.shape[0])
            res = open_memmap(out._filename, mode="r+")[selector, :, :, :]
            res[...] = tmp
            del res
            data.clear()

    def compute_with_dask(self, data, out):
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


WaveletTransform.computeMethods = {"dask": WaveletTransform.compute_with_dask,
                                   "sequential": WaveletTransform.compute_sequentially}


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
