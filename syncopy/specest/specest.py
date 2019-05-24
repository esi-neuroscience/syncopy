# -*- coding: utf-8 -*-
# 
# SyNCoPy spectral estimation methods
# 
# Created: 2019-01-22 09:07:47
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-05-24 16:55:03>

# Builtin/3rd party package imports
import sys
import numpy as np
import scipy.signal.windows as spwin
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
import syncopy.specest.wavelets as spywave 
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.shared.parsers import unwrap_cfg
from syncopy import __dask__
if __dask__:
    import dask
    import dask.array as da
    import dask.distributed as dd

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
                 keeptrials=True, foi=None, pad='nextpow2', polyremoval=False,
                 polyorder=None, padtype='zero',
                 taper="hann", taperopt={}, tapsmofrq=None, keeptapers=True,
                 wav="Morlet", toi=None, width=6,
                 out=None, cfg=None):
    """
    Explain taperopt...
    Explain default of toi
    """
    
    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", dataclass="AnalogData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc

    # Get everything of interest in local namespace
    defaults = get_defaults(freqanalysis)
    lcls = locals()
    glbls = globals()

    # Ensure a valid computational method was selected
    avail_methods = ["mtmfft", "wavelet"]
    if method not in avail_methods:
        lgl = "'" + "or '".join(opt + "' " for opt in avail_methods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # Ensure a valid output format was selected
    options = ["pow", "fourier", "abs"]
    if output not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="output", actual=output)

    # Patch `output` keyword to not collide w/dask's ``blockwise`` output
    defaults["output_fmt"] = defaults.pop("output")
    output_fmt = output

    # Parse all Boolean keyword arguments
    for vname in ["keeptrials", "keeptapers", "polyremoval"]:
        if not isinstance(lcls[vname], bool):
            raise SPYTypeError(lcls[vname], varname=vname, expected="Bool")

    # Ensure padding selection makes sense
    options = [None, "nextpow2", "zero"]
    if pad not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="pad", actual=pad)
    options = ["zero"]
    if padtype not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="padtype", actual=padtype)

    # For vetting `toi` and `foi`: get timing information of input object
    timing = np.array([np.array([-data.t0[k], end - start - data.t0[k]])/data.samplerate
                       for k, (start, end) in enumerate(data.sampleinfo)])
    
    # Ensure frequency-of-interest is below Nyquist and above reciprocal min trial length
    if foi is not None:
        minlen = timing[0, :].max() - timing[1, :].min()
        try:
            array_parser(foi, varname="foi", hasinf=False, hasnan=False,
                         lims=[1/minlen, data.samplerate/2], dims=(None,))
        except Exception as exc:
            raise exc
        foi = np.array(foi)
        foi.sort()

    # FIXME: implement detrending
    if polyremoval or polyorder is not None:
        raise NotImplementedError("Detrending has not been implemented yet.")
    
    # # Check detrending options for consistency
    # if polyremoval:
    #     try:
    #         scalar_parser(polyorder, varname="polyorder", lims=[0, 8], ntype="int_like")
    #     except Exception as exc:
    #         raise exc
    # else:
    #     if polyorder != defaults["polyorder"]:
    #         print("<freqanalysis> WARNING: `polyorder` keyword will be ignored " +\
    #               "since `polyremoval` is `False`!")

    # Ensure consistency of method-specific options
    if method == "mtmfft":
        options = ["hann", "dpss"]
        if taper not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(legal=lgl, varname="taper", actual=taper)
        taper = getattr(spwin, taper)
        if not isinstance(taperopt, dict):
            raise SPYTypeError(taperopt, varname="taperopt", expected="dictionary")
        if tapsmofrq is not None:
            try:
                scalar_parser(tapsmofrq, varname="tapsmofrq", lims=[0.1, np.inf])
            except Exception as exc:
                raise exc

    elif method == "wavelet":
        
        options = ["Morlet", "Paul", "DOG", "Ricker", "Marr", "Mexican_hat"]
        if wav not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(legal=lgl, varname="wav", actual=wav)
        wav = getattr(spywave, wav)
        
        if toi is not None:
            try:
                array_parser(toi, varname="toi", hasinf=False, hasnan=False,
                             lims=[timing.min(), timing.max()], dims=(None,))
            except Exception as exc:
                raise exc
            toi = np.array(toi)
            toi.sort()
            
        if width is not None:
            try:
                scalar_parser(width, varname="width", lims=[1, np.inf])
            except Exception as exc:
                raise exc

    # Warn the user in case other method-specifc options are set
    other = list(avail_methods)
    other.pop(other.index(method))
    mth_defaults = {}
    for mth_str in other:
        mth_defaults.update(get_defaults(glbls[mth_str]))
    kws = list(get_defaults(glbls[method]).keys())
    shared_kws = set(defaults.keys()).difference(kws)
    settings = shared_kws.intersection(mth_defaults.keys()) 
    for key in settings:
        m_default = mth_defaults[key]
        if callable(m_default):
            m_default = m_default.__name__
        if lcls[key] != m_default:
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
    # (update `lcls` to reflect changes in method-specifc options)
    lcls = locals()
    mth_input = {}
    kws.remove("noCompute")
    for kw in kws:
        mth_input[kw] = lcls[kw]

    # Construct dict of classes of available methods
    methods = {
        "mtmfft": MultiTaperFFT(1/data.samplerate, **mth_input)
    }

    # Compute length of trials in output object to get stacking size
    # stacking = {"mtmfft" : np.ones((len(data.trials),))}
    stacking = {"mtmfft" : len(data.trials)}
    #             "wavelet" : np.int(np.sum(np.floor(np.array(data._shapes)[:, 0] /
    #                                                self.cfg["stepsize"])))}


    # Perform actual computation (w/ or w/o dask)
    try:
        dd.get_client()
        use_dask = True
    except:
        use_dask = False
    specestMethod = methods[method]
    specestMethod.initialize(data, stacking[method])
    specestMethod.compute(data, out, parallel=use_dask)
    import ipdb; ipdb.set_trace()

    # if output == "power":
    #     powerOut = out.copy(deep=T)
    #     for trial in out.trials:
    #         trial = np.absolute(trial)

    # Either return newly created output container or simply quit
    return out if newOut else None


# Local workhorse that performs the computational heavy lifting
def mtmfft(dat, dt,  
           taper=spwin.hann, taperopt={}, tapsmofrq=None,
           pad="nextpow2", padtype="zero", foi=None, keeptapers=True, keeptrials=True,
           polyorder=None, output_fmt="pow",
           noCompute=False):

    # Get dimensional information
    nSamples = dat.shape[0]
    nChannels = dat.shape[1]

    # Padding (updates no. of samples)
    if pad:
        padWidth = np.zeros((dat.ndim, 2), dtype=int)
        if pad == "nextpow2":
            padWidth[0, 0] = _nextpow2(nSamples) - nSamples
        if padtype == "zero":
            dat = np.pad(dat, pad_width=padWidth,
                         mode="constant", constant_values=0)
        nSamples = dat.shape[0]

    # Construct taper(s)
    if taper == spwin.dpss and (not taperopt):
        nTaper = np.int(np.floor(tapsmofrq * nSamples * dt))
        taperopt = {"NW": tapsmofrq, "Kmax": nTaper}
    win = np.atleast_2d(taper(nSamples, **taperopt))
    nTaper = win.shape[0]

    # Determine frequency band and shape of output (taper x freq x channel)
    nFreq = int(np.floor(nSamples / 2) + 1)
    fidx = slice(None)
    if foi is not None:
        freqs = np.linspace(0, 1, nFreq) * 1/(2*dt)
        foi = foi[foi <= freqs.max()]
        foi = foi[foi >= freqs.min()]
        fidx = np.unique(np.searchsorted(freqs, foi))
        nFreq = fidx.size
    chunkShape = (max(1, nTaper * keeptapers), nFreq, nChannels)

    # For initialization of computational routine, just return output shape and dtype
    if noCompute:
        return chunkShape, spectralDTypes[output_fmt]

    # Actual computation
    spec = np.zeros((1, nTaper, nFreq, nChannels), dtype=spectralDTypes[output_fmt])
    for taperIdx, taper in enumerate(win):
        if dat.ndim > 1:
            taper = np.tile(taper, (nChannels, 1)).T
        spec[0, taperIdx, ...] = spectralConversions[output_fmt](np.fft.rfft(dat * taper, axis=0)[fidx, :])

    # asdf_idx = tuple([slice(*dim) for dim in block_info[1]["array-location"]])
    # cnt = np.ravel_multi_index(block_info[1]["chunk-location"],
    #                            block_info[1]["num-chunks"])
    # asdf[asdf_idx] = cnt

    # Average across tapers if wanted
    if not keeptapers:
        return spec.mean(axis=0) #FIXME!
    else:
        return spec


class MultiTaperFFT(ComputationalRoutine):

    computeFunction = staticmethod(mtmfft)

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    # def preallocate_output(self, data, out, parallel=False):
    #     with h5py.File(out._filename, mode="w") as h5f:
    #         h5f.create_dataset(name="SpectralData",
    #                            dtype=self.dtype,
    #                            shape=(len(data.trials),) + self.chunkShape)


    def compute_parallel(self, data, out):
        
        # Point to trials on disk by using delayed **static** method calls
        lazy_trial = dask.delayed(data._copy_trial, traverse=False)
        lazy_trls = [lazy_trial(trialno,
                                data._filename,
                                data.dimord,
                                data.sampleinfo,
                                data.hdr)
                     for trialno in range(data.sampleinfo.shape[0])]

        trl_block = da.vstack([da.from_delayed(trl, shape=data._shapes[sk],
                                               dtype=data.data.dtype)
                              for sk, trl in enumerate(lazy_trls)])


        # b = db.from_sequence(['1.dat', '2.dat', ...]).map(load_from_filename)

        import dask.bag as db
        trl_bag = db.from_sequence([trialno for trialno in range(data.sampleinfo.shape[0])]).map(data._copy_trial,
                                                                                                 data._filename,
                                                                                                 data.dimord,
                                                                                                 data.sampleinfo,
                                                                                                 data.hdr)

        specs_bag = trl_bag.map(self.computeFunction, *self.argv, **self.cfg)
        
        # import dask.bag as dbb
        # bb = dbb.from_sequence([da.from_delayed(trl, shape=data._shapes[sk], dtype=data.data.dtype) for sk, trl in enumerate(lazy_trls)])   
        
        # import ipdb; ipdb.set_trace()

        # Construct a distributed dask array block by stacking delayed trials
        trl_block = da.hstack([da.from_delayed(trl, shape=data._shapes[sk],
                                               dtype=data.data.dtype)
                               for sk, trl in enumerate(lazy_trls)])


        # Use `map_blocks` to compute spectra for each trial in the
        # constructed dask array
        specs = trl_block.map_blocks(self.computeFunction,
                                     *self.argv, **self.cfg,
                                     dtype="complex",
                                     chunks=self.chunkShape,
                                     new_axis=[0])
        
        import ipdb;ipdb.set_trace()
        
        # bb = da.empty(self.outputShape, dtype="complex", chunks=(1,) + self.chunkShape)
        # 
        # self.cfg.pop("block_info")
        # da.map_blocks(self.computeFunction, trl_block, bb, **self.cfg, dtype="complex", chunks=self.chunkShape, new_axis=[0])
        # 
        # def ff(x, block_info=None):
        #     print(x.shape)
        #     print(block_info)
        #     return x
        # 
        # 
        # 
        # 
        # da.map_blocks(self.computeFunction, trl_block, bb, **self.cfg, dtype="complex", chunks=self.chunkShape, new_axis=[0])
        # 
        # specs = trl_block.map_blocks(self.computeFunction, b,
        #                              *self.argv, **self.cfg,
        #                              dtype="complex",
        #                              chunks=self.chunkShape,
        #                              new_axis=[0])
        
        # Average across trials if wanted
        if not self.cfg["keeptrials"]:
            specs = specs.mean(axis=0)

        import inspect
        import syncopy as spy

        h5f = h5py.File(data._filename, mode="r")  
        h5keys = list(h5f.keys())
        idx = [h5keys.count(dclass) for dclass in spy.datatype.__all__ \
               if not (inspect.isfunction(getattr(spy.datatype, dclass)))]

        import ipdb;ipdb.set_trace()

        chksh = 20*(1,), *tuple((s,) for s in self.chunkShape)
        
        return specs

        # # # Write computed spectra in pre-allocated memmap
        # # if out is not None:
        # if True:
        #     daskResult = specs.map_blocks(self.write_block, out._filename,
        #                                   dtype=self.dtype, drop_axis=[0, 1],
        #                                   chunks=(1,))
        # return daskResult.compute()

    def compute_sequential(self, data, out):
        with h5py.File(out._filename, "r+") as h5f:
            for tk, trl in enumerate(tqdm(data.trials, desc="Computing MTMFFT...")):                
                h5f["SpectralData"][tk, :,:,:] = self.computeFunction(trl, 1 / data.samplerate,
                                                                      **self.cfg)                          
                h5f.flush()

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
        out.taper = np.array([self.cfg["taper"].__name__] * self.chunkShape[0])
        nFreqs = self.outputShape[out.dimord.index("freq") - 1]
        out.freq = np.linspace(0, 1, nFreqs) * (data.samplerate / 2)
        out.cfg = self.cfg
        
    # @staticmethod
    # def write_block(blk, filename, block_info=None):
    #     idx = block_info[0]["chunk-location"][-1]
    #     with h5py.File(filename, "r+") as h5f:
    #          h5f["SpectralData"][idx, :, :, :] = blk
    #     return idx


def _nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n


def wavelet(dat, dt,
            toi=None, foi=None, polyorder=None, wav=spywave.Morlet,
            width=6, output_fmt="pow", noCompute=False):
    """ dat = samples x channel
    """

    dat = np.atleast_2d(dat)
    scales = wav.scale_from_period(1/foi)

    # time x taper=1 x freq x channel
    stepsize = 100 # FIXME: stepsize = toi
    chunkShape = (int(np.floor(dat.shape[0] / stepsize)),
                   1,
                   len(scales),
                   dat.shape[1])
    if noCompute:
        return chunkShape, spectralDTypes[output_fmt]

    # cwt returns (len(scales),) + dat.shape
    transformed = cwt(dat, axis=0, wavelet=wav, widths=scales, dt=dt)
    transformed = transformed[:, 0:-1:int(stepsize), :, np.newaxis].transpose([1, 3, 0, 2])

    return spectralConversions[output_fmt](transformed)


class WaveletTransform(ComputationalRoutine):
    computeFunction = staticmethod(wavelet)

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    def initialize(self, data):
        timeAxis = data.dimord.index("time")
        minTrialLength = np.array(data._shapes)[:, timeAxis].min()
        if self.cfg["foi"] is None:
            self.cfg["foi"] = 1 / _get_optimal_wavelet_scales(minTrialLength,
                                                              1 / data.samplerate,
                                                              dj=0.25)

        if self.cfg["toi"] is None:
            asdf
            1/foi[-1] * w0

        dryRunKwargs = copy(self.cfg)
        dryRunKwargs["noCompute"] = True
        self.chunkShape, self.dtype = self.computeFunction(data.trials[0],
                                                            *self.argv, **dryRunKwargs)

    def preallocate_output(self, data, out):
        totalTriallength = np.int(np.sum(np.floor(np.array(data._shapes)[:, 0] /
                                                  self.cfg["stepsize"])))

        result = open_memmap(out._filename,
                             shape=(totalTriallength,) + self.chunkShape[1:],
                             dtype=self.dtype,
                             mode="w+")
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

    def compute_with_dask(self, data, out):
        raise NotImplementedError("Dask computation of wavelet transform is not yet implemented")

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
                                   "sequential": WaveletTransform.compute_sequential}


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
