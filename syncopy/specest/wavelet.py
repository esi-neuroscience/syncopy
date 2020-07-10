# -*- coding: utf-8 -*-
# 
# Time-frequency analysis with wavelets
# 
# Created: 2019-09-02 14:44:41
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-07-10 14:50:33>

# Builtin/3rd party package imports
import numpy as np
from numbers import Number

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
import syncopy.specest.wavelets as spywave 

def wavelet(
    trl_dat, preselect, postselect, padbegin, padend,
    samplerate=None, toi=None, foi=None, timeAxis=0, 
    wav=None, scales=None,
    polyremoval=None, output_fmt="pow",
    noCompute=False, chunkShape=None):
    
    # dt, timeAxis, foi,
    #         toi=0.1, polyremoval=None, wav=spywave.Morlet,
    #         width=6, output_fmt="pow",
    #         noCompute=False, chunkShape=None):
    """ 
    dat = samples x channel
    """

    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat

    # Pad input array if wanted/necessary
    if padbegin > 0 or padend > 0:
        dat = padding(dat, "zero", pad="relative", padlength=None, 
                      prepadlength=padbegin, postpadlength=padend)

    # Get shape of output for dry-run phase
    nChannels = dat.shape[1]
    if isinstance(toi, np.ndarray):     # `toi` is an array of time-points
        nTime = toi.size
    else:                               # `toi` is 'all'
        nTime = dat.shape[0]
    nFreq = foi.size
    outShape = (nTime, nFreq, nChannels)
    if noCompute:
        return outShape, spyfreq.spectralDTypes[output_fmt]


    spec = cwt(dat[preselect, :], axis=0, wavelet=wav, widths=scales, dt=1/samplerate)


    # Get time-stepping or explicit time-points of interest
    if isinstance(toi, Number):
        toi /= dt
        tsize = int(np.floor(nSamples / toi))
    else:
        tsize = toi.size

    # Output shape: time x taper=1 x freq x channel
    import ipdb; ipdb.set_trace()
    scales = wav.scale_from_period(1/foi) # use wav.fourier_period instead?
    outShape = (tsize,
                1,
                len(scales),
                nChannels)

    # For initialization of computational routine, just return output shape and dtype
    if noCompute:
        return outShape, spectralDTypes[output_fmt]

    # Actual computation: ``cwt`` returns `(len(scales),) + dat.shape`
    transformed = spywave.cwt(dat, axis=0, wavelet=wav, widths=scales, dt=dt)
    transformed = transformed[:, 0:-1:tsize, :, np.newaxis].transpose([1, 3, 0, 2])

    return spectralConversions[output_fmt](transformed)

    # METADATA:
    # use fourier_period(self, s) to get foi from scales


class WaveletTransform(ComputationalRoutine):

    computeFunction = staticmethod(wavelet)

    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

    def compute_parallel(self, data, out):

        # Point to trials on disk by using delayed **static** method calls
        lazy_trial = dask.delayed(data._copy_trial, traverse=False)
        lazy_trls = [lazy_trial(trialno,
                                data._filename,
                                data.dimord,
                                data.sampleinfo,
                                data.hdr)
                     for trialno in range(data.sampleinfo.shape[0])]

        # Stack trials along new (3rd) axis inserted on the left
        trl_block = da.stack([da.from_delayed(trl, shape=data._shapes[sk],
                                              dtype=data.data.dtype)
                              for sk, trl in enumerate(lazy_trls)])

        # Use `map_blocks` to compute spectra for each trial in the
        # constructed dask array
        specs = trl_block.map_blocks(self.computeFunction,
                                     *self.argv, **self.cfg,
                                     dtype="complex",
                                     chunks=self.cfg["chunkShape"],
                                     new_axis=[0])
        specs = specs.reshape(self.outputShape)

        # Average across trials if wanted
        if not self.keeptrials:
            specs = specs.mean(axis=0)

        return specs

    # def initialize(self, data):
    #     timeAxis = data.dimord.index("time")
    #     minTrialLength = np.array(data._shapes)[:, timeAxis].min()
    #     if self.cfg["foi"] is None:
    #         self.cfg["foi"] = 1 / _get_optimal_wavelet_scales(minTrialLength,
    #                                                           1 / data.samplerate,
    #                                                           dj=0.25)
    # 
    #     if self.cfg["toi"] is None:
    #         asdf
    #         1/foi[-1] * w0
    # 
    #     dryRunKwargs = copy(self.cfg)
    #     dryRunKwargs["noCompute"] = True
    #     self.chunkShape, self.dtype = self.computeFunction(data.trials[0],
    #                                                         *self.argv, **dryRunKwargs)

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

    def process_metadata(self, data, out):
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


def _get_optimal_wavelet_scales(self, nSamples, dt, dj=0.25, s0=None):
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
    
    # Compute `s0` so that the equivalent Fourier period is approximately ``2 * dt```
    if s0 is None:
        s0 = self.scale_from_period(2*dt)
        
    # Largest scale
    J = int((1 / dj) * np.log2(nSamples * dt / s0))
    return s0 * 2 ** (dj * np.arange(0, J + 1))
