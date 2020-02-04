# -*- coding: utf-8 -*-
# 
# Spectral estimation with (multi-)tapered FFT
# 
# Created: 2019-09-02 14:25:34
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-02-04 14:32:31>

# Builtin/3rd party package imports
import numpy as np
import scipy.signal.windows as spwin

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.datatype import padding
import syncopy.specest.freqanalysis as freq
from syncopy.shared.kwarg_decorators import unwrap_io


# Local workhorse that performs the computational heavy lifting
@unwrap_io
def mtmfft(trl_dat, dt, nTaper=1, timeAxis=0,
           taper=spwin.hann, taperopt={}, tapsmofrq=None,
           pad="nextpow2", padtype="zero", padlength=None, foi=None,
           keeptapers=True, polyorder=None, output_fmt="pow",
           noCompute=False, chunkShape=None):
    """Compute (multi-)tapered Fourier transform
    
    Parameters
    ----------
    trl_dat : 2D :class:`numpy.ndarray`
        Multi-channel uniformly sampled time-series 
    dt : float
        sampling interval (between 0 and 1)
    nTaper : int
        number of filter windows to use
    timeAxis : int
        Index of time axis (0 or 1)
    taper : function
        Windowing function handle, one of :mod:`scipy.signal.windows`. This 
        function is called as ``taper(nSamples, **taperopt)``
    taperopt : dict
        Additional keyword arguments passed to the `taper` function
    tapsmofrq : float
        The amount of spectral smoothing through  multi-tapering (Hz).
        Note that 4 Hz smoothing means plus-minus 4 Hz, i.e. a 8 Hz 
        smoothing box.  
    pad : str
        `'absolute'`, `'relative'`, `'maxlen'`, or `'nextpow2'`.
        See :func:`syncopy.padding` for more information.
    padtype : str
        Values to be used for padding. Can be 'zero', 'nan', 'mean', 
        'localmean', 'edge' or 'mirror'. See :func:`syncopy.padding` for 
        more information.
    padlength : None, bool or positive scalar
        length to be padded to data in samples if `pad` is 'absolute' or 
        'relative'. See :func:`syncopy.padding` for more information.
    foi : array-like
        List of frequencies of interest  (Hz) for output. If desired frequencies
        cannot be exactly matched using the given data length and padding,
        the closest frequencies will be used.
    keeptapers : bool
        Flag for keeping individual tapers or average
    output_fmt : str               
        Output of spectral estimation, `'pow'` for power spectrum 
        (:obj:`numpy.float32`),  `'fourier'` (:obj:`numpy.complex128`)
        for complex fourier coefficients or `'abs'` for absolute values
        (:obj:`numpy.float32`).
        
    Returns
    -------
    :class:`numpy.ndarray`
        Complex or real spectrum of input (padded) data

    """

    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat

    # Padding (updates no. of samples)
    if pad is not None:
        dat = padding(dat, padtype, pad=pad, padlength=padlength, prepadlength=True)
    nSamples = dat.shape[0]
    nChannels = dat.shape[1]
    
    # Determine frequency band and shape of output (time=1 x taper x freq x channel)
    nFreq = int(np.floor(nSamples / 2) + 1)
    fidx = slice(None)
    if foi is not None:
        freqs = np.linspace(0, 1 /(2 * dt), nFreq)
        foi = foi[foi <= freqs.max()]
        foi = foi[foi >= freqs.min()]
        fidx = np.searchsorted(freqs, foi, side="left")
        for k, fid in enumerate(fidx):
            if np.abs(freqs[fid - 1] - foi[k]) < np.abs(freqs[fid] - foi[k]):
                fidx[k] = fid -1
        fidx = np.unique(fidx)
        nFreq = fidx.size
    outShape = (1, max(1, nTaper * keeptapers), nFreq, nChannels)
    
    # For initialization of computational routine, just return output shape and dtype
    if noCompute:
        return outShape, freq.spectralDTypes[output_fmt]

    # In case tapers aren't kept allocate `spec` "too big" and average afterwards
    spec = np.full((1, nTaper, nFreq, nChannels), np.nan, dtype=freq.spectralDTypes[output_fmt])
    fill_idx = tuple([slice(None, dim) for dim in outShape[2:]])

    # Actual computation
    win = np.atleast_2d(taper(nSamples, **taperopt))
    for taperIdx, taper in enumerate(win):
        if dat.ndim > 1:
            taper = np.tile(taper, (nChannels, 1)).T
        spec[(0, taperIdx,) + fill_idx] = freq.spectralConversions[output_fmt](np.fft.rfft(dat * taper, axis=0)[fidx, :])

    # Average across tapers if wanted
    if not keeptapers:
        return spec.mean(axis=1, keepdims=True)
    else:
        return spec


class MultiTaperFFT(ComputationalRoutine):

    computeFunction = staticmethod(mtmfft)

    def process_metadata(self, data, out):
        
        # Some index gymnastics to get trial begin/end "samples"
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
            for row in range(trl.shape[0]):
                trl[row, :2] = [row, row + 1]
        else:
            chanSec = slice(None)
            time = np.arange(len(data.trials))
            time = time.reshape((time.size, 1))
            trl = np.hstack((time, time + 1, 
                             np.zeros((len(data.trials), 1)), 
                             np.array(data.trialinfo)))

        # Attach constructed trialdef-array (if even necessary)
        if self.keeptrials:
            out.trialdefinition = trl
        else:
            out.trialdefinition = np.array([[0, 1, 0]])

        # Attach remaining meta-data
        out.samplerate = data.samplerate
        out.channel = np.array(data.channel[chanSec])
        out.taper = np.array([self.cfg["taper"].__name__] * self.outputShape[out.dimord.index("taper")])
        out.freq = self.cfg["foi"]
