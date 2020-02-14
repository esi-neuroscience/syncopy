# -*- coding: utf-8 -*-
# 
# Time-frequency analysis based on a short-time Fourier transform
# 
# Created: 2020-02-05 09:36:38
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-02-14 16:49:42>

# Builtin/3rd party package imports
import numpy as np
from scipy import signal

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.datatype import padding
import syncopy.specest.freqanalysis as spyfreq
from syncopy.shared.errors import SPYWarning


# Local workhorse that performs the computational heavy lifting
@unwrap_io
def mtmconvol(
    trl_dat, soi, padbegin, padend,
    samplerate=None, noverlap=None, nperseg=None, equidistant=True, toi=None, foi=None,
    nTaper=1, timeAxis=0, taper=signal.windows.hann, taperopt={}, 
    pad=None, padtype="zero", padlength=None, prepadlength=True, postpadlength=True, 
    keeptapers=True, polyorder=None, output_fmt="pow",
    noCompute=False, chunkShape=None):
    """
    Coming soon...
    """
    
    # Re-arrange array if necessary and get dimensional information
    if timeAxis != 0:
        dat = trl_dat.T       # does not copy but creates view of `trl_dat`
    else:
        dat = trl_dat
        
    # Padding: either combination of user-choice/necessity or just necessity
    if pad is not None:
        padKw = padding(dat, padtype, pad=pad, padlength=padlength, 
                        prepadlength=prepadlength, postpadlength=postpadlength,
                        create_new=False)
        padbegin = max(0, padbegin - padKw["pad_width"][0, 0])
        padend = max(0, padend - padKw["pad_width"][0, 1])
        padKw["pad_width"][0, :] += [padbegin, padend]
        padbegin = 0
        padend = 0
        dat = np.pad(dat, **padKw)
    if padbegin > 0 or padend > 0:
        dat = padding(dat, padtype, pad="relative", padlength=None, 
                      prepadlength=padbegin, postpadlength=padend)

    # Get shape of output for dry-run phase
    nChannels = dat.shape[1]
    if isinstance(toi, np.ndarray):
        nTime = toi.size
    else:
        nTime = dat.shape[0]
    nFreq = foi.size
    outShape = (nTime, max(1, nTaper * keeptapers), nFreq, nChannels)
    if noCompute:
        return outShape, spyfreq.spectralDTypes[output_fmt]
    
    # In case tapers aren't kept allocate `spec` "too big" and average afterwards
    spec = np.full((nTime, nTaper, nFreq, nChannels), np.nan, dtype=spyfreq.spectralDTypes[output_fmt])
    
    # Collect keyword args for `stft` in dictionary
    stftKw = {"fs": samplerate,
              "nperseg": nperseg,
              "noverlap": noverlap,
              "return_onesided": True,
              "boundary": None,
              "padded": False,
              "axis": 0}
    
    # Call `stft` w/first taper to get freq/time indices
    win = np.atleast_2d(taper(nperseg, **taperopt))
    stftKw["window"] = win[0, :]
    if equidistant:
        freq, _, pxx = signal.stft(dat[soi, :], **stftKw)
        fIdx = np.searchsorted(freq, foi)
        spec[:, 0, ...] = \
            spyfreq.spectralConversions[output_fmt](
                pxx.reshape(nTime, nFreq, nChannels))[:, fIdx, :]
    else:
        halfWin = int(nperseg/2)
        import ipdb; ipdb.set_trace()
        freq, _, pxx = signal.stft(dat[soi[0] - halfWin: soi[0] + halfWin, :], **stftKw)
        fIdx = np.searchsorted(freq, foi)
        spec[0, 0, ...] = \
            spyfreq.spectralConversions[output_fmt](
                pxx.reshape(nFreq, nChannels))[fIdx, :]
        for tk in range(1, soi.size):
            spec[tk, 0, ...] = \
                spyfreq.spectralConversions[output_fmt](
                    signal.stft(
                        dat[soi[tk] - halfWin: soi[tk] + halfWin, :],
                        **stftKw)[2].reshape(nFreq, nChannels))[fIdx, :]

    # Compute FT using determined indices above for the remaining tapers (if any)
    for taperIdx in range(1, win.shape[0]):
        stftKw["window"] = win[taperIdx, :]
        if equidistant:
            spec[:, taperIdx, ...] = \
                spyfreq.spectralConversions[output_fmt](
                    signal.stft(
                        dat[soi, :],
                        **stftKw)[2].reshape(nTime, nFreq, nChannels))[:, fIdx, :]
        else:
            for tk, sample in enumerate(soi):
                spec[tk, taperIdx, ...] = \
                    spyfreq.spectralConversions[output_fmt](
                        signal.stft(
                            dat[sample - halfWin: sample + halfWin, :],
                            **stftKw)[2].reshape(nFreq, nChannels))[fIdx, :]

    # Average across tapers if wanted
    if not keeptapers:
        return np.nanmean(spec, axis=1, keepdims=True)
    return spec
    

class MultiTaperFFTConvol(ComputationalRoutine):

    computeFunction = staticmethod(mtmconvol)

    def process_metadata(self, data, out):
        
        # Extract user-provided time selection        
        toi = self.cfg["toi"]
        nToi = toi.size 

        # Some index gymnastics to get trial begin/end samples
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
        else:
            chanSec = slice(None)
            trl = data.trialdefinition
        time = np.cumsum([nToi] * trl.shape[0])
        # time = time.reshape((time.size, 1))
        trl[:, 0] = time - nToi
        trl[:, 1] = time

        # Important: differentiate b/w equidistant time ranges and disjoint points        
        if self.cfg["equidistant"]:
            out.samplerate = 1 / (toi[1] - toi[0])
            trl[:, 2] = int(toi[0] * out.samplerate)
        else:
            msg = "`SpectralData`'s `time` property currently does not support " +\
                  "unevenly spaced `toi` selections!"
            SPYWarning(msg, caller="freqanalysis")
            out.samplerate = 1.0
            trl[:, 2] = 0

        # Attach constructed trialdef-array (if necessary)
        if self.keeptrials:
            out.trialdefinition = trl
        else:
            out.trialdefinition = np.array([[0, 1, 0]])

        # Attach remaining meta-data
        out.channel = np.array(data.channel[chanSec])
        out.taper = np.array([self.cfg["taper"].__name__] * self.outputShape[out.dimord.index("taper")])
        out.freq = self.cfg["foi"]
