# -*- coding: utf-8 -*-
# 
# Time-frequency analysis based on a short-time Fourier transform
# 
# Created: 2020-02-05 09:36:38
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-05-22 14:32:23>

# Builtin/3rd party package imports
import numpy as np
from scipy import signal

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.datatype import padding
import syncopy.specest.freqanalysis as spyfreq
from syncopy.shared.errors import SPYWarning
from syncopy.shared.tools import best_match


# Local workhorse that performs the computational heavy lifting
@unwrap_io
def mtmconvol(
    trl_dat, soi, padbegin, padend,
    samplerate=None, noverlap=None, nperseg=None, equidistant=True, toi=None, foi=None,
    nTaper=1, timeAxis=0, taper=signal.windows.hann, taperopt={}, 
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
        
    # Pad input array if necessary
    if padbegin > 0 or padend > 0:
        dat = padding(dat, "zero", pad="relative", padlength=None, 
                      prepadlength=padbegin, postpadlength=padend)

    # Get shape of output for dry-run phase
    nChannels = dat.shape[1]
    if isinstance(toi, np.ndarray):
        nTime = toi.size
        stftBdry = None
        stftPad = False
    else:
        nTime = dat.shape[0]
        stftBdry = "zeros"
        stftPad = True
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
              "boundary": stftBdry,
              "padded": stftPad,
              "axis": 0}
    
    # Call `stft` w/first taper to get freq/time indices
    win = np.atleast_2d(taper(nperseg, **taperopt))
    stftKw["window"] = win[0, :]
    if equidistant:
        freq, _, pxx = signal.stft(dat[soi, :], **stftKw)
        _, fIdx = best_match(freq, foi, squash_duplicates=True)
        spec[:, 0, ...] = \
            spyfreq.spectralConversions[output_fmt](
                pxx.reshape(nTime, nFreq, nChannels))[:, fIdx, :]
    else:
        freq, _, pxx = signal.stft(dat[soi[0], :], **stftKw)
        _, fIdx = best_match(freq, foi, squash_duplicates=True)
        spec[0, 0, ...] = \
            spyfreq.spectralConversions[output_fmt](
                pxx.reshape(nFreq, nChannels))[fIdx, :]
        for tk in range(1, len(soi)):
            spec[tk, 0, ...] = \
                spyfreq.spectralConversions[output_fmt](
                    signal.stft(
                        dat[soi[tk], :], 
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
                            dat[sample, :],
                            **stftKw)[2].reshape(nFreq, nChannels))[fIdx, :]

    # Average across tapers if wanted
    if not keeptapers:
        return np.nanmean(spec, axis=1, keepdims=True)
    return spec
    

class MultiTaperFFTConvol(ComputationalRoutine):

    computeFunction = staticmethod(mtmconvol)

    def process_metadata(self, data, out):

        # Get trialdef array + channels from source        
        if data._selection is not None:
            chanSec = data._selection.channel
            trl = data._selection.trialdefinition
        else:
            chanSec = slice(None)
            trl = data.trialdefinition

        # Construct trialdef array (if necessary)
        if self.keeptrials:
            
            # If `toi` is array, construct timing, otherwise... 
            toi = self.cfg["toi"]
            if isinstance(toi, np.ndarray):
                
                # Some index gymnastics to get trial begin/end samples
                nToi = toi.size
                time = np.cumsum([nToi] * trl.shape[0])
                trl[:, 0] = time - nToi
                trl[:, 1] = time
                
                # If trigger onset was part of `toi`, get its relative position wrt 
                # to other elements, otherwise use first element as "onset"
                t0Idx = np.where(toi == 0)[0]
                if t0Idx:
                    trl[:, 2] = -t0Idx[0]
                else:
                    trl[:, 2] = 0
                    
                # Important: differentiate b/w equidistant time ranges and disjoint points        
                if self.cfg["equidistant"]:
                    out.samplerate = 1 / (toi[1] - toi[0])
                else:
                    msg = "`SpectralData`'s `time` property currently does not support " +\
                        "unevenly spaced `toi` selections!"
                    SPYWarning(msg, caller="freqanalysis")
                    out.samplerate = 1.0
                    trl[:, 2] = 0
                    
            # ... i.e., `toi='all'`, simply copy from source
            else:
                out.samplerate = data.samplerate
                
            out.trialdefinition = trl
        else:
            out.trialdefinition = np.array([[0, 1, 0]])
            
        # Attach remaining meta-data
        out.channel = np.array(data.channel[chanSec])
        out.taper = np.array([self.cfg["taper"].__name__] * self.outputShape[out.dimord.index("taper")])
        out.freq = self.cfg["foi"]
