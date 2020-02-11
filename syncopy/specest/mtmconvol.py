# -*- coding: utf-8 -*-
# 
# Time-frequency analysis based on a short-time Fourier transform
# 
# Created: 2020-02-05 09:36:38
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-02-11 14:35:01>

# Builtin/3rd party package imports
import numpy as np
from scipy import signal

# Local imports
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import unwrap_io
from syncopy.datatype import padding
import syncopy.specest.freqanalysis as freq


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
        
    import ipdb; ipdb.set_trace()

    # Padding (updates no. of samples)
    padKw = None
    if pad is not None:
        padKw = padding(dat, padtype, pad=pad, padlength=padlength, 
                        prepadlength=prepadlength, postpadlength=postpadlength,
                        create_new=False)
        padbegin = max(0, padbegin - padKw["pad_width"][0, 0])
        padend = max(0, padend - padKw["pad_width"][0, 1])
        padKw["pad_width"][0, :] += [padbegin, padend]
        padbegin = 0
        padend = 0
    if padbegin > 0 or padend > 0:
        padKw = {"pad_width": np.array([[padbegin, padend], [0, 0]]),
                 "mode": "constant",
                 "constant_values": 0}
    if padKw is not None:
        dat = np.pad(dat, **padKw)

    # Get shape of output for dry-run phase
    nChannels = dat.shape[1]
    nFreq = foi.size
    nTime = toi.size
    outShape = (nTime, max(1, nTaper * keeptapers), nFreq, nChannels)
    if noCompute:
        return outShape, freq.spectralDTypes[output_fmt]
    
    # In case tapers aren't kept allocate `spec` "too big" and average afterwards
    spec = np.full((nTime, nTaper, nFreq, nChannels), np.nan, dtype=freq.spectralDTypes[output_fmt])
    
    # Collect keyword args for `stft` in dictionary
    stftKw = {"fs": samplerate,
              "window": taper,
              "nperseg": nperseg,
              "noverlap": noverlap,
              "return_onesided": True,
              "boundary": None,
              "padded": False,
              "axis": 0}
    
    # Call `stft` w/first taper to get freq/time indices
    win = np.atleast_2d(taper(nperseg, **taperopt))
    if equidistant:
        freq, time, pxx = signal.stft(dat[soi], **stftKw)
        fIdx = np.searchsorted(freq, foi)
        tIdx = np.searchsorted(time, toi)
        spec[:, 0, ...] = \
            freq.spectralConversions[output_fmt](
                pxx.reshape(nTime, 1, nFreq, nChannels))[tIdx, :, fIdx, :]
    else:
        halfWin = int(nperseg/2)
        freq, _, pxx = signal.stft(dat[soi[0] - halfWin, soi[0] + halfWin], **stftKw)
        fIdx = np.searchsorted(freq, foi)
        spec[0, 0, ...] = \
            freq.spectralConversions[output_fmt](
                pxx.reshape(1, 1, nFreq, nChannels))[:, :, fIdx, :]
        for tk in range(1, soi.size):
            spec[tk, 0, ...] = \
                freq.spectralConversions[output_fmt](
                    signal.stft(
                        dat[soi[tk] - halfWin, soi[tk] + halfWin],
                        **stftKw)[2].reshape(1, 1, nFreq, nChannels))[:, :, fIdx, :]

    # Compute FT using determined indices above for the remaining tapers (if any)
    for taperIdx in range(1, win.shape[0]):
        if equidistant:
            spec[:, taperIdx, ...] = \
                freq.spectralConversions[output_fmt](
                    signal.stft(
                        dat[soi],
                        **stftKw)[2].reshape(nTime, 1, nFreq, nChannels))[tIdx, :, fIdx, :]
        else:
            for tk, sample in enumerate(soi):
                spec[tk, taperIdx, ...] = \
                    freq.spectralConversions[output_fmt](
                        signal.stft(
                            dat[sample - halfWin, sample + halfWin], 
                            **stftKw)[2].reshape(1, 1, nFreq, nChannels))[:, :, fIdx, :]
    
    # Average across tapers if wanted
    if not keeptapers:
        return spec.mean(axis=1, keepdims=True)
    return spec
    

class MultiTaperFFTConvol(ComputationalRoutine):

    computeFunction = staticmethod(mtmconvol)

    def process_metadata(self, data, out):
        
        import ipdb; ipdb.set_trace()

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
    