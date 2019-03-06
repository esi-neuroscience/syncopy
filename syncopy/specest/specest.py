# -*- coding: utf-8 -*-
#
# SynCoPy spectral estimation methods
# 
# Created: 2019-01-22 09:07:47
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-03-06 11:25:39>

# Builtin/3rd party package imports
import sys
import numpy as np
import scipy.signal as signal
import scipy.signal.windows as windows
from numpy.lib.format import open_memmap
from tqdm import tqdm
from syncopy import __dask__
if __dask__:
    import dask
    import dask.array as da
    from dask.distributed import get_client

# Local imports
from syncopy.utils import spy_data_parser
from syncopy.datatype import SpectralData

__all__ = ["mtmfft"]

##########################################################################################
def mtmfft(obj, taper=windows.hann, pad="nextpow2", padtype="zero",
           polyorder=None, taperopt={}, fftAxis=1, tapsmofrq=None, out=None):

    # FIXME: parse remaining input arguments
    if polyorder:
        raise NotImplementedError("Detrending has not been implemented yet.")

    # Make sure input object can be processed
    try:
        spy_data_parser(obj, varname="obj", dataclass="AnalogData",
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

    # Set parameters applying to all trials: FIXME: make sure trials
    # are consistent, i.e., padding results in same no. of freqs across all trials
    fftAxis = obj.dimord.index("time")
    if pad == "nextpow2":
        nSamples = _nextpow2(obj._shapes[0][fftAxis])
    else:
        raise NotImplementedError("Coming soon...")
    if taper == windows.dpss and (not taperopt):
        nTaper = np.int(np.floor(tapsmofrq * nSamples/obj.samplerate))
        taperopt = {"NW": tapsmofrq, "Kmax": nTaper}

    # Compute taper in shape nTaper x nSamples and determine size of freq. axis
    win = np.atleast_2d(taper(nSamples, **taperopt))
    nFreq = int(np.floor(nSamples / 2) + 1)
    freq = np.arange(0, np.floor(nSamples / 2) + 1) * obj.samplerate/nSamples
    
    # Allocate memory map for results
    res = open_memmap(out._filename,
                      shape=(len(obj.trials), win.shape[0], obj._shapes[0][0], nFreq),
                      dtype="complex",
                      mode="w+")
    del res

    # See if a dask client is running
    try:
        use_dask = bool(get_client())
    except:
        use_dask = False
    
    # Perform parallel computation
    if use_dask:

        # Point to trials on disk by using delayed **static** method calls
        lazy_trial = dask.delayed(obj._copy_trial, traverse=False)
        lazy_trls = [lazy_trial(trialno,
                                obj._filename,
                                obj.dimord,
                                obj.sampleinfo,
                                obj.hdr)\
                     for trialno in range(obj.sampleinfo.shape[0])]

        # Construct a distributed dask array block by stacking delayed trials
        trl_block = da.hstack([da.from_delayed(trl,
                                               shape=obj._shapes[sk],
                                               dtype=obj.data.dtype) for sk, trl in enumerate(lazy_trls)])

        # Use `map_blocks` to compute spectra for each trial in the constructred dask array
        specs = trl_block.map_blocks(_mtmfft_bytrl, win, nFreq,  pad, padtype, fftAxis, use_dask,
                                     dtype="complex",
                                     chunks=(win.shape[0], obj.data.shape[0], nFreq),
                                     new_axis=[0])

        # Write computed spectra in pre-allocated memmap
        result = specs.map_blocks(_mtmfft_writer, out._filename,
                                  dtype="complex",
                                  chunks=(1,),
                                  drop_axis=[0,1])

        # Perform actual computation
        result.compute()

    # Serial calculation solely relying on NumPy
    else:
        for tk, trl in enumerate(tqdm(obj.trials, desc="Computing MTMFFT...")):
            res = open_memmap(out._filename, mode="r+")[tk, :, :, :]
            res[...] = _mtmfft_bytrl(trl, win, nFreq, pad, padtype, fftAxis, use_dask)
            del res
            obj.clear()

    # First things first: attach data to output object
    out._data = open_memmap(out._filename, mode="r+")

    # We can't simply use ``redefinetrial`` here, prep things by hand
    time = np.arange(len(obj.trials))
    time = time.reshape((time.size, 1))
    out.sampleinfo = np.hstack([time, time + 1])
    out.trialinfo = np.array(obj.trialinfo)
    out._t0 = np.zeros((len(obj.trials),))
    
    # Attach meta-data
    out.samplerate = obj.samplerate
    out.channel = np.array(obj.channel)
    out.taper = np.array([taper.__name__] * win.shape[0])
    out.freq = freq
    cfg = {"method" : sys._getframe().f_code.co_name,
           "taper" : taper.__name__,
           "padding" : pad,
           "padtype" : padtype,
           "polyorder" : polyorder,
           "taperopt" : taperopt,
           "tapsmofrq" : tapsmofrq}
    out.cfg = cfg
    out.cfg = dict(obj.cfg)

    # Write log
    out._log = str(obj._log) + out._log
    log = "computed multi-taper FFT with settings\n" +\
          "\ttaper = {tpr:s}\n" +\
          "\tpadding = {pad:s}\n" +\
          "\tpadtype = {pat:s}\n" +\
          "\tpolyorder = {pol:s}\n" +\
          "\ttaperopt = {topt:s}\n" +\
          "\ttapsmofrq = {tfr:s}\n"
    out.log = log.format(tpr=cfg["taper"],
                         pad=cfg["padding"],
                         pat=cfg["padtype"],
                         pol=str(cfg["polyorder"]),
                         topt=str(cfg["taperopt"]),
                         tfr=str(cfg["tapsmofrq"]))

    # Happy breakdown
    return out if new_out else None
    
##########################################################################################
def _mtmfft_writer(blk, resname, block_info=None):
    """
    Pumps computed spectra into target memmap
    """
    idx = block_info[0]["chunk-location"][-1]
    res = open_memmap(resname, mode="r+")[idx, :, :, :]
    res[...] = blk
    res = None
    del res
    return idx

##########################################################################################
def _mtmfft_bytrl(trl, win, nFreq,  pad, padtype, fftAxis, use_dask):
    """
    Performs the actual heavy-lifting
    """

    # move fft/samples dimension into first place
    trl = np.moveaxis(np.atleast_2d(trl), fftAxis, 1)
    nSamples = trl.shape[1]
    nChannels = trl.shape[0]

    # padding
    if pad:
        padWidth = np.zeros((trl.ndim, 2), dtype=int)
        if pad == "nextpow2":
            padWidth[1, 0] = _nextpow2(nSamples) - nSamples
        else:
            padWidth[1, 0] = np.ceil((pad - T) / dt).astype(int)
        if padtype == "zero":
            trl = np.pad(trl, pad_width=padWidth,
                          mode="constant", constant_values=0)

        # update number of samples
        nSamples = trl.shape[1]

    # Decide whether to further parallelize or plow through entire chunk
    if use_dask and trl.size * trl.dtype.itemsize * 1024**(-2) > 1000:
        spex = []
        for tap in win:
            if trl.ndim > 1:
                tap = np.tile(tap, (nChannels, 1))
            prod = da.from_array(trl * tap, chunks=(1, trl.shape[1]))
            spex.append(da.fft.rfft(prod))
        spec = da.stack(spex)
    else:
        # taper x chan x freq
        spec = np.zeros((win.shape[0],) + (nChannels,) + (nFreq,), dtype=complex)
        for wIdx, tap in enumerate(win):
            if trl.ndim > 1:
                tap = np.tile(tap, (nChannels, 1))
            spec[wIdx, ...] = np.fft.rfft(trl * tap, axis=1)

    return spec

##########################################################################################
def _nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n
