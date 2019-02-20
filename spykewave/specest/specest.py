# specest.py - SpykeWave spectral estimation methods
# 
# Created: January 22 2019
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2019-02-20 13:15:28>

###########
# Add spykewave package to Python search path
import os
import sys
spw_path = os.path.abspath(".." + os.sep + "..")
if spw_path not in sys.path:
    sys.path.insert(0, spw_path)

# Import Spykewave
import spykewave as sw

from memory_profiler import memory_usage

###########

# Builtin/3rd party package imports
import numpy as np
import scipy.signal as signal
import scipy.signal.windows as windows
from numpy.lib.format import open_memmap
from tempfile import mkstemp
from spykewave import __dask__
if __dask__:
    import dask
    import dask.array as da
    from dask.distributed import Lock, get_worker


from copy import copy

# Local imports
from spykewave.utils import spw_basedata_parser
from spykewave.datatype import SpectralData

__all__ = ["mtmfft"]

##########################################################################################
def mtmfft(obj, taper=windows.hann, pad="nextpow2", padtype="zero",
           polyorder=None, taperopt={}, fftAxis=1, tapsmofrq=None, out=None):

    # FIXME: parse remaining input arguments
    if polyorder:
        raise NotImplementedError("Detrending has not been implemented yet.")

    # Make sure input object can be processed
    try:
        spw_basedata_parser(obj, varname="obj", dimord=["label", "sample"],
                            writable=None, empty=False)
    except Exception as exc:
        raise exc
    
    # If provided, make sure output object is appropriate 
    if out is not None:
        try:
            spw_basedata_parser(out, varname="out", writable=True,
                                dimord=["freq", "spec"], segmentlabel="freq")
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = SpectralData()
        new_out = True

    # Set parameters applying to all segments: FIXME: make sure segments
    # are consistent, i.e., padding results in same no. of freqs across all segments
    fftAxis = obj.dimord.index("sample")
    if pad == "nextpow2":
        nSamples = _nextpow2(obj.shapes[0][1])
    else:
        raise NotImplementedError("Coming soon...")
    if taper == windows.dpss and (not taperopt):
        nTaper = np.int(np.floor(tapsmofrq * T))
        taperopt = {"NW": tapsmofrq, "Kmax": nTaper}

    # Compute taper in shape nTaper x nSamples and determine size of freq. axis
    win = np.atleast_2d(taper(nSamples, **taperopt))
    nFreq = int(np.floor(nSamples / 2) + 1)
    freq = np.arange(0, np.floor(nSamples / 2) + 1) * obj.samplerate/nSamples
    
    # Allocate memory map for results
    res = open_memmap(out._filename,
                      shape=(win.shape[0], obj.shapes[0][0], nFreq*len(obj.segments)),
                      dtype="complex",
                      mode="w+")
    del res

    # Point to data segments on disk by using delayed **static** method calls
    lazy_segment = dask.delayed(obj._copy_segment, traverse=False)
    lazy_segs = [lazy_segment(segno, obj._filename, obj.seg, obj.hdr, obj.dimord, obj.segmentlabel)\
                 for segno in range(obj._seg.shape[0])]

    # Construct a distributed dask array block by stacking delayed segments
    seg_block = da.hstack([da.from_delayed(seg,
                                           shape=obj.shapes[sk],
                                           dtype=obj.data.dtype) for sk, seg in enumerate(lazy_segs)])
    
    # Use `map_blocks` to compute spectra for each segment in the constructred dask array
    specs = seg_block.map_blocks(_mtmfft_byseg, win, nFreq,  pad, padtype, fftAxis, 
                                 dtype="complex",
                                 chunks=(win.shape[0], obj.data.shape[0], nFreq),
                                 new_axis=[0])

    # Write computed spectra in pre-allocated memmap
    result = specs.map_blocks(_mtmfft_writer, nFreq, out._filename,
                          dtype="complex",
                          chunks=(1,),
                          drop_axis=[0,1])

    # Perform actual computation
    result.compute()

    # Attach results to output object: start w/ dimensional info (order matters!)
    out._dimlabels["taper"] = [taper.__name__] * win.shape[0]
    out._dimlabels["label"] = obj.label
    out._dimlabels["freq"] = freq

    # Write data and meta-info
    out._samplerate = obj.samplerate
    seg = obj.seg
    for k in range(seg.shape[0]):
        seg[k, [0, 1]] = [k*nFreq, (k+1)*nFreq]
    out._seg = seg
    out._data = open_memmap(out._filename, mode="r+")
    out.cfg = {"method" : sys._getframe().f_code.co_name,
               "taper" : taper.__name__,
               "padding" : pad,
               "padtype" : padtype,
               "polyorder" : polyorder,
               "taperopt" : taperopt,
               "tapsmofrq" : tapsmofrq}

    # Write log
    log = "computed multi-taper FFT with settings..."
    out.log = log

    # import ipdb; ipdb.set_trace()
    
    # Happy breakdown
    return out if new_out else None
    
##########################################################################################
def _mtmfft_writer(blk, nFreq, resname, block_info=None):
    """
    Pumps computed spectra into target memmap
    """
    idx = block_info[0]["chunk-location"][-1]
    res = open_memmap(resname, mode="r+")[:, :, idx*nFreq : (idx + 1)*nFreq]
    res[...] = blk
    del res
    return idx

##########################################################################################
def _mtmfft_byseg(seg, win, nFreq,  pad, padtype, fftAxis):
    """
    Performs the actual heavy-lifting
    """

    # move fft/samples dimension into first place
    seg = np.moveaxis(np.atleast_2d(seg), fftAxis, 1)
    nSamples = seg.shape[1]
    nChannels = seg.shape[0]

    # padding
    if pad:
        padWidth = np.zeros((seg.ndim, 2), dtype=int)
        if pad == "nextpow2":
            padWidth[1, 0] = _nextpow2(nSamples) - nSamples
        else:
            padWidth[1, 0] = np.ceil((pad - T) / dt).astype(int)
        if padtype == "zero":
            seg = np.pad(seg, pad_width=padWidth,
                          mode="constant", constant_values=0)

        # update number of samples
        nSamples = seg.shape[1]

    # Decide whether to further parallelize or plow through entire chunk
    if seg.size * seg.dtype.itemsize * 1024**(-2) > 1000:
        spex = []
        for tap in win:
            if seg.ndim > 1:
                tap = np.tile(tap, (nChannels, 1))
            prod = da.from_array(seg * tap, chunks=(1, seg.shape[1]))
            spex.append(da.fft.rfft(prod))
        spec = da.stack(spex)
    else:
        # taper x chan x freq
        spec = np.zeros((win.shape[0],) + (nChannels,) + (nFreq,), dtype=complex)
        for wIdx, tap in enumerate(win):
            if seg.ndim > 1:
                tap = np.tile(tap, (nChannels, 1))
            spec[wIdx, ...] = np.fft.rfft(seg * tap, axis=1)

    return spec

##########################################################################################
def _nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n


def create_test_data(frequencies=(6.5, 22.75, 67.2, 100.5)):
    freq = np.array(frequencies)
    amp = np.ones(freq.shape)
    phi = np.random.rand(len(freq)) * 2 * np.pi
    signal = np.random.rand(2000) + 0.3
    dt = 0.001
    t = np.arange(signal.size) * dt
    for idx, f in enumerate(freq):
        signal += amp[idx] * np.sin(2 * np.pi * f * t + phi[idx])
    return signal


if __name__ == '__main__':

    datadir = ".." + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + "Data"\
              + os.sep + "testdata" + os.sep
    basename = "MT_RFmapping_session-168a1"
    intervals = np.load('../examples/intervals.npy')
    dataFiles = [os.path.join(datadir, basename + ext) for ext in ["_xWav.lfp", "_xWav.mua"]]
    data = sw.BaseData(filename=dataFiles, trialdefinition=intervals, filetype="esi")

    tdata = sw.load_spw('../examples/regular_segs')
    # # tdata = sw.load_spw('../examples/mtmfft')
    res = mtmfft(data)
    # import matplotlib.pyplot as plt
    # plt.ion()
    # data = create_test_data()
    # data = np.vstack([data, data])
    # spec = _mtmfft_compute(data, dt=0.001, pad="nextpow2",
    #                                   taper=windows.hann,
    #                                   tapsmofrq=2)
    # fig, ax = plt.subplots(2)
    # ax[0].plot(data)
    # ax[1].plot(freq, np.squeeze(np.mean(np.absolute(spec), axis=0)), '.-')
    # 
    # ax[1].set_xlim([-0.5, 105.5])
    # plt.draw()
