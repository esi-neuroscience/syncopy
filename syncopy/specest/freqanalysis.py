# -*- coding: utf-8 -*-
# 
# SyNCoPy spectral estimation methods
# 
# Created: 2019-01-22 09:07:47
# Last modified by: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Last modification time: <2020-02-07 16:11:04>

# Builtin/3rd party package imports
from numbers import Number
import numpy as np
import scipy.signal.windows as spwin

# Local imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser 
from syncopy.shared import get_defaults
from syncopy.datatype import SpectralData, padding
from syncopy.datatype.methods.padding import _nextpow2
import syncopy.specest.wavelets as spywave 
from syncopy.shared.errors import SPYValueError, SPYTypeError, SPYWarning
from syncopy.shared.kwarg_decorators import (unwrap_cfg, unwrap_select, 
                                             detect_parallel_client)
from syncopy.specest.mtmfft import MultiTaperFFT
from syncopy.specest.wavelet import _get_optimal_wavelet_scales, WaveletTransform

# Module-wide output specs
spectralDTypes = {"pow": np.float32,
                  "fourier": np.complex128,
                  "abs": np.float32}

#: output conversion of complex fourier coefficients
spectralConversions = {"pow": lambda x: (x * np.conj(x)).real.astype(np.float32),
                       "fourier": lambda x: x.astype(np.complex128),
                       "abs": lambda x: (np.absolute(x)).real.astype(np.float32)}

#: available outputs of :func:`freqanalysis`
availableOutputs = tuple(spectralConversions.keys())

#: available tapers of :func:`freqanalysis`
availableTapers = ("hann", "dpss")

#: available spectral estimation methods of :func:`freqanalysis`
availableMethods = ("mtmfft", "wavelet")

__all__ = ["freqanalysis"]


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def freqanalysis(data, method='mtmfft', output='fourier',
                 keeptrials=True, foi=None, foilim=None, pad='nextpow2', padtype='zero',
                 padlength=None, polyremoval=False, polyorder=None,
                 taper="hann", tapsmofrq=None, keeptapers=False,
                 wav="Morlet", t_ftimwin=None, toi=None, width=6, 
                 out=None, **kwargs):
    """
    Perform a (time-)frequency analysis of time series data

    Parameters
    ----------
    data : `~syncopy.AnalogData`
        A child of :class:`syncopy.datatype.AnalogData`
    method : str
        Spectral estimation method, one of :data:`~.availableMethods` 
        (see below).
    output : str
        Output of spectral estimation, `'pow'` for power spectrum 
        (:obj:`numpy.float32`),  `'fourier'` (:obj:`numpy.complex128`)
        for complex fourier coefficients or `'abs'` for absolute values
        (:obj:`numpy.float32`).
    keeptrials : bool
        Flag whether to return individual trials or average
    foi : array-like or None
        List of frequencies of interest (Hz) for output. If desired frequencies
        cannot be exactly matched using the given data length and padding,
        the closest frequencies will be used. If `foi` is `None`
        or ``foi = "all"``, all frequencies are selected. 
    foilim : array-like (floats [fmin, fmax]) or None or "all"
        Frequency-window ``[fmin, fmax]`` (in Hz) of interest. Window 
        specifications must be sorted (e.g., ``[90, 70]`` is invalid) and not NaN 
        but may be unbounded (e.g., ``[-np.inf, 60.5]`` is valid). Edges `fmin` 
        and `fmax` are included in the selection. If `foilim` is `None` or 
        ``foilim = "all"``, all frequencies are selected. 
    pad : str or None
        One of `'absolute'`, `'relative'`, `'maxlen'`, `'nextpow2'` or `None`. 
        Padding method to be used in case trial do not have equal length. To
        ensure consistency of the output object, padding is always performed 
        with respect to the longest trial found in `data`. For instance, 
        `pad = 'nextpow2'` pads all trials in `data` to the next power of 2 higher 
        than the sample-count of the longest trial in `data`. See :func:`syncopy.padding` 
        for more information. If `pad` is `None`, no padding is performed and
        all trials have to have approximately the same length (up to next even 
        sample-count). 
    padtype : str
        Values to be used for padding. Can be 'zero', 'nan', 'mean', 
        'localmean', 'edge' or 'mirror'. See :func:`syncopy.padding` for 
        more information.
    padlength : None, bool or positive scalar
        Length to be padded to data in samples if `pad` is 'absolute' or 
        'relative'. See :func:`syncopy.padding` for more information.
    polyremoval : bool
        Flag whether a polynomial of order `polyorder` should be fitted and 
        subtracted from each trial before spectral analysis. 
        FIXME: not implemented yet.
    polyorder : int
        Order of the removed polynomial. For example, a value of 1 
        corresponds to a linear trend. The default is a mean subtraction, 
        thus a value of 0. 
        FIXME: not implemented yet.
    taper : str
        Windowing function, one of :data:`~.availableTapers` (see below).
    tapsmofrq : float
        The amount of spectral smoothing through  multi-tapering (Hz).
        Note that 4 Hz smoothing means plus-minus 4 Hz, i.e. a 8 Hz 
        smoothing box.        
    keeptapers : bool
        Flag for whether individual trials or average should be returned.            
    t_ftimwin : scalar
        Time-window length (in seconds). **Mandatory** if ``method = "mtmconvol"``.
    toi : scalar or array-like or "all"
        **Mandatory input** for time-frequency analysis methods (`method` is either 
        `"mtmconvo"l` or `"wavelet"`). 
        If `toi` is scalar, it must be a value between 0 and 1 indicating the 
        percentage of overlap between time-windows specified by `t_ftimwin` (only
        valid if `method` is `'mtmconvol'`, invalid for `'wavelet'`). 
        If `toi` is an array it explicitly selects the centroids of analysis 
        windows (in seconds). If `toi` is `"all"`, analysis windows are centered
        on all samples in the data. 
    width : scalar
        Nondimensional frequency constant of wavelet. For a Morlet wavelet 
        this number should be >= 6, which corresponds to 6 cycles within the 
        analysis window. 
    out : None or :class:`SpectralData` object
        None if a new :class:`SpectralData` object should be created,
        or the (empty) object into which the result should be written.

    Returns
    -------
    :class:`~syncopy.SpectralData`
        (Time-)frequency spectrum of input data
        
    Notes
    -----
    Coming soon...
        

    .. autodata:: syncopy.specest.freqanalysis.availableMethods

    .. autodata:: syncopy.specest.freqanalysis.availableOutputs

    .. autodata:: syncopy.specest.freqanalysis.availableTapers

    """
    
    # Make sure our one mandatory input object can be processed
    try:
        data_parser(data, varname="data", dataclass="AnalogData",
                    writable=None, empty=False)
    except Exception as exc:
        raise exc
    timeAxis = data.dimord.index("time")

    # Get everything of interest in local namespace
    defaults = get_defaults(freqanalysis)
    lcls = locals()

    # Ensure a valid computational method was selected    
    if method not in availableMethods:
        lgl = "'" + "or '".join(opt + "' " for opt in availableMethods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # Ensure a valid output format was selected    
    if output not in spectralConversions.keys():
        lgl = "'" + "or '".join(opt + "' " for opt in spectralConversions.keys())
        raise SPYValueError(legal=lgl, varname="output", actual=output)

    # Parse all Boolean keyword arguments
    for vname in ["keeptrials", "keeptapers", "polyremoval"]:
        if not isinstance(lcls[vname], bool):
            raise SPYTypeError(lcls[vname], varname=vname, expected="Bool")
        
    # If only a subset of `data` is to be processed, make some necessary adjustments
    # and compute minimal sample-count across (selected) trials
    if data._selection is not None:
        trialList = data._selection.trials
        sinfo = np.zeros((len(trialList), 2))
        for tk, trlno in enumerate(trialList):
            trl = data._preview_trial(trlno)
            tsel = trl.idx[timeAxis]
            if isinstance(tsel, list):
                sinfo[tk, :] = [0, len(tsel)]
            else:
                sinfo[tk, :] = [trl.idx[timeAxis].start, trl.idx[timeAxis].stop]
    else:
        trialList = list(range(len(data.trials)))
        sinfo = data.sampleinfo
    lenTrials = np.diff(sinfo)
        
    # Ensure padding selection makes sense: do not pad on a by-trial basis but 
    # use the longest trial as reference and compute `padlength` from there
    # (only relevant for "global" padding options such as `maxlen` or `nextpow2`)
    if not isinstance(pad, (str, type(None))):
        raise SPYTypeError(pad, varname="pad", expected="str or None")
    if pad:
        if pad == "maxlen":
            padlength = lenTrials.max()
        elif pad == "nextpow2":
            padlength = 0
            for ltrl in lenTrials:
                padlength = max(padlength, _nextpow2(ltrl))
            pad = "absolute"
        padding(data._preview_trial(trialList[0]), padtype, pad=pad, padlength=padlength,
                prepadlength=True)
    
        # Update `minSampleNum` to account for padding
        minSamplePos = lenTrials.argmin()
        minSampleNum = padding(data._preview_trial(trialList[minSamplePos]), padtype, pad=pad,
                               padlength=padlength, prepadlength=True).shape[timeAxis]
    
    else:
        if np.unique((np.floor(lenTrials / 2))).size > 1:
            lgl = "trials of approximately equal length"
            act = "trials of unequal length"
            raise SPYValueError(legal=lgl, varname="data", actual=act)
        minSampleNum = lenTrials.min()
        
    # Compute length (in samples) of shortest trial
    minTrialLength = minSampleNum/data.samplerate
    
    # Basic sanitization of frequency specifications
    if foi is not None:
        if isinstance(foi, str):
            if foi == "all":
                foi = None
            else:
                raise SPYValueError(legal="'all' or `None` or list/array", 
                                    varname="foi", actual=foi)
        else:
            try:
                array_parser(foi, varname="foi", hasinf=False, hasnan=False,
                            lims=[0, data.samplerate/2], dims=(None,))
            except Exception as exc:
                raise exc
    if foilim is not None:
        if isinstance(foilim, str):
            if foilim == "all":
                foilim = None
            else:
                raise SPYValueError(legal="'all' or `None` or `[fmin, fmax]`", 
                                    varname="foilim", actual=foilim)
        else:
            try:
                array_parser(foilim, varname="foilim", hasinf=False, hasnan=False,
                            lims=[0, data.samplerate/2], dims=(2,))
            except Exception as exc:
                raise exc
    if foi is not None and foilim is not None:
        lgl = "either `foi` or `foilim` specification"
        act = "both"
        raise SPYValueError(legal=lgl, varname="foi/foilim", actual=act)


    # Abort if desired frequency selection is empty
    if foi.size == 0:
        lgl = "non-empty frequency specification"
        act = "empty frequency selection"
        raise SPYValueError(legal=lgl, varname="foi/foilim", actual=act)
        
    # FIXME: implement detrending
    # see also https://docs.obspy.org/_modules/obspy/signal/detrend.html#polynomial
    if polyremoval is True or polyorder is not None:
        raise NotImplementedError("Detrending has not been implemented yet.")

    # Check detrending options for consistency
    if polyremoval:
        try:
            scalar_parser(polyorder, varname="polyorder", lims=[0, 8], ntype="int_like")
        except Exception as exc:
            raise exc
    else:
        if polyorder != defaults["polyorder"]:
            msg = "`polyorder` keyword will be ignored since `polyremoval` is `False`!"
            SPYWarning(msg)

    # Prepare keyword dict for logging (use `lcls` to get actually provided 
    # keyword values, not defaults set above)
    log_dct = {"method": method,
               "output": output,
               "keeptapers": keeptapers,
               "keeptrials": keeptrials,
               "polyremoval": polyremoval,
               "polyorder": polyorder,
               "pad": lcls["pad"],
               "padtype": lcls["padtype"],
               "padlength": lcls["padlength"],
               "foi": lcls["foi"]}
    
    import ipdb; ipdb.set_trace()

    # 1st: Check time-frequency inputs to prepare/sanitize `toi`
    if method in ["mtmconvol", "wavelet"]:
        
        # Get start/end timing info respecting potential in-place selection
        if toi is None:
            raise SPYTypeError(toi, varname="toi", expected="scalar or array-like or 'all'")
        if data._selection is not None:
            tStart = data._selection.trialdefinition[:, 2] / data.samplerate
        else:
            tStart = data._t0 / data.samplerate
        tEnd = tStart + sinfo.squeeze() / data.samplerate

        # Process `toi`: `overlap > 1` => all, `0 < overlap < 1` => percentage, 
        # `overlap < 0` => discrete `toi`
        if isinstance(toi, str):
            if toi != "all":
                lgl = "`toi = 'all'` to center analysis windows on all time-points"
                raise SPYValueError(legal=lgl, varname="toi", actual=toi)
            overlap = 1.1
            toi = Ellipsis
            equidistant = True
        if isinstance(toi, Number):
            if method == "wavelet":
                lgl = "array of time-points wavelets are to be centered on"
                act = "scalar value"
                raise SPYValueError(legal=lgl, varname="toi", actual=act)
            try:
                scalar_parser(toi, varname="toi", lims=[0, 1])
            except Exception as exc:
                raise exc
            overlap = toi
            toi = Ellipsis
            equidistant = True
        else:
            overlap = -1
            try:
                array_parser(toi, varname="toi", hasinf=False, hasnan=False,
                             lims=[tStart.min(), tEnd.max()], dims=(None,))
            except Exception as exc:
                raise exc
            tSteps = np.diff(toi)
            if (tSteps < 0).any():
                lgl = "ordered list/array of time-points"
                act = "unsorted list/array"
                raise SPYValueError(legal=lgl, varname="toi", actual=act)
            if np.unique(tSteps).size > 1:
                equidistant = False
            else:
                equidistant = True

        # The above `overlap`, `equidistant` etc. is really only relevant for `mtmconvol`        
        if method == "mtmconvol":
            try:
                scalar_parser(t_ftimwin, varname="t_ftimwin", lims=[0, minTrialLength])
            except Exception as exc:
                raise exc
            nperseg = int(t_ftimwin * data.samplerate)
            minSampleNum = nperseg
            
            if overlap < 0:         # `toi` is equidistant range or disjoint point
                noverlap = nperseg - int(tSteps[0] * data.samplerate)
            elif 0 <= overlap <= 1: # `toi` is percentage
                noverlap = int(overlap * nperseg)
            else:                   # `toi` is "all"
                noverlap = nperseg - 1

            # Compute necessary padding at begin/end of trials to fit sliding windows
            offStart = ((toi[0] - tStart) * data.samplerate).astype(int)
            padBegin = nperseg/2 - offStart
            padBegin = (padBegin > 0) * padBegin
            
            offEnd = ((tEnd - toi[-1]) * data.samplerate).astype(int)
            padEnd = nperseg/2 - offEnd
            padEnd = (padEnd > 0) * padEnd

            # Compute sample-indices (one slice/array per trial) from time-selections
            soi = []            
            if not equidistant:
                for tk in range(len(trialList)):
                    soi.append((data.samplerate * (toi + tStart[tk]).astype(int)))
            else:
                for tk in range(len(trialList)):
                    soi.append(slice(toi[0], toi[-1]))
            
        else: # wavelets: probably some `toi` gymnastics
            pass
        
    # mtmconvol: iterated
    # f, t, Zxx = signal.stft(x[500: 500+500], fs=1000, nperseg=500, noverlap=0, boundary=None)
        
    # padding:
    # spy.padding(np.ones((250,2)), padtype='zero', pad='relative', prepadlength=6)
        
    # 2nd: Preprocess frequency selection 
    
    # Check options specific to mtm*-methods (particularly tapers and foi/freqs alignment)
    if "mtm" in method:

        #: available tapers
        options = ["hann", "dpss"]
        if taper not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(legal=lgl, varname="taper", actual=taper)
        taper = getattr(spwin, taper)

        # Advanced usage: see if `taperopt` was provided - if not, leave it empty
        taperopt = kwargs.get("taperopt", {})
        if not isinstance(taperopt, dict):
            raise SPYTypeError(taperopt, varname="taperopt", expected="dictionary")

        # Set/get `tapsmofrq` if we're working w/Slepian tapers
        if taper.__name__ == "dpss":

            # Try to derive "sane" settings by using 3/4 octave smoothing of highest `foi`
            # following Hill et al. "Oscillatory Synchronization in Large-Scale
            # Cortical Networks Predicts Perception", Neuron, 2011
            if tapsmofrq is None:
                foimax = foi.max()
                tapsmofrq = (foimax * 2**(3/4/2) - foimax * 2**(-3/4/2)) / 2
            else:
                try:
                    scalar_parser(tapsmofrq, varname="tapsmofrq", lims=[1, np.inf])
                except Exception as exc:
                    raise exc
            
            # Get/compute number of tapers to use (at least 1 and max. 50)
            nTaper = taperopt.get("Kmax", 1)
            if not taperopt:
                nTaper = int(max(2, min(50, np.floor(tapsmofrq * minSampleNum * 1 / data.samplerate))))
                taperopt = {"NW": tapsmofrq, "Kmax": nTaper}
                
        else:
            nTaper = 1

        # Warn the user in case `tapsmofrq` has no effect
        if tapsmofrq is not None and taper.__name__ != "dpss":
            msg = "`tapsmofrq` is only used if `taper` is `dpss`!"
            SPYWarning(msg)
            
        # Construct array of maximally attainable frequencies
        nFreq = int(np.floor(minSampleNum / 2) + 1)
        freqs = np.linspace(0, data.samplerate / 2, nFreq)
        
        # Match desired frequencies as close as possible to actually attainable freqs
        if foi is not None:
            foi = np.array(foi)
            foi.sort()
            foi = foi[foi <= freqs.max()]
            foi = foi[foi >= freqs.min()]
            fidx = np.searchsorted(freqs, foi, side="left")
            for k, fid in enumerate(fidx):
                if np.abs(freqs[fid - 1] - foi[k]) < np.abs(freqs[fid] - foi[k]):
                    fidx[k] = fid -1
            fidx = np.unique(fidx)
            foi = freqs[fidx]
        else:
            foi = freqs

        # Crop desired frequency band from array of actually attainable freqs
        if foilim is not None:
            foi = np.intersect1d(np.where(freqs >= foilim[0])[0], np.where(freqs <= foilim[1])[0])
        else:
            foi = freqs
            
        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["taper"] = lcls["taper"]
        log_dct["tapsmofrq"] = lcls["tapsmofrq"]
        log_dct["nTaper"] = nTaper
        
        # Check for non-default values of options not supported by chosen method
        kwdict = {"wav": wav, "width": width}
        for name, kwarg in kwdict.items():
            if kwarg is not lcls[name]:
                msg = "option `{}` has no effect in methods `mtmfft` and `mtmconvol`!"
                SPYWarning(msg.format(name))
            
    # Now, prepare explicit compute-classes for chosen method
    if method == "mtmfft":
        
        # Check for non-default values of options not supported by chosen method
        kwdict = {"t_ftimwin": t_ftimwin, "toi": toi}
        for name, kwarg in kwdict.items():
            if kwarg is not lcls[name]:
                msg = "option `{}` has no effect in method `mtmfft`!"
                SPYWarning(msg.format(name))
        
        # Set up compute-class
        specestMethod = MultiTaperFFT(1 / data.samplerate,
                                      nTaper=nTaper, 
                                      timeAxis=timeAxis, 
                                      taper=taper, 
                                      taperopt=taperopt,
                                      tapsmofrq=tapsmofrq,
                                      pad=pad,
                                      padtype=padtype,
                                      padlength=padlength,
                                      foi=foi,
                                      keeptapers=keeptapers,
                                      polyorder=polyorder,
                                      output_fmt=output)
        
    elif method == "mtmconvol":
        # set up class
        pass

    elif method == "wavelet":
        pass

        # check if taper, tapsmofrq, keeptapers is defined
        
        # check for consistency of width, wav
        
        options = ["Morlet", "Paul", "DOG", "Ricker", "Marr", "Mexican_hat"]
        if wav not in options:
            lgl = "'" + "or '".join(opt + "' " for opt in options)
            raise SPYValueError(legal=lgl, varname="wav", actual=wav)
        wav = getattr(spywave, wav)

        if isinstance(toi, Number):
            try:
                scalar_parser(toi, varname="toi", lims=[0, 1])
            except Exception as exc:
                raise exc
        else:
            try:
                array_parser(toi, varname="toi", hasinf=False, hasnan=False,
                             lims=[timing.min(), timing.max()], dims=(None,))
            except Exception as exc:
                raise exc
            toi = np.array(toi)
            toi.sort()

        if foi is None:
            foi = 1 / _get_optimal_wavelet_scales(minTrialLength,
                                                  1/data.samplerate,
                                                  dj=0.25)

        # FIXME: width setting depends on chosen wavelet
        if width is not None:
            try:
                scalar_parser(width, varname="width", lims=[1, np.inf])
            except Exception as exc:
                raise exc

        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["wav"] = lcls["wav"]
        log_dct["toi"] = lcls["toi"]
        log_dct["width"] = lcls["width"]

        # Set up compute-class
        specestMethod = WaveletTransform(1/data.samplerate, 
                                         timeAxis,
                                         foi,
                                         toi=toi,
                                         polyorder=polyorder,
                                         wav=wav,
                                         width=width,
                                         output_fmt=output)
        
    # If provided, make sure output object is appropriate
    if out is not None:
        try:
            data_parser(out, varname="out", writable=True, empty=True,
                        dataclass="SpectralData",
                        dimord=SpectralData().dimord)
        except Exception as exc:
            raise exc
        new_out = False
    else:
        out = SpectralData(dimord=SpectralData._defaultDimord)        
        new_out = True

    # Perform actual computation
    specestMethod.initialize(data, 
                             chan_per_worker=kwargs.get("chan_per_worker"),
                             keeptrials=keeptrials)
    specestMethod.compute(data, out, parallel=kwargs.get("parallel"), log_dict=log_dct)

    # Either return newly created output container or simply quit
    return out if new_out else None
