# -*- coding: utf-8 -*-
# 
# SyNCoPy spectral estimation methods
# 
# Created: 2019-01-22 09:07:47
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-10-02 09:49:54>

# Builtin/3rd party package imports
import numpy as np
import scipy.signal.windows as spwin
from numbers import Number

# Local imports
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser 
from syncopy.shared import get_defaults
from syncopy.datatype import SpectralData, padding
import syncopy.specest.wavelets as spywave 
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy.shared.parsers import unwrap_cfg
from syncopy import __dask__
from syncopy.specest.mtmfft import MultiTaperFFT
from syncopy.specest.wavelet import _get_optimal_wavelet_scales, WaveletTransform
# import syncopy.specest
if __dask__:
    import dask.distributed as dd

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
def freqanalysis(data, method='mtmfft', output='fourier',
                 keeptrials=True, foi=None, pad='nextpow2', padtype='zero',
                 padlength=None, polyremoval=False, polyorder=None,
                 taper="hann", tapsmofrq=None, keeptapers=True,
                 wav="Morlet", toi=0.1, width=6, select=None,
                 out=None, **kwargs):
    """Perform a (time-)frequency analysis of time series data

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
    foi : array-like
        List of frequencies of interest (Hz) for output. If desired frequencies
        cannot be exactly matched using the given data length and padding,
        the closest frequencies will be used.
    pad : str
        One of `'absolute'`, `'relative'`, `'maxlen'`, or `'nextpow2'`.
        See :func:`syncopy.padding` for more information.
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
    toi : scalar or array-like
        If `toi` is scalar, it must be a value between 0 and 1 indicating
        percentage of samplerate to use as stepsize. If `toi` is an array it
        explicitly selects time points (seconds). This option does not affect
        all frequency analysis methods.
    width : scalar
        Nondimensional frequency constant of wavelet. For a Morlet wavelet 
        this number should be >= 6, which correspondonds to 6 cycles within
        FIXME standard deviations of the enveloping Gaussian.       
    select : dict or :class:`~syncopy.datatype.base_data.StructDict`
        Select subset of input data for processing, e.g., using 
        ``select = {"channel": range(50)}`` performs spectral analysis using
        only the first 50 channels in `data`. Please refer to 
        :func:`syncopy.selectdata` for further usage details. 
    out : None or :class:`SpectralData` object
        None if a new :class:`SpectralData` object should be created,
        or the (empty) object into which the result should be written.


    .. autodata:: syncopy.specest.freqanalysis.availableMethods

    .. autodata:: syncopy.specest.freqanalysis.availableOutputs

    .. autodata:: syncopy.specest.freqanalysis.availableTapers


    Returns
    -------
    :class:`~syncopy.SpectralData`
        (Time-)frequency spectrum of input data


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
    glbls = globals()

    # Ensure a valid computational method was selected    
    if method not in availableMethods:
        lgl = "'" + "or '".join(opt + "' " for opt in availableMethods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # Ensure a valid output format was selected    
    if output not in spectralConversions.keys():
        lgl = "'" + "or '".join(opt + "' " for opt in spectralConversions.keys())
        raise SPYValueError(legal=lgl, varname="output", actual=output)

    # # Patch `output` keyword to not collide w/dask's ``blockwise`` output
    # defaults["output_fmt"] = defaults.pop("output")
    # output_fmt = output

    # Parse all Boolean keyword arguments
    for vname in ["keeptrials", "keeptapers", "polyremoval"]:
        if not isinstance(lcls[vname], bool):
            raise SPYTypeError(lcls[vname], varname=vname, expected="Bool")

    # Ensure padding selection makes sense (just leverage `padding`'s error checking)
    try:
        padding(data.trials[0], padtype, pad=pad, padlength=padlength,
                prepadlength=True)
    except Exception as exc:
        raise exc

    # For vetting `toi` and `foi`: get timing information of input object
    timing = np.array([np.array([-data._t0[k], end - start - data._t0[k]])/data.samplerate
                       for k, (start, end) in enumerate(data.sampleinfo)])

    # Construct array of maximally attainable frequency band and set/align `foi`
    minSampleNum = np.diff(data.sampleinfo).min()
    if pad:
        minSamplePos = np.diff(data.sampleinfo).argmin()
        minSampleNum = padding(data.trials[minSamplePos], padtype, pad=pad,
                               padlength=padlength, prepadlength=True).shape[timeAxis]
    minTrialLength = minSampleNum/data.samplerate
    nFreq = int(np.floor(minSampleNum / 2) + 1)
    freqs = np.linspace(0, data.samplerate/2, nFreq)

    # Match desired frequencies as close as possible to actually attainable freqs
    if foi is not None:
        try:
            array_parser(foi, varname="foi", hasinf=False, hasnan=False,
                         lims=[1/minTrialLength, data.samplerate/2], dims=(None,))
        except Exception as exc:
            raise exc
        foi = np.array(foi)
        foi.sort()
        foi = foi[foi <= freqs.max()]
        foi = foi[foi >= freqs.min()]
        foi = freqs[np.unique(np.searchsorted(freqs, foi, side="right") - 1)]
    else:
        foi = freqs

    # FIXME: implement detrending
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
            print("<freqanalysis> WARNING: `polyorder` keyword will be ignored " +
                  "since `polyremoval` is `False`!")

    # Prepare keyword dict for logging (use `lcls` to get actually provided 
    # keyword values, not defaults set in here)
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
    
    # Ensure consistency of method-specific options
    if method == "mtmfft":
        
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

        # Warn the user in case `tapsmofrq` has no effect
        if tapsmofrq is not None and taper.__name__ != "dpss":
            print("<freqanalysis> WARNING: `tapsmofrq` is only used if `taper` is `dpss`!")
            
        # Update `log_dct` w/method-specific options (use `lcls` to get actually
        # provided keyword values, not defaults set in here)
        log_dct["taper"] = lcls["taper"]
        log_dct["tapsmofrq"] = lcls["tapsmofrq"]
        
        # Set up compute-kernel
        specestMethod = MultiTaperFFT(1/data.samplerate, 
                                      timeAxis, 
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

    elif method == "wavelet":
        
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

        # Set up compute-kernel
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

    # Detect if dask client is running and set `parallel` keyword accordingly
    if __dask__:
        try:
            dd.get_client()
            use_dask = True
        except ValueError:
            use_dask = False
    else:
        use_dask = False

    # Perform actual computation
    specestMethod.initialize(data, 
                             chan_per_worker=kwargs.get("chan_per_worker"),
                             keeptrials=keeptrials)
    specestMethod.compute(data, out, parallel=use_dask, log_dict=log_dct)

    # Either return newly created output container or simply quit
    return out if new_out else None
