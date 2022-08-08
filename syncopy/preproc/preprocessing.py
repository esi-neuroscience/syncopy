# -*- coding: utf-8 -*-
#
# Syncopy preprocessing frontend
#

# Builtin/3rd party package imports
import numpy as np

# Syncopy imports
from syncopy import AnalogData
from syncopy.shared.parsers import data_parser, scalar_parser, array_parser
from syncopy.shared.tools import get_defaults, get_frontend_cfg
from syncopy.shared.errors import SPYValueError, SPYInfo
from syncopy.shared.kwarg_decorators import (
    unwrap_cfg,
    unwrap_select,
    detect_parallel_client,
)
from syncopy.shared.input_processors import (
    check_effective_parameters,
    check_passed_kwargs,
)

from .compRoutines import (
    ButFiltering,
    SincFiltering,
    Rectify,
    Hilbert,
    Detrending,
    Standardize
)

availableFilters = ("but", "firws")
availableFilterTypes = ("lp", "hp", "bp", "bs")
availableDirections = ("twopass", "onepass", "onepass-minphase")
availableWindows = ("hamming", "hann", "blackman")

hilbert_outputs = {"abs", "complex", "real", "imag", "absreal", "absimag", "angle"}


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def preprocessing(
    data,
    filter_class="but",
    filter_type="lp",
    freq=None,
    order=None,
    direction='twopass',
    window="hamming",
    polyremoval=None,
    zscore=False,
    rectify=False,
    hilbert=False,
    **kwargs,
):
    """
    Preprocessing of time continuous raw data with IIR and FIR filters

    Parameters
    ----------
    data : `~syncopy.AnalogData`
        A non-empty Syncopy :class:`~syncopy.AnalogData` object
    filter_class : {'but', 'firws'} or None
        Butterworth (IIR) or windowed sinc (FIR)
        Set to `None` to disable filtering altogether
    filter_type : {'lp', 'hp', 'bp', 'bs'}, optional
        Select type of filter, either low-pass `'lp'`,
        high-pass `'hp'`, band-pass `'bp'` or band-stop (Notch) `'bs'`.
    freq : float or array_like
        Cut-off frequency for low- and high-pass filters or sequence
        of two frequencies for band-stop and band-pass filter.
    order : int, optional
        Order of the filter, default is 4 for `filter_class='but'` and
        1000 for filter_class='firws'.
        Higher orders yield a sharper transition width
        or less 'roll off' of the filter, but are more computationally expensive.
    direction : {'twopass', 'onepass', 'onepass-minphase'}
       Filter direction:
       `'twopass'` - zero-phase forward and reverse filter, IIR and FIR
       `'onepass'` - forward filter, introduces group delays for IIR, zerophase for FIR
       `'onepass-minphase' - forward causal/minimum phase filter, FIR only
    window : {"hamming", "hann", "blackman"}, optional
        The type of window to use for the FIR filter
    polyremoval : int or None, optional
        Order of polynomial used for de-trending data in the time domain prior
        to filtering. A value of 0 corresponds to subtracting the mean
        ("de-meaning"), ``polyremoval = 1`` removes linear trends (subtracting the
        least squares fit of a linear polynomial).
    zscore : bool, optional
        Set to `True` to individually standardize all signals
    rectify : bool, optional
        Set to `True` to rectify (after filtering)
    hilbert : None or one of {'abs', 'complex', 'real', 'imag', 'absreal', 'absimag', 'angle'}
        Choose one of the supported output types to perform
        Hilbert transformation after filtering. Set to `'angle'` to return the phase.

    Returns
    -------
    filtered : `~syncopy.AnalogData`
        The filtered dataset with the same shape and dimord as the input `data`

    Examples
    --------
    In the following `adata` is an instance of :class:`~syncopy.AnalogData`

    Low-pass filtering with a Butterworth filter and a cut-off of 100Hz:

    >>> spy.preprocessing(adata, filter_class='but', filter_type='lp', freq=100)

    Notch (band-stop) filtering with a FIR filter of order 2000 around 50Hz:

    >>> spy.preprocessing(adata, filter_class='firws', filter_type='bs', freq=[49, 51], order=2000)

    Remove linear trends and standardize but no filtering:

    >>> spy.preprocessing(adata, filter_class=None, polyremoval=1, zscore=True)

    """

    # -- Basic input parsing --

    # Make sure our one mandatory input object can be processed
    try:
        data_parser(
            data, varname="data", dataclass="AnalogData", writable=None, empty=False
        )
    except Exception as exc:
        raise exc
    timeAxis = data.dimord.index("time")

    # Get everything of interest in local namespace
    defaults = get_defaults(preprocessing)
    lcls = locals()
    # check for ineffective additional kwargs
    check_passed_kwargs(lcls, defaults, frontend_name="preprocessing")

    new_cfg = get_frontend_cfg(defaults, lcls, kwargs)

    # filter specific settings
    if filter_class is not None:
        if filter_class not in availableFilters:
            lgl = "'" + "or '".join(opt + "' " for opt in availableFilters)
            raise SPYValueError(legal=lgl, varname="filter_class", actual=filter_class)

        if not isinstance(filter_type, str) or filter_type not in availableFilterTypes:
            lgl = f"one of {availableFilterTypes}"
            act = filter_type
            raise SPYValueError(lgl, "filter_type", filter_type)

        # check `freq` setting
        if filter_type in ("lp", "hp"):
            scalar_parser(freq, varname="freq", lims=[0, data.samplerate / 2])
        elif filter_type in ("bp", "bs"):
            array_parser(
                freq,
                varname="freq",
                hasinf=False,
                hasnan=False,
                lims=[0, data.samplerate / 2],
                dims=(2,),
            )
            if freq[0] == freq[1]:
                lgl = "two different frequencies"
                raise SPYValueError(lgl, varname="freq", actual=freq)
            freq = np.sort(freq)

        # -- here the defaults are filter specific and get set later --

        # filter order
        if order is not None:
            scalar_parser(order, varname="order", lims=[0, np.inf], ntype="int_like")

    # check if anything else was requested
    elif filter_class is None and polyremoval is None and zscore is False:
        lgl = "a preprocessing method"
        act = "neither filtering, detrending or zscore requested"
        raise SPYValueError(lgl, "filter_class/polyremoval/zscore", act)

    # check polyremoval
    if polyremoval is not None:
        scalar_parser(polyremoval, varname="polyremoval", ntype="int_like", lims=[0, 1])

    if not isinstance(zscore, bool):
        raise SPYValueError("either `True` or `False`", varname="zscore", actual=zscore)

    if not isinstance(rectify, bool):
        raise SPYValueError("either `True` or `False`", varname="rectify", actual=rectify)

    # -- get trial info

    # if a subset selection is present
    # get sampleinfo and check for equidistancy
    if data.selection is not None:
        sinfo = data.selection.trialdefinition[:, :2]
        # user picked discrete set of time points
        if isinstance(data.selection.time[0], list):
            lgl = "equidistant time points (toi) or time slice (toilim)"
            actual = "non-equidistant set of time points"
            raise SPYValueError(legal=lgl, varname="select", actual=actual)
    else:
        sinfo = data.sampleinfo
    lenTrials = np.diff(sinfo).squeeze()

    # check for equidistant sampling as needed for filtering
    # FIXME: could be too slow, see #259
    # if not all([np.allclose(np.diff(time), 1 / data.samplerate) for time in data.time]):
    #     lgl = "equidistant sampling in time"
    #     act = "non-equidistant sampling"
    #     raise SPYValueError(lgl, varname="data", actual=act)

    # -- post processing
    if rectify and hilbert:
        lgl = "either rectification or Hilbert transform"
        raise SPYValueError(lgl, varname="rectify/hilbert", actual=(rectify, hilbert))

    # `hilbert` acts both as a switch and a parameter to set the output (like in FT)
    if hilbert:
        if hilbert not in hilbert_outputs:
            lgl = f"one of {hilbert_outputs}"
            raise SPYValueError(lgl, varname="hilbert", actual=hilbert)

    # -- Method calls

    # Prepare keyword dict for logging
    log_dict = {
        "polyremoval": polyremoval,
        "zscore": zscore
    }

    # pre-processing
    if zscore:

        std_data = AnalogData(dimord=data.dimord)
        stdCR = Standardize(polyremoval=polyremoval, timeAxis=timeAxis)
        stdCR.initialize(
            data,
            data._stackingDim,
            chan_per_worker=kwargs.get("chan_per_worker"),
            keeptrials=True,
        )

        stdCR.compute(
            data, std_data, parallel=kwargs.get("parallel"), log_dict=log_dict
        )

        data = std_data

    if filter_class == "but":

        if window != defaults["window"] and window is not None:
            lgl = "no `window` setting for IIR filtering"
            act = window
            raise SPYValueError(lgl, "window", act)

        # set filter specific defaults here
        if direction is None:
            direction = "twopass"
            msg = f"Setting default direction for IIR filter to '{direction}'"
            SPYInfo(msg)
        elif not isinstance(direction, str) or direction not in ("onepass", "twopass"):
            lgl = "'" + "or '".join(opt + "' " for opt in ("onepass", "twopass"))
            raise SPYValueError(legal=lgl, varname="direction", actual=direction)

        if order is None:
            order = 4
            msg = f"Setting default order for IIR filter to {order}"
            SPYInfo(msg)

        log_dict["order"] = order
        log_dict["direction"] = direction
        log_dict["filter_class"] = filter_class
        log_dict["filter_type"] = filter_type
        log_dict["freq"] = freq

        check_effective_parameters(
            ButFiltering, defaults, lcls, besides=("hilbert", "rectify",
                                                   "zscore")
        )

        filterMethod = ButFiltering(
            samplerate=data.samplerate,
            filter_type=filter_type,
            freq=freq,
            order=order,
            direction=direction,
            polyremoval=polyremoval,
            timeAxis=timeAxis,
        )

    elif filter_class == "firws":

        if window not in availableWindows:
            lgl = "'" + "or '".join(opt + "' " for opt in availableWindows)
            raise SPYValueError(legal=lgl, varname="window", actual=window)

        # set filter specific defaults here
        if direction is None:
            direction = "onepass"
            msg = f"Setting default direction for FIR filter to '{direction}'"
            SPYInfo(msg)
        elif not isinstance(direction, str) or direction not in availableDirections:
            lgl = "'" + "or '".join(opt + "' " for opt in availableDirections)
            raise SPYValueError(legal=lgl, varname="direction", actual=direction)

        if order is None:
            order = int(lenTrials.min()) if lenTrials.min() < 1000 else 1000
            msg = f"Setting order for FIR filter to {order}"
            SPYInfo(msg)

        log_dict["order"] = order
        log_dict["direction"] = direction
        log_dict["filter_class"] = filter_class
        log_dict["filter_type"] = filter_type
        log_dict["freq"] = freq

        check_effective_parameters(
            SincFiltering,
            defaults,
            lcls,
            besides=["filter_class", "hilbert", "rectify", "zscore"],
        )

        filterMethod = SincFiltering(
            samplerate=data.samplerate,
            filter_type=filter_type,
            freq=freq,
            order=order,
            window=window,
            direction=direction,
            polyremoval=polyremoval,
            timeAxis=timeAxis,
        )

    # only detrending
    elif filter_class is None and polyremoval is not None and zscore is False:

        check_effective_parameters(
            Detrending,
            defaults,
            lcls,
            besides=["filter_class", "hilbert", "rectify", "zscore"],
        )

        # not really a `filterMethod` though..
        filterMethod = Detrending(polyremoval=polyremoval, timeAxis=timeAxis)
    # only zscoring
    else:
        filterMethod = None
    # -------------------------------------------
    # Call the chosen filter ComputationalRoutine
    # -------------------------------------------

    # unlikely but possible: post-processing without filtering
    if filterMethod is None:
        filtered = data
    else:
        filtered = AnalogData(dimord=data.dimord)
        # Perform actual computation
        filterMethod.initialize(
            data,
            data._stackingDim,
            chan_per_worker=kwargs.get("chan_per_worker"),
            keeptrials=True,
        )
        filterMethod.compute(
            data, filtered, parallel=kwargs.get("parallel"), log_dict=log_dict
        )

    # -- check for post-processing flags --

    if rectify:
        log_dict["rectify"] = rectify
        rectified = AnalogData(dimord=data.dimord)
        rectCR = Rectify()
        rectCR.initialize(
            filtered,
            data._stackingDim,
            chan_per_worker=kwargs.get("chan_per_worker"),
            keeptrials=True,
        )
        rectCR.compute(
            filtered, rectified, parallel=kwargs.get("parallel"), log_dict=log_dict
        )
        del filtered
        rectified.cfg.update(data.cfg)
        rectified.cfg.update({'preprocessing': new_cfg})
        return rectified

    elif hilbert:
        log_dict["hilbert"] = hilbert
        htrafo = AnalogData(dimord=data.dimord)
        hilbertCR = Hilbert(output=hilbert, timeAxis=timeAxis)
        hilbertCR.initialize(
            filtered,
            data._stackingDim,
            chan_per_worker=kwargs.get("chan_per_worker"),
            keeptrials=True,
        )
        hilbertCR.compute(
            filtered, htrafo, parallel=kwargs.get("parallel"), log_dict=log_dict
        )
        del filtered
        htrafo.cfg.update(data.cfg)
        htrafo.cfg.update({'preprocessing': new_cfg})
        return htrafo

    # no post-processing
    else:
        # attach potential older cfg's from the input
        # to support chained frontend calls..
        filtered.cfg.update(data.cfg)
        filtered.cfg.update({'preprocessing': new_cfg})
        return filtered
