# -*- coding: utf-8 -*-
#
# Syncopy down-/resampling frontend
#

# Builtin/3rd party package imports
import numpy as np

# Syncopy imports
from syncopy import AnalogData
from syncopy.shared.parsers import data_parser, scalar_parser

from syncopy.shared.tools import get_defaults, get_frontend_cfg
from syncopy.shared.errors import SPYValueError, SPYWarning

from syncopy.shared.kwarg_decorators import (
    unwrap_cfg,
    unwrap_select,
    detect_parallel_client,
)
from syncopy.shared.input_processors import check_passed_kwargs

from .compRoutines import Downsample, Resample, SincFiltering

availableMethods = ("downsample", "resample")


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def resampledata(data,
                 resamplefs=1.,
                 method="resample",
                 lpfreq=None,
                 order=None,
                 **kwargs):
    """
    Performs resampling or downsampling of :class:`~syncopy.AnalogData` objects,
    representing uniformly sampled time series data.

    Two methods are supported:

    "downsample" : Take every nth sample
        The new sampling rate `resamplefs` must be an integer division of
        the old sampling rate, e. g., 500Hz to 250Hz.
        NOTE: No anti-aliasing filtering is performed before downsampling,
        it is strongly recommended to apply a low-pass filter
        via explicitly setting `lpfreq` to the new Nyquist frequency
        (`resamplefs / 2`) as cut-off.
        Alternatively filter the data with :func:`~syncopy.preprocessing` beforehand.

    "resample" : Resample to a new sampling rate
        The new sampling rate `resamplefs` can be any (rational) fraction
        of the original sampling rate (`data.samplerate`). Automatic
        anti-aliasing FIRWS filtering with the new Nyquist frequency
        is performed before resampling. Optionally set `lpfreq` in Hz
        for manual control over the low-pass filtering.

    Parameters
    ----------
    data : `~syncopy.AnalogData`
        A non-empty Syncopy :class:`~syncopy.AnalogData` object
    resamplefs : float
        The new sampling rate, needs to be an integer division
        of the original sampling rate for `method='downsample'`
    lpfreq : None or float, optional
        Leave at `None` for standard anti-alias filtering with
        the new Nyquist for `method='resample'` or set explicitly in Hz
    order : None or int, optional
        Order (length) of the firws anti-aliasing filter
        The default `None` will create a filter with a length of 1000 samples

    Returns
    -------
    resampled : `~syncopy.AnalogData`
        The resampled dataset with the same shape and dimord as the input `data`

    Examples
    --------
    In the following `adata` is an instance of :class:`~syncopy.AnalogData`
    with a samplerate of 2kHz.

    Downsample (decimate) to 1kHz without low-pass filtering:

    >>> downsampled = spy.resampledata(adata, method='downsample', resamplefs=1000)

    Repeat, but this time remove aliases via explicit low-pass filter:

    >>> downsampled = spy.resampledata(adata, method='downsample', resamplefs=1000, lpfreq=500)

    Resample to 600Hz, low-pass filtering to new Nyquist is implicit:

    >>> resampled = spy.resampledata(adata, resamplefs=600)
    """

    # -- Basic input parsing --

    if method not in availableMethods:
        lgl = "'" + "or '".join(opt + "' " for opt in availableMethods)
        raise SPYValueError(legal=lgl, varname="method", actual=method)

    # Make sure our one mandatory input object can be processed
    try:
        data_parser(
            data, varname="data", dataclass="AnalogData", writable=None, empty=False
        )
    except Exception as exc:
        raise exc
    timeAxis = data.dimord.index("time")

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

    # Get everything of interest in local namespace
    defaults = get_defaults(resampledata)
    lcls = locals()
    # check for ineffective additional kwargs
    check_passed_kwargs(lcls, defaults, frontend_name="resampledata")

    new_cfg = get_frontend_cfg(defaults, lcls, kwargs)

    # check resampling frequency
    scalar_parser(resamplefs, varname="resamplefs", lims=[1, data.samplerate])

    # filter order
    if order is not None:
        scalar_parser(order, varname="order", lims=[0, np.inf], ntype="int_like")
        if order < 100:
            msg = ("You have chosen an anti-alias filter of very low "
                   f"`order={order}`, expect a slow roll-off!"
                   )
            SPYWarning(msg)

    # set default
    else:
        order = int(lenTrials.min()) if lenTrials.min() < 1000 else 1000

    # check for anti-alias low-pass filter settings
    # minimum requirement: new Nyquist limit
    if lpfreq is not None:
        scalar_parser(lpfreq, varname="lpfreq", lims=[0, resamplefs / 2])

    # -- downsampling --
    if method == "downsample":

        if data.samplerate % resamplefs != 0:
            lgl = (
                "integer division of the original sampling rate "
                "for `method='downsample'`"
            )
            raise SPYValueError(lgl, varname="resamplefs", actual=resamplefs)

        # explicit low-pass filtering on the fly
        if lpfreq is not None:
            AntiAliasFilter = SincFiltering(
                samplerate=data.samplerate,
                filter_type='lp',
                freq=lpfreq,
                order=order,
                direction='twopass',
                timeAxis=timeAxis,
            )
            # keyword dict for logging
            aa_log_dict = {"filter_type": 'lp',
                           "lpfreq": lpfreq,
                           "order": order,
                           "direction": 'twopass'}

        else:
            AntiAliasFilter = None

        resampleMethod = Downsample(
            samplerate=data.samplerate, new_samplerate=resamplefs, timeAxis=timeAxis
        )
        # keyword dict for logging
        log_dict = {"method": method,
                    "resamplefs": resamplefs,
                    "origfs": data.samplerate}

    # -- resampling --
    elif method == "resample":

        if data.samplerate % resamplefs == 0:
            msg = ("New sampling rate is integeger division of the "
                   "original sampling rate, consider using `method='downsample'`"
                   )
            SPYWarning(msg)

        # has anti-alias filtering included
        # configured by lpfreq and order
        resampleMethod = Resample(
            samplerate=data.samplerate,
            new_samplerate=resamplefs,
            lpfreq=lpfreq,
            order=order,
            timeAxis=timeAxis
        )
        # keyword dict for logging
        log_dict = {"method": method,
                    "resamplefs": resamplefs,
                    "origfs": data.samplerate,
                    "lpfreq": lpfreq,
                    "order": order}

    # ------------------------------------
    # Call the chosen ComputationalRoutine
    # ------------------------------------

    resampled = AnalogData(dimord=data.dimord)

    if method == 'downsample' and AntiAliasFilter is not None:
        filtered = AnalogData(dimord=data.dimord)
        AntiAliasFilter.initialize(
            data,
            filtered._stackingDim,
            chan_per_worker=kwargs.get("chan_per_worker"),
            keeptrials=True
        )

        AntiAliasFilter.compute(data,
                                filtered,
                                parallel=kwargs.get("parallel"),
                                log_dict=aa_log_dict)
        target = filtered
    else:
        target = data  # just rebinds the name

    resampleMethod.initialize(
        target,
        resampled._stackingDim,
        chan_per_worker=kwargs.get("chan_per_worker"),
        keeptrials=True,
    )
    resampleMethod.compute(
        target, resampled, parallel=kwargs.get("parallel"), log_dict=log_dict
    )

    resampled.cfg.update(data.cfg)
    resampled.cfg.update({'resampledata': new_cfg})
    return resampled
