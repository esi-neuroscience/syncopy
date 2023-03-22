# -*- coding: utf-8 -*-
#
# Update trialdefintions/time axis of Syncopy data objects
# akin to ft_redefinetrial
#

# Builtin/3rd party package imports
import numpy as np
from numbers import Number

# Local imports
import syncopy as spy
from syncopy.shared.kwarg_decorators import unwrap_cfg
from syncopy.shared.parsers import data_parser, array_parser, scalar_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError, SPYError
from syncopy.shared.tools import get_defaults, get_frontend_cfg

__all__ = ["redefinetrial"]


@unwrap_cfg
def redefinetrial(data_obj, trials=None, minlength=None,
                  offset=None, toilim=None,
                  begsample=None, endsample=None, trl=None):
    """
    This function allows you to adjust the time axis of your data, i.e. to
    change from stimulus-locked to response-locked. Furthermore, it allows
    you to select a time window of interest, or to resegment your long trials
    into shorter fragments.

    Parameters
    ----------
    data_obj : Syncopy data object (:class:`BaseData`-like)
    trials : list or ndarray
        List of integers representing trial numbers to be selected
    minlength : float or str
        Minimum length of trials in seconds, can be 'maxperlen'. Trials which
        are shorter get thrown out.
    toilim : [begin, end]
        specify latency window in seconds, to cut out a time window of interest from each trial.
    offset : int or Mx1 ndarray
        Realign the time axes of all trials to a new reference time point (i.e. change the definition of t=0).
        Expressed in samples relative to current t=0.
    begsample : int or Mx1 ndarray of ints
        Specify the begin sample for each trial (see also `endsample`).
        Expressed in samples relative to the start of the input trial.
    endsample : int or Mx1 ndarray of ints
        Specify the end sample for each trial (see also `begsample`).
        Expressed in samples relative to the start of the input trial.
    trl : Mx3 ndarray
        New trial definition.
        [start, stop, trigger_offset] sample indices for `M` trials

    Returns
    -------
    ret_obj : Syncopy data object (:class:`BaseData`-like))

    Notes
    -----
    This is a compatibility function, mimicking ``ft_redefinetrial``. However,
    if selected sample ranges (via ``toilim`` or ``trl``) are (partially) outside
    of the available data, an error is thrown. This is different to FieldTrips' implementation,
    where missing data is filled up with NaNs.

    See also
    --------
    definetrial : :func:`~syncopy.definetrial`
        Manipulate the trial definition
    selectdata : :func:`~syncopy.selectdata`
        Select along general attributes like channel, frequency, etc.
    `FieldTrip's redefinetrial <https://https://github.com/fieldtrip/fieldtrip/blob/master/ft_redefinetrial.m>`
    """

    # Start by vetting input object
    data_parser(data_obj, varname="data_obj", empty=False)

    defaults = get_defaults(redefinetrial)
    lcls = locals()
    new_cfg = get_frontend_cfg(defaults, lcls, kwargs={})

    # -- sort out mutually exclusive parameters --

    vals = [new_cfg[par] for par in ['minlength', 'toilim', 'begsample', 'trl']]
    if vals.count(None) < 3:
        msg = "either `minlength` or `begsample`/`endsample` or `trl` or `toilim`"
        raise SPYError("Incompatible input arguments, " + msg)
    # now we made sure only one of the 4 parameters above is set.

    # total number of samples
    scount = data_obj.data.shape[data_obj._stackingDim]

    # -- first select trials --

    if trials is not None:
        array_parser(trials, dims=1, ntype=int)
        ret_obj = spy.selectdata(data_obj, trials=trials)
    else:
        # copy in any case, that's the difference to definetrial
        ret_obj = spy.copy(data_obj)

    # -- apply latency window --

    if toilim is not None:
        array_parser(toilim, dims=(2,))
        # use latency selection mechanic
        ret_obj = spy.selectdata(ret_obj, latency=toilim)

    elif minlength is not None:

        scalar_parser(minlength, varname='minlength', lims=[0, np.inf])

        min_samples = int(minlength * data_obj.samplerate)
        trl_sel = []
        for trl_idx, trial in enumerate(ret_obj.trials):
            nSamples = trial.shape[data_obj._stackingDim]
            if nSamples >= min_samples:
                trl_sel.append(trl_idx)

        spy.log(f"discarding {len(data_obj.trials) - len(trl_sel)} trials", level='INFO',
                caller='redefinetrial')

        if len(trl_sel) == 0:
            spy.log("No trial fits the desired `minlength`, returning empty object!",
                    caller='redefinetrial')
            return data_obj.__class__()

        ret_obj = spy.selectdata(ret_obj, trials=trl_sel)

    # helper variable
    new_trldef = ret_obj.trialdefinition

    # -- OR manipulate sampleinfo --

    if new_cfg['trl'] is not None:
        vals = [new_cfg[par] for par in ['begsample', 'endsample', 'offset']]
        if vals.count(None) < 3:
            msg = ("either complete trialdefinition `trl` or  "
                   "`begsample`/`endsample` and `offset`")
            raise SPYError("Incompatible input arguments, " + msg)

        # accepts also lists
        array_parser(trl, varname="trl")
        trl = np.array(trl)

        # to allow simple single trial definitions
        if trl.ndim == 1:
            trl = trl[None, :]
        if trl.ndim != 2:
            lgl = "2-dimensional array"
            act = f"{trl.ndim} array"
            raise SPYValueError(lgl, 'trl', act)

        new_trldef = trl

        # selecting trials and applying a new trialdefinition in one go
        # is rather dangerous, but possible..

    elif begsample is not None or endsample is not None:
        vals = [new_cfg[par] for par in ['begsample', 'endsample']]
        if vals.count(None) != 0:
            lgl = "both `begsample` and `endsample`"
            act = f"got [{begsample}, {endsample}]"
            raise SPYValueError(lgl, 'begsample/endsample', act)

        try:
            begsample = np.array(begsample, dtype=int)
        except ValueError:
            raise SPYTypeError(begsample, 'begsample', "integer number or array")

        try:
            endsample = np.array(endsample, dtype=int)
        except ValueError:
            raise SPYTypeError(endsample, 'endsample', "scalar or array")

        if np.any(begsample < 0):
            lgl = "integers >= 0"
            act = "relative `begsample` < 0"
            raise SPYValueError(lgl, 'begsample', act)

        if begsample.size != 1 and begsample.size != len(new_trldef):
            raise SPYValueError(f"scalar or array of length {len(new_trldef)}", "begsample/endsample",
                                "wrong sized `begsample`")

        if begsample.size != endsample.size:
            raise SPYValueError("same sizes for `begsample/endsample`", '',
                                "different sizes")

        if np.any(new_trldef[:, 0] + endsample > scount):
            lgl = f"integers < {int(scount - new_trldef[:, 0].max())}"
            act = "out of range"
            raise SPYValueError(lgl, 'endsample', act)

        # this also catches negative endsample
        if np.any(endsample - begsample < 0):
            raise SPYValueError("endsample > begsample", "begsample/endsample",
                                "endsample < begsample")

        # construct new trialdefinition
        new_trldef[:, 1] = new_trldef[:, 0] + endsample
        new_trldef[:, 0] += begsample

    # -- manipulate offset --

    if isinstance(offset, Number):
        new_trldef[:, 2] = np.ones(len(new_trldef)) * offset

    elif isinstance(offset, np.ndarray):
        if len(offset) != len(new_trldef):
            lgl = f"array of length {len(new_trldef)}"
            act = f"array of length {len(offset)}"
            raise SPYValueError(lgl, 'offset', act)
        new_trldef[:, 2] = offset
    elif offset is None:
        pass
    else:
        raise SPYTypeError(offset, 'offset', "scalar, array or None")

    # -- apply (new) trialdefinition --

    array_parser(new_trldef, varname="trl", dims=2)

    array_parser(new_trldef[:, :2], varname="trl", dims=(None, 2),
                 hasnan=False, hasinf=False, ntype="int_like", lims=[0, scount])

    # apply new trialdefinition and be done with it
    spy.definetrial(ret_obj, trialdefinition=new_trldef)

    # Attach potential older cfg's from the input
    # to support chained frontend calls.
    ret_obj.cfg.update(data_obj.cfg)

    # Attach frontend parameters for replay.
    ret_obj.cfg.update({'redefinetrial': new_cfg})

    return ret_obj
