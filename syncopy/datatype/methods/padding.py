# -*- coding: utf-8 -*-
#
# Pad Syncopy data objects
#

# Builtin/3rd party package imports
import numpy as np

# Local imports
from syncopy.datatype.continuous_data import AnalogData
from syncopy.shared.computational_routine import ComputationalRoutine
from syncopy.shared.kwarg_decorators import process_io
from syncopy.shared.parsers import data_parser, array_parser, scalar_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError, SPYWarning
from syncopy.shared.kwarg_decorators import unwrap_cfg, unwrap_select, detect_parallel_client

__all__ = ["padding"]


@unwrap_cfg
@unwrap_select
@detect_parallel_client
def padding(data, padtype, pad="absolute", padlength=None, prepadlength=None,
            postpadlength=None, unit="samples", create_new=True, **kwargs):
    """
    Perform data padding on Syncopy object or :class:`numpy.ndarray`

    **Usage Summary**

    Depending on the value of `pad` the following padding length specifications
    are supported:

    +------------+----------------------+---------------+----------------------+----------------------+
    | `pad`      | `data`               | `padlength`   | `prepadlength`       | `postpadlength`      |
    +============+======================+===============+======================+======================+
    | 'absolute' | Syncopy object/array | number        | `None`/`bool`        | `None`/`bool`        |
    +------------+----------------------+---------------+----------------------+----------------------+
    | 'relative' | Syncopy object/array | number/`None` | number/`None`/`bool` | number/`None`/`bool` |
    +------------+----------------------+---------------+----------------------+----------------------+
    | 'maxlen'   | Syncopy object       | `None`/`bool` | `None`/`bool`        | `None`/`bool`        |
    +------------+----------------------+---------------+----------------------+----------------------+
    | 'nextpow2' | Syncopy object/array | `None`/`bool` | `None`/`bool`        | `None`/`bool`        |
    +------------+----------------------+---------------+----------------------+----------------------+

    * `data` can be either a Syncopy object containing multiple trials or a
      :class:`numpy.ndarray` representing a single trial
    * (pre/post)padlength: can be either `None`, `True`/`False` or a positive
      number: if `True` indicates where to pad, e.g., by using ``pad =
      'maxlen'`` and  ``prepadlength = True``, `data` is padded at the beginning
      of each trial. **Only** if `pad` is 'relative' are scalar values supported
      for `prepadlength` and `postpadlength`
    * ``pad = 'absolute'``: pad to desired absolute length, e.g., by using ``pad
      = 5`` and ``unit = 'time'`` all trials are (if necessary) padded to 5s
      length. Here, `padlength` **has** to be provided, `prepadlength` and
      `postpadlength` can be `None` or `True`/`False`
    * ``pad = 'relative'``: pad by provided `padlength`, e.g., by using
      ``padlength = 20`` and ``unit = 'samples'``, 20 samples are padded
      symmetrically around (before and after) each trial. Use ``padlength = 20``
      and ``prepadlength = True`` **or** directly ``prepadlength = 20`` to pad
      before each trial. Here, at least one of `padlength`, `prepadlength` or
      `postpadlength` **has** to be provided.
    * ``pad = 'maxlen'``: (only valid for **Syncopy objects**) pad up to maximal
      trial length found in `data`. All lengths have to be either Boolean
      indicating padding location or `None` (if all are `None`, symmetric
      padding is performed)
    * ``pad = 'nextpow2'``: pad each trial up to closest power of two. All
      lengths have to be either Boolean indicating padding location or `None`
      (if all are `None`, symmetric padding is performed)

    Full documentation below.

    Parameters
    ----------
    data : Syncopy object or :class:`numpy.ndarray`
        Non-empty Syncopy data object or array representing numeric data to be
        padded. **NOTE**: if `data` is a :class:`numpy.ndarray`, it is assumed
        that it represents recordings from only a single trial, where its first
        axis corresponds to time. In other words, `data` is a
        'time'-by-'channel' array such that its rows reflect samples and its
        columns represent channels. If `data` is a Syncopy object, trial
        information and dimensional order are fetched from `data.trials` and
        `data.dimord`, respectively.
    padtype : str
        Padding value(s) to be used. Available options are:

        * 'zero' : pad using zeros
        * 'nan' : pad using `np.nan`'s
        * 'mean' : pad with by-channel mean value across each trial
        * 'localmean' : pad with by-channel mean value using only `padlength` or
          `prepadlength`/`postpadlength` number of boundary-entries for averaging
        * 'edge' : pad with trial-boundary values
        * 'mirror' : pad with reflections of trial-boundary values

    pad : str
        Padding mode to be used. Available options are:

        * 'absolute' : pad each trial to achieve a desired absolute length such
          that all trials have identical length post padding. If `pad` is `absolute`
          a `padlength` **has** to be provided, `prepadlength` and `postpadlength`
          may be `True` or `False`, respectively (see Examples for details).
        * 'relative' : pad each trial by provided `padlength` such that all trials
          are extended by the same amount regardless of their original lengths.
          If `pad` is `relative`, `prepadlength` and `postpadlength` can either
          be specified directly (using numerical values) or implicitly by only
          providing `padlength` and setting `prepadlength` and `postpadlength`
          to `True` or `False`, respectively (see Examples for details). If `pad`
          is `relative` at least one of `padlength`, `prepadlength` or `postpadlength`
          **has** to be provided.
        * 'maxlen' : only usable if `data` is a Syncopy object. If `pad` is
          'maxlen' all trials are padded to achieve the length of the longest
          trial in `data`, i.e., post padding, all trials have the same length,
          which equals the size of the longest trial pre-padding. For
          ``pad = 'maxlen'``, `padlength`, `prepadlength` as well as `postpadlength`
          have to be either Boolean or `None` indicating the preferred padding
          location (pre-trial, post-trial or symmetrically pre- and post-trial).
          If all are `None`, symmetric padding is performed (see Examples for
          details).
        * 'nextpow2' : pad each trial to achieve a length equals the closest power
          of two of its original length. For ``pad = 'nextpow2'``, `padlength`,
          `prepadlength` as well as `postpadlength` have to be either Boolean
          or `None` indicating the preferred padding location (pre-trial, post-trial
          or symmetrically pre- and post-trial). If all are `None`, symmetric
          padding is performed (see Examples for details).

    padlength : None, bool or positive scalar
        Length to be padded to `data` (if `padlength` is scalar-valued) or
        padding location (if `padlength` is Boolean). Depending on the value of
        `pad`, `padlength` can be used to pre-pend (if `padlength` is a positive
        number and `prepadlength` is `True`) or append trials (if `padlength` is
        a positive number and `postpadlength` is `True`). If neither
        `prepadlength` nor `postpadlength` are specified (i.e, both are `None`),
        symmetric pre- and post-trial padding is performed (i.e., ``0.5 * padlength``
        before and after each trial - note that odd sample counts are rounded downward
        to the nearest even integer). If ``unit = 'time'``, `padlength` is assumed
        to be given in seconds, otherwise (``unit = 'samples'``), `padlength` is
        interpreted as sample-count. Note that only ``pad = 'relative'`` and
        ``pad = 'absolute'`` support numeric values of `padlength`.
    prepadlength : None, bool or positive scalar
        Length to be pre-pended before each trial (if `prepadlength` is
        scalar-valued) or pre-padding flag (if `prepadlength` is `True`). If
        `prepadlength` is `True`, pre-padding length is either directly inferred
        from `padlength` or implicitly derived from chosen padding mode defined
        by `pad`. If ``unit = 'time'``, `prepadlength` is assumed to be given in
        seconds, otherwise (``unit = 'samples'``), `prepadlength` is interpreted
        as sample-count. Note that only ``pad = 'relative'`` supports numeric
        values of `prepadlength`.
    postpadlength : None, bool or positive scalar
        Length to be appended after each trial (if `postpadlength` is
        scalar-valued) or post-padding flag (if `postpadlength` is `True`). If
        `postpadlength` is `True`, post-padding length is either directly inferred
        from `padlength` or implicitly derived from chosen padding mode defined
        by `pad`. If ``unit = 'time'``, `postpadlength` is assumed to be given in
        seconds, otherwise (``unit = 'samples'``), `postpadlength` is interpreted
        as sample-count. Note that only ``pad = 'relative'`` supports numeric
        values of `postpadlength`.
    unit : str
        Unit of numerical values given by `padlength` and/or `prepadlength`
        and/or `postpadlength`. If ``unit = 'time'``, `padlength`,
        `prepadlength`, and `postpadlength` are assumed to be given in seconds,
        otherwise (``unit = 'samples'``), `padlength`, `prepadlength`, and
        `postpadlength` are interpreted as sample-counts. **Note** Providing
        padding lengths in seconds (i.e., ``unit = 'time'``) is only supported
        if `data` is a Syncopy object.
    create_new : bool
        If `True`, a padded copy of the same type as `data` is returned (a
        :class:`numpy.ndarray` or Syncopy object). If `create_new` is `False`,
        either a single dictionary (if `data` is a :class:`numpy.ndarray`) or a
        ``len(data.trials)``-long list of dictionaries (if `data` is a Syncopy
        object) with all necessary options for performing the actual padding
        operation with :func:`numpy.pad` is returned.

    Returns
    -------
    pad_dict : dict, if `data` is a :class:`numpy.ndarray` and ``create_new = False``
        Dictionary whose items contain all necessary parameters for calling
        :func:`numpy.pad` to perform the desired padding operation on `data`.
    pad_dicts : list, if `data` is a Syncopy object and ``create_new = False``
        List of dictionaries for calling :func:`numpy.pad` to perform the
        desired padding operation on all trials found in `data`.
    out : :class:`numpy.ndarray`, if `data` is a :class:`numpy.ndarray` and ``create_new = True``
        Padded version (deep copy) of `data`
    out : Syncopy object, if `data` is a Syncopy object and ``create_new = True``
        Padded version (deep copy) of `data`

    Notes
    -----
    This method emulates (and extends) FieldTrip's `ft_preproc_padding` by
    providing a convenience wrapper for NumPy's :func:`numpy.pad` that performs
    the actual heavy lifting.

    Examples
    --------
    Consider the following small array representing a toy-problem-trial of `ns`
    samples across `nc` channels:

    >>> nc = 7; ns = 30
    >>> trl = np.random.randn(ns, nc)

    We start by padding a total of 10 zeros symmetrically to `trl`

    >>> padded = spy.padding(trl, 'zero', pad='relative', padlength=10)
    >>> padded[:6, :]
    array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [-1.0866,  2.3358,  0.8758,  0.5196,  0.8049, -0.659 , -0.9173]])
    >>> padded[-6:, :]
    array([[ 0.027 ,  1.8069,  1.5249, -0.7953, -0.8933,  1.0202, -0.6862],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
       [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ]])
    >>> padded.shape
    (40, 7)

    Note that the above call is equivalent to

    >>> padded_ident = spy.padding(trl, 'zero', pad='relative', padlength=10, prepadlength=True, postpadlength=True)
    >>> np.array_equal(padded_ident, padded)
    True
    >>> padded_ident = spy.padding(trl, 'zero', pad='relative', prepadlength=5, postpadlength=5)
    >>> np.array_equal(padded_ident, padded)
    True

    Similarly,

    >>> prepad = spy.padding(trl, 'nan', pad='relative', prepadlength=10)

    is the same as

    >>> prepad_ident = spy.padding(trl, 'nan', pad='relative', padlength=10, prepadlength=True)
    >>> np.allclose(prepad, prepad_ident, equal_nan=True)
    True

    Define bogus trials on `trl` and create a dummy object with unit samplerate

    >>> tdf = np.vstack([np.arange(0, ns, 5),
                         np.arange(5, ns + 5, 5),
                         np.ones((int(ns / 5), )),
                         np.ones((int(ns / 5), )) * np.pi]).T
    >>> adata = spy.AnalogData(trl, trialdefinition=tdf, samplerate=1)

    Pad each trial to the closest power of two by appending by-trial channel
    averages. However, do not perform actual padding, but only prepare dictionaries
    of parameters to be passed on to :func:`numpy.pad`

    >>> pad_dicts = spy.padding(adata, 'mean', pad='nextpow2', postpadlength=True, create_new=False)
    >>> len(pad_dicts) == len(adata.trials)
    True
    >>> pad_dicts[0]
    {'pad_width': array([[0, 3],
        [0, 0]]), 'mode': 'mean'}

    Similarly, the following call generates a list of dictionaries preparing
    absolute padding by prepending zeros with :func:`numpy.pad`

    >>> pad_dicts = spy.padding(adata, 'zero', pad='absolute', padlength=10, prepadlength=True, create_new=False)
    >>> pad_dicts[0]
    {'pad_width': array([[5, 0],
        [0, 0]]), 'mode': 'constant', 'constant_values': 0}

    See also
    --------
    numpy.pad : fast array padding in NumPy
    """

    # Detect whether input is data object or array-like
    if any(["BaseData" in str(base) for base in data.__class__.__mro__]):
        try:
            data_parser(data, varname="data", dataclass="AnalogData",
                        empty=False)
        except Exception as exc:
            raise exc
        timeAxis = data.dimord.index("time")
        is_spydata = True
    elif data.__class__.__name__ == "FauxTrial":
        if len(data.shape) != 2:
            lgl = "two-dimensional AnalogData trial segment"
            act = "{}-dimensional trial segment"
            raise SPYValueError(legal=lgl, varname="data",
                                actual=act.format(len(data.shape)))
        timeAxis = data.dimord.index("time")
        is_spydata = False
    else:
        try:
            array_parser(data, varname="data", dims=2)
        except Exception as exc:
            raise exc
        timeAxis = 0
        is_spydata = False

    # If input is a syncopy object, fetch trial list and `sampleinfo` (thereby
    # accounting for in-place selections); to not repeat this later, save relevant
    # quantities in tmp attributes (all prefixed by `'_pad'`)
    if is_spydata:
        if data.selection is not None:
            trialList = data.selection.trial_ids
            data._pad_sinfo = np.zeros((len(trialList), 2))
            data._pad_t0 = np.zeros((len(trialList),))
            for tk, trlno in enumerate(trialList):
                trl = data._preview_trial(trlno)
                tsel = trl.idx[timeAxis]
                if isinstance(tsel, list):
                    lgl = "Syncopy AnalogData object with no or channe/trial selection"
                    raise SPYValueError(lgl, varname="data", actual=data.selection)
                else:
                    data._pad_sinfo[tk, :] = [trl.idx[timeAxis].start, trl.idx[timeAxis].stop]
            data._pad_t0[tk] = data._t0[trlno]
            data._pad_channel = data.channel[data.selection.channel]
        else:
            trialList = list(range(len(data.trials)))
            data._pad_sinfo = data.sampleinfo
            data._pad_t0 = data._t0
            data._pad_channel = data.channel

    # Ensure `create_new` is not weird
    if not isinstance(create_new, bool):
        raise SPYTypeError(create_new, varname="create_new", expected="bool")

    # Use FT-compatible options (sans FT option 'remove')
    if not isinstance(padtype, str):
        raise SPYTypeError(padtype, varname="padtype", expected="string")
    options = ["zero", "nan", "mean", "localmean", "edge", "mirror"]
    if padtype not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="padtype", actual=padtype)

    # Check `pad` and ensure we can actually perform the requested operation
    if not isinstance(pad, str):
        raise SPYTypeError(pad, varname="pad", expected="string")
    options = ["absolute", "relative", "maxlen", "nextpow2"]
    if pad not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="pad", actual=pad)
    if pad == "maxlen" and not is_spydata:
        lgl = "syncopy data object when using option 'maxlen'"
        raise SPYValueError(legal=lgl,
                            varname="pad", actual="maxlen")

    # Make sure a data object was provided if we're working with time values
    if not isinstance(unit, str):
        raise SPYTypeError(unit, varname="unit", expected="string")
    options = ["samples", "time"]
    if unit not in options:
        lgl = "'" + "or '".join(opt + "' " for opt in options)
        raise SPYValueError(legal=lgl, varname="unit", actual=unit)
    if unit == "time" and not is_spydata:
        raise SPYValueError(legal="syncopy data object when using option 'time'",
                            varname="unit", actual="time")

    # Set up dictionary for type-checking of provided padding lengths
    nt_dict = {"samples": "int_like", "time": None}

    # If we're padding up to an absolute bound or the max. length across
    # trials, compute lower bound for padding (in samples or seconds)
    if pad in ["absolute", "maxlen"]:
        if is_spydata:
            maxTrialLen = np.diff(data._pad_sinfo).max()
        else:
            maxTrialLen = data.shape[timeAxis] # if `pad="absolute" and data is array
    else:
        maxTrialLen = np.inf
    if unit == "time":
        padlim = maxTrialLen/data.samplerate
    else:
        padlim = maxTrialLen

    # To ease option processing, collect padding length keywords in dict
    plengths = {"padlength": padlength, "prepadlength": prepadlength,
                "postpadlength": postpadlength}

    # In case of relative padding, we need at least one scalar value to proceed
    if pad == "relative":

        # If `padlength = None`, pre- or post- need to be set; if `padlength`
        # is set, both pre- and post- need to be `None` or `True`/`False`.
        # After this code block, pre- and post- are guaranteed to be numeric.
        if padlength is None:
            for key in ["prepadlength", "postpadlength"]:
                if plengths[key] is not None:
                    try:
                        scalar_parser(plengths[key], varname=key, ntype=nt_dict[unit],
                                      lims=[0, np.inf])
                    except Exception as exc:
                        raise exc
                else:
                    plengths[key] = 0
        else:
            try:
                scalar_parser(padlength, varname="padlength", ntype=nt_dict[unit],
                              lims=[0, np.inf])
            except Exception as exc:
                raise exc
            for key in ["prepadlength", "postpadlength"]:
                if not isinstance(plengths[key], (bool, type(None))):
                    raise SPYTypeError(plengths[key], varname=key, expected="bool or None")

            if prepadlength is None and postpadlength is None:
                prepadlength = True
                postpadlength = True
            else:
                prepadlength = prepadlength is not None
                postpadlength = postpadlength is not None

            if prepadlength and postpadlength:
                plengths["prepadlength"] = padlength/2
                plengths["postpadlength"] = padlength/2
            else:
                plengths["prepadlength"] = prepadlength*padlength
                plengths["postpadlength"] = postpadlength*padlength

        # Under-determined: abort if requested padding length is 0
        if all(value == 0 for value in plengths.values() if value is not None):
            lgl = "either non-zero value of `padlength` or `prepadlength` " + \
                  "and/or `postpadlength` to be set"
            raise SPYValueError(legal=lgl, varname="padlength", actual="0|None")

    else:

        # For absolute padding, the desired length has to be >= max. trial length
        if pad == "absolute":
            try:
                scalar_parser(padlength, varname="padlength", ntype=nt_dict[unit],
                              lims=[padlim, np.inf])
            except Exception as exc:
                raise exc
            for key in ["prepadlength", "postpadlength"]:
                if not isinstance(plengths[key], (bool, type(None))):
                    raise SPYTypeError(plengths[key], varname=key, expected="bool or None")

        # For `maxlen` or `nextpow2` we don't want any numeric entries at all
        else:
            for key, value in plengths.items():
                if not isinstance(value, (bool, type(None))):
                    raise SPYTypeError(value, varname=key, expected="bool or None")

            # Warn of potential conflicts
            if padlength and (prepadlength or postpadlength):
                msg = "Found `padlength` and `prepadlength` and/or " +\
                    "`postpadlength`. Symmetric padding is performed. "
                SPYWarning(msg)

        # If both pre-/post- are `None`, set them to `True` to use symmetric
        # padding, otherwise convert `None` entries to `False`
        if prepadlength is None and postpadlength is None:
            plengths["prepadlength"] = True
            plengths["postpadlength"] = True
        else:
            plengths["prepadlength"] = plengths["prepadlength"] is not None
            plengths["postpadlength"] = plengths["postpadlength"] is not None

    # Update pre-/post-padding and (if required) convert time to samples
    prepadlength = plengths["prepadlength"]
    postpadlength = plengths["postpadlength"]
    if unit == "time":
        if pad == "relative":
            prepadlength = int(prepadlength*data.samplerate)
            postpadlength = int(postpadlength*data.samplerate)
        elif pad == "absolute":
            padlength = int(padlength*data.samplerate)

    # Construct dict of keywords for ``np.pad`` depending on chosen `padtype`
    kws = {"zero": {"mode": "constant", "constant_values": 0},
           "nan": {"mode": "constant", "constant_values": np.nan},
           "localmean": {"mode": "mean", "stat_length": -1},
           "mean": {"mode": "mean"},
           "edge": {"mode": "edge"},
           "mirror": {"mode": "reflect"}}

    # If input was syncopy data object, padding is done on a per-trial basis
    if is_spydata:

        # A list of input keywords for ``np.pad`` is constructed, no matter if
        # we actually want to build a new object or not
        pad_opts = []
        for tk in trialList:
            nSamples = data._preview_trial(tk).shape[timeAxis]
            if pad == "absolute":
                padding = (padlength - nSamples)/(prepadlength + postpadlength)
            elif pad == "relative":
                padding = True
            elif pad == "maxlen":
                padding = (maxTrialLen - nSamples)/(prepadlength + postpadlength)
            elif pad == "nextpow2":
                padding = (_nextpow2(nSamples) - nSamples)/(prepadlength + postpadlength)
            pw = np.zeros((2, 2), dtype=int)
            pw[timeAxis, :] = [prepadlength * padding, postpadlength * padding]
            pad_opts.append(dict({"pad_width": pw}, **kws[padtype]))
            if padtype == "localmean":
                pad_opts[-1]["stat_length"] = pw[timeAxis, :]

        # If a new object is requested, use the legwork performed above to fire
        # up the corresponding ComputationalRoutine
        if create_new:
            out = AnalogData(dimord=data.dimord)
            log_dct = {"padtype": padtype,
                       "pad": pad,
                       "padlength": padlength,
                       "prepadlength": prepadlength,
                       "postpadlength": postpadlength,
                       "unit": unit}

            chanAxis = list(set([0, 1]).difference([timeAxis]))[0]
            padMethod = PaddingRoutine(timeAxis, chanAxis, pad_opts)
            padMethod.initialize(data,
                                 out._stackingDim,
                                 chan_per_worker=kwargs.get("chan_per_worker"),
                                 keeptrials=True)
            padMethod.compute(data, out, parallel=kwargs.get("parallel"), log_dict=log_dct)
            return out
        else:
            return pad_opts

    # Input was a array/FauxTrial (i.e., single trial) - we have to do the padding just once
    else:

        nSamples = data.shape[timeAxis]
        if pad == "absolute":
            prepadding = (padlength - nSamples)/(prepadlength + postpadlength)
            postpadding = prepadding
            if int(prepadding) != prepadding:
                prepadding = int(prepadding)
                postpadding = prepadding + 1
        elif pad == "relative":
            prepadding = True
            postpadding = prepadding
        elif pad == "nextpow2":
            prepadding = (_nextpow2(nSamples) - nSamples)/(prepadlength + postpadlength)
            postpadding = prepadding
            if int(prepadding) != prepadding:
                prepadding = int(prepadding)
                postpadding = prepadding + 1
        pw = np.zeros((2, 2), dtype=int)
        pw[timeAxis, :] = [prepadlength * prepadding, postpadlength * postpadding]
        pad_opts = dict({"pad_width": pw}, **kws[padtype])
        if padtype == "localmean":
            pad_opts["stat_length"] = pw[timeAxis, :]

        if create_new:
            if isinstance(data, np.ndarray):
                return np.pad(data, **pad_opts)
            else:
                shp = list(data.shape)
                shp[timeAxis] += pw[timeAxis, :].sum()
                idx = list(data.idx)
                if isinstance(idx[timeAxis], slice):
                    idx[timeAxis] = slice(idx[timeAxis].start,
                                          idx[timeAxis].start + shp[timeAxis])
                else:
                    idx[timeAxis] = pw[timeAxis, 0] * [idx[timeAxis][0]] + idx[timeAxis] \
                                    + pw[timeAxis, 1] * [idx[timeAxis][-1]]
                return data.__class__(shp, idx, data.dtype, data.dimord)
        else:
            return pad_opts


def _nextpow2(number):
    n = 1
    while n < number:
        n *= 2
    return n


@process_io
def padding_cF(trl_dat, timeAxis, chanAxis, pad_opt, noCompute=False, chunkShape=None):
    """
    Perform trial data padding

    Parameters
    ----------
    trl_dat : :class:`numpy.ndarray`
        Trial data
    timeAxis : int
        Index of running time axis in `trl_dat` (0 or 1)
    chanAxis : int
        Index of channel axis in `trl_dat` (0 or 1)
    pad_opt : dict
        Dictionary of options for :func:`numpy.pad`
    noCompute : bool
        Preprocessing flag. If `True`, do not perform actual padding but
        instead return expected shape and :class:`numpy.dtype` of output
        array.
    chunkShape : None or tuple
        If not `None`, represents shape of output

    Returns
    -------
    res : :class:`numpy.ndarray`
        Padded array

    Notes
    -----
    This method is intended to be used as
    :meth:`~syncopy.shared.computational_routine.ComputationalRoutine.computeFunction`
    inside a :class:`~syncopy.shared.computational_routine.ComputationalRoutine`.
    Thus, input parameters are presumed to be forwarded from a parent metafunction.
    Consequently, this function does **not** perform any error checking and operates
    under the assumption that all inputs have been externally validated and cross-checked.

    See also
    --------
    syncopy.padding : pad :class:`syncopy.AnalogData` objects
    PaddingRoutine : :class:`~syncopy.shared.computational_routine.ComputationalRoutine` subclass
    """

    nSamples = trl_dat.shape[timeAxis]
    nChannels = trl_dat.shape[chanAxis]

    if noCompute:
        outShape = [None] * 2
        outShape[timeAxis] = pad_opt['pad_width'].sum() + nSamples
        outShape[chanAxis] = nChannels
        return outShape, trl_dat.dtype

    # Symmetric Padding (updates no. of samples)
    return np.pad(trl_dat, **pad_opt)

class PaddingRoutine(ComputationalRoutine):
    """
    Compute class for performing data padding on Syncopy AnalogData objects

    Sub-class of :class:`~syncopy.shared.computational_routine.ComputationalRoutine`,
    see :doc:`/developer/compute_kernels` for technical details on Syncopy's compute
    classes and metafunctions.

    See also
    --------
    syncopy.padding : pad :class:`syncopy.AnalogData` objects
    """

    computeFunction = staticmethod(padding_cF)

    def process_metadata(self, data, out):

        # Fetch index of running time and used padding options from provided
        # positional args and use them to compute new start/stop/trigger onset samples
        timeAxis = self.argv[0]
        pad_opts = self.argv[2]
        prePadded = [pad_opt["pad_width"][timeAxis, 0] for pad_opt in pad_opts]
        totalPadded = [pad_opt["pad_width"].sum() for pad_opt in pad_opts]
        accumSamples = np.cumsum(np.diff(data._pad_sinfo).squeeze() + totalPadded)

        # Construct trialdefinition array (columns: start/stop/t0/etc)
        trialdefinition = np.zeros((len(totalPadded), data.trialdefinition.shape[1]))
        trialdefinition[1:, 0] = accumSamples[:-1]
        trialdefinition[:, 1] = accumSamples
        trialdefinition[:, 2] = data._pad_t0 - prePadded

        # Set relevant properties in output object
        out.samplerate = data.samplerate
        out.trialdefinition = trialdefinition
        out.channel = data._pad_channel

        # Remove inpromptu attributes generated above
        delattr(data, "_pad_sinfo")
        delattr(data, "_pad_t0")
        delattr(data, "_pad_channel")
