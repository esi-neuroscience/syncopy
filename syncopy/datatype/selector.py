# Builtin/3rd party package imports
from functools import reduce
import numpy as np

# syncopy imports
from syncopy.shared.parsers import array_parser, data_parser
from syncopy.shared.errors import SPYTypeError, SPYValueError, SPYError

# local imports
from .util import TrialIndexer


class Selector:
    """
    Auxiliary class for data selection

    Parameters
    ----------
    data : Syncopy data object
        A non-empty Syncopy data object
    select : dict or :class:`~syncopy.shared.tools.StructDict` or None or str
        Dictionary or :class:`~syncopy.shared.tools.StructDict` with keys
        specifying data selectors. **Note**: some keys are only valid for certain types
        of Syncopy objects, e.g., "freq" is not a valid selector for an
        :class:`~syncopy.AnalogData` object. Supported keys are (please see
        :func:`~syncopy.selectdata` for a detailed description of each selector)

        * 'trials' : list (integers)
        * 'channel' : list (integers or strings), slice or range
        * 'toi' : list (floats)
        * 'toilim' : list (floats [tmin, tmax])
        * 'foi' : list (floats)
        * 'foilim' : list (floats [fmin, fmax])
        * 'taper' : list (integers or strings), slice or range
        * 'unit' : list (integers or strings), slice or range
        * 'eventid' : list (integers), slice or range

        Any property of `data` that is not specifically accessed via one of
        the above keys is taken as is, e.g., ``select = {'trials': [1, 2]}``
        selects the entire contents of trials no. 2 and 3, while
        ``select = {'channel': range(0, 50)}`` selects the first 50 channels
        of `data` across all defined trials. Consequently, if `select` is
        `None` or if ``select = "all"`` the entire contents of `data` is selected.

    Returns
    -------
    selection : Syncopy :class:`Selector` object
        An instance of this class whose main properties are either lists or slices
        to be used as (fancy) indexing tuples. Note that the properties `time`,
        `unit` and `eventid` are **by-trial** selections, i.e., list of lists
        and/or slices encoding per-trial sample-indices, e.g., ``selection.time[0]``
        is intended to be used with ``data.trials[selection.trial_ids[0]]``.
        Addditional class attributes of note:

        * `_useFancy` : bool

          If `True`, selection requires "fancy" (or "advanced") array indexing

        * `_dataClass` : str

          Class name of `data`

        * `_samplerate` : float

          Samplerate of `data` (only relevant for objects supporting time-selections)

        * `_timeShuffle` : bool

          If `True`, time-selection contains unordered/repeated time-points.

        * `_allProps` : list

          List of all selection properties in class

        * `_byTrialProps` : list

          List off by-trial selection properties (see above)

        * `_dimProps` : list

          List off trial-independent selection properties (computed as
          `self._allProps` minus `self._byTrialProps`)

    Notes
    -----
    Whenever possible, this class performs extensive input parsing to ensure
    consistency of provided selectors. Some exceptions to this rule include
    `toi` and `toilim`: depending on the size of `data` and the number of
    defined trials, `data.time` might generate a list of arrays of substantial
    size. To not overflow memory and slow down computations, neither `toi`
    nor `toilim` is checked for consistency with respect to `data.time`, i.e.,
    the code does not verify that min/max of `toi`/`toilim` are within the
    bounds of `data.time` for each selected trial.

    For objects that have a `time` property, a suitable new `trialdefinition`
    array (accessible via the identically named `Selector` class property)
    is automatically constructed based on the provided selection. For unsorted
    time-selections with or without repetitions, the `timepoints` property
    encodes the timing of the selected (discrete) points. To permit this
    functionality, the input object's samplerate is stored in the identically
    named hidden attribute `_samplerate`. In addition, the hidden `_timeShuffle`
    attribute is a binary flag encoding whether selected time-points are
    unordered and/or contain repetitions (`Selector._timeShuffle = True`).

    By default, each selection property tries to convert a user-provided
    selection to a contiguous slice-indexer so that simple NumPy array
    indexing can be used for best performance. However, after setting all
    selection indices appropriate for the input object, a consistency
    check is performed by :meth:`_make_consistent` to ensure that the
    calculated indices can actually be jointly used on a multi-dimensional
    NumPy array without violating indexing arithmetic. Thus, if a given
    Selector instance ends up containing more than two conjoint index-lists,
    all other selection properties are converted (if necessary) to lists as well
    for use with :func:`numpy.ix_`. These selections require special array
    manipulation techniques (colloquially referred to as "fancy" or "advanced"
    indexing) and the :class:`Selector` marks such indexers by setting the
    hidden `self._useFancy` attribute to `True`. Note that :func:`numpy.ix_`
    always creates copies of the indexed reference array, hence, the attempt
    to use slice-based indexing whenever possible.

    Examples
    --------
    See :func:`syncopy.selectdata` for usage examples.

    See also
    --------
    syncopy.selectdata : extract data selections from Syncopy objects
    """

    def __init__(self, data, select):

        # Ensure input makes sense
        try:
            data_parser(data, varname="data", empty=False)
        except Exception as exc:
            raise exc
        if select is None:
            select = {}
        if isinstance(select, str):
            if select == "all":
                select = {}
            else:
                raise SPYValueError(
                    legal="'all' or `None` or dict", varname="select", actual=select
                )
        if not isinstance(select, dict):
            raise SPYTypeError(select, "select", expected="dict")

        # Keep list of supported selectors in sync w/supported keywords of `selectdata`
        supported = data._selectionKeyWords
        # `selectdata` already throws out not supported keywords
        # so this is just a hard check when setting a selection via assignment
        if not set(select.keys()).issubset(supported):
            lgl = (
                "dict with one or all of the following keys: '"
                + "'".join(opt + "', " for opt in supported)[:-2]
            )
            act = (
                "dict with keys '" + "'".join(key + "', " for key in select.keys())[:-2]
            )
            raise SPYValueError(legal=lgl, varname="select", actual=act)

        # Save class of input object for posterity
        self._dataClass = data.__class__.__name__

        # Set up lists of (a) all selectable properties (b) trial-dependent ones
        # and (c) selectors independent from trials
        self._allProps = [
            "channel",
            "channel_i",
            "channel_j",
            "time",
            "freq",
            "taper",
            "unit",
            "eventid",
        ]
        self._byTrialProps = ["time", "unit", "eventid"]
        self._dimProps = list(self._allProps)
        for prop in self._byTrialProps:
            self._dimProps.remove(prop)

        # Special adjustment for `CrossSpectralData`: remove (invalid) `channel` property
        # from `_dimProps` (avoid pitfalls in code-blocks iterating over `_dimProps`)
        if self._dataClass == "CrossSpectralData":
            self._dimProps.remove("channel")

        # Assign defaults (trials are not a "real" property, handle it separately,
        # same goes for `trialdefinition`)
        self._trials = None
        self._trial_ids = None
        self._trialdefinition = None
        for prop in self._allProps:
            setattr(self, "_{}".format(prop), None)
        self._useFancy = False  # flag indicating whether fancy indexing is necessary
        self._samplerate = None  # for objects supporting time-selections
        self._timeShuffle = (
            False  # flag indicating whether time-points are repeated/unordered
        )

        # We first need to know which trials are of interest here (assuming
        # that any valid input object *must* have a `trials_ids` attribute)
        self.trial_ids = (data, select)

        # Now set any possible selection attribute (depending on type of `data`)
        # Note: `trialdefinition` is set *after* harmonizing indexing selections
        # in `_make_consistent`
        for prop in self._allProps:
            setattr(self, prop, (data, select))

        # Ensure correct indexing: harmonize selections for `DiscreteData`-children
        # or convert everything to lists for use w/`np.ix_` if we ended up w/more
        # than 2 list selectors for `ContinuousData`-offspring
        self._make_consistent(data)

        # store for later re-application/modification
        self.select = select

        # create the Selector._get_trial helper
        self.create_get_trial(data)

    @property
    def trial_ids(self):
        """Index list of selected trials"""
        return self._trial_ids

    @trial_ids.setter
    def trial_ids(self, dataselect):
        data, select = dataselect
        trlList = list(range(len(data.trials)))
        trials = select.get("trials", None)
        vname = "select: trials"

        if isinstance(trials, str):
            if trials == "all":
                trials = None
            else:
                raise SPYValueError(
                    legal="'all' or `None` or list/array", varname=vname, actual=trials
                )
        if trials is not None:
            if np.issubdtype(type(trials), np.number):
                trials = [trials]
            try:
                array_parser(
                    trials,
                    varname=vname,
                    ntype="int_like",
                    hasinf=False,
                    hasnan=False,
                    lims=[0, len(data.trials)],
                    dims=1,
                )
            except Exception as exc:
                raise exc
            if not set(trials).issubset(trlList):
                lgl = "list/array of values b/w 0 and {}".format(trlList[-1])
                act = "Values b/w {} and {}".format(min(trials), max(trials))
                raise SPYValueError(legal=lgl, varname=vname, actual=act)
        else:
            trials = trlList
        self._trial_ids = list(trials)  # ensure `trials` is a list cf. #180

    @property
    def trials(self):
        """
        Returns an iterable indexing single trial arrays respecting the selection
        Indices are ABSOLUTE with respect to existing trial selections:

        >>> selection.trials[11]

        indexes the 11th trial of the original dataset, if and only if
        trial number 11 is part of the selection.

        Selections must be "simple": ordered and without repetitions
        """

        if self.sampleinfo is not None:
            # this is cheap as it just initializes a list-like object
            # with no real data and/or computations!
            return TrialIndexer(self, self.trial_ids)
        else:
            return None

    def create_get_trial(self, data):
        """ Closure to allow emulation of BaseData._get_trial"""

        # trl_id has to be part of selection for coherence
        def _get_trial(trl_id):
            if trl_id not in self.trial_ids:
                lgl = "a trial part of the selection"
                act = trl_id
                raise SPYValueError(lgl, "Selector.trials", act)
            # extract the selection respecting FauxTrial idx tuple
            # which has length len(data.dimord) or 2 if `data` is a DiscreteData instance
            trl_idx = data._preview_trial(trl_id).idx

            # now massage/validate it such that we can use it to
            # directly index the hdf5 dataset
            # tuple elements can only be lists or ordered slices, see concrete
            # `_preview_trial` implementations which generate those idx tuples
            # maybe TODO: allow fancy indexing like in the CR
            for i, dim_idx in enumerate(trl_idx):
                if isinstance(dim_idx, list):
                    # no fancy indexing, no repetitions
                    if len(set(dim_idx)) != len(dim_idx):
                        lgl = "simple selections w/o repetitions"
                        act = f"fancy selection with repetitions for selector {data.dimord[i]}"
                        raise SPYValueError(lgl, "Selector.trials", act)

                    # DiscreteData selections inherently re-order the sample dim. idx
                    # so these we sort, all others we need ordered
                    if 'discrete_data' in str(data.__class__):
                        # sorts in place!
                        dim_idx.sort()
                    elif np.any(np.diff(dim_idx) < 0):
                        lgl = "simple selection in ascending order"
                        act = f"fancy non-ordered selection of selector {data.dimord[i]}"
                        raise SPYValueError(lgl, "Selector.trials", act)
            # if we landed here all is good and we take
            # a leap of faith into the hdf5 dataset
            return data.data[trl_idx]

        # finally bind it to the Selector instance
        self._get_trial = _get_trial

    @property
    def channel(self):
        """List or slice encoding channel-selection"""
        return self._channel

    @channel.setter
    def channel(self, dataselect):
        data, select = dataselect
        chanSpec = select.get("channel")
        if self._dataClass == "CrossSpectralData":
            if chanSpec is not None:
                lgl = "`channel_i` and/or `channel_j` selectors for `CrossSpectralData`"
                raise SPYValueError(
                    legal=lgl, varname="select: channel", actual=data.__class__.__name__
                )
            else:
                return
        self._selection_setter(data, select, "channel")

    @property
    def channel_i(self):
        """List or slice encoding principal channel-pair selection"""
        return self._channel_i

    @channel_i.setter
    def channel_i(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "channel_i")

    @property
    def channel_j(self):
        """List or slice encoding principal channel-pair selection"""
        return self._channel_j

    @channel_j.setter
    def channel_j(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "channel_j")

    @property
    def time(self):
        """len(self.trial_ids) list of lists/slices of by-trial time-selections"""
        return self._time

    @time.setter
    def time(self, dataselect):

        # Unpack input and perform error-checking
        data, select = dataselect
        timeSpec = select.get("latency", None)
        checkInf = None
        vname = "select: latency"

        hasTime = hasattr(data, "time") or hasattr(data, "trialtime")
        if timeSpec is not None and hasTime is False:
            lgl = "Syncopy data object with time-dimension"
            raise SPYValueError(
                legal=lgl, varname=vname, actual=data.__class__.__name__
            )

        # If `data` has a `time` property, fill up `self.time`
        if hasTime:
            if isinstance(timeSpec, str):
                if timeSpec == "all":
                    timeSpec = None
                    select["latency"] = None
                else:
                    raise SPYValueError(
                        legal="'all' or `None` or list/array",
                        varname=vname,
                        actual=timeSpec,
                    )
            if timeSpec is not None:
                if np.issubdtype(type(timeSpec), np.number):
                    timeSpec = [timeSpec]
                    array_parser(
                        timeSpec, varname=vname, hasinf=checkInf, hasnan=False, dims=1
                    )
                # can only be 2-sequence [start, end]
                else:
                    if len(timeSpec) != 2:
                        lgl = "`select: latency` selection with two components"
                        act = "`select: latency` with {} components".format(
                            len(timeSpec)
                        )
                        raise SPYValueError(legal=lgl, varname=vname, actual=act)
                    if timeSpec[0] >= timeSpec[1]:
                        lgl = (
                            "`select: latency` selection with `latency[0]` < `latency[1]`"
                        )
                        act = "selection range from {} to {}".format(
                            timeSpec[0], timeSpec[1]
                        )
                        raise SPYValueError(legal=lgl, varname=vname, actual=act)
            timing = data._get_time(self.trial_ids, toi=None, toilim=select.get("latency"))

            # ---------------------------------------------------------------------------
            # this is legacy, might be needed later if ppl really want to "time shuffle"
            # to destroy any correlations and produce white noise from their data..
            # .. which is questionable

            # Determine, whether time-selection is unordered/contains repetitions
            # and set `self._timeShuffle` accordingly
            if timeSpec is not None:  # saves time for `timeSpec = None` "selections"
                for tsel in timing:
                    if isinstance(tsel, list) and len(tsel) > 1:
                        if np.diff(tsel).min() <= 0:
                            self._timeShuffle = True
                            break
            # ---------------------------------------------------------------------------

            # Assign timing selection and copy over samplerate from source object
            self._time = timing
            self._samplerate = data.samplerate

        else:
            return

    @property
    def trialdefinition(self):
        """len(self.trial_ids)-by-(3+) :class:`numpy.ndarray` encoding trial-information of selection"""
        return self._trialdefinition

    @trialdefinition.setter
    def trialdefinition(self, data):

        # Get original `trialdefinition` array for reference
        trl = data.trialdefinition

        # `DiscreteData`: simply copy relevant sample-count -> trial assignments,
        # for other classes build new trialdefinition array using `t0`-offsets
        if self._dataClass in ["SpikeData", "EventData"]:
            trlDef = trl[self.trial_ids, :]
        else:
            trlDef = np.zeros((len(self.trial_ids), trl.shape[1]))
            counter = 0
            for tk, trlno in enumerate(self.trial_ids):
                tsel = self.time[tk]
                if isinstance(tsel, slice):
                    start, stop, step = tsel.start, tsel.stop, tsel.step
                    if start is None:
                        start = 0
                    if stop is None:
                        trlTime = data._get_time([trlno], toilim=[-np.inf, np.inf])[0]
                        if isinstance(trlTime, list):
                            stop = np.max(trlTime)
                            # Avoid creating empty arrays for "static" `SpectralData` objects
                            if stop == start == 0:
                                stop += 1
                        else:
                            stop = trlTime.stop
                    if step is None:
                        step = 1
                    nSamples = (stop - start) / step
                    endSample = stop + data._t0[trlno]
                    t0 = int(endSample - nSamples)
                else:
                    nSamples = len(tsel)
                    if nSamples == 0:
                        t0 = 0
                    else:
                        t0 = data._t0[trlno]
                trlDef[tk, :3] = [counter, counter + nSamples, t0]
                trlDef[tk, 3:] = trl[trlno, 3:]
                counter += nSamples
        self._trialdefinition = trlDef

    @property
    def sampleinfo(self):
        """nTrials x 2 :class:`numpy.ndarray` of [start, end] sample indices"""
        if self._trialdefinition is not None:
            return self._trialdefinition[:, :2]
        else:
            return None

    @sampleinfo.setter
    def sampleinfo(self, sinfo):
        raise SPYError("Cannot set sampleinfo. Use `Selector.trialdefinition` instead.")

    @property
    def trialintervals(self):
        """nTrials x 2 :class:`numpy.ndarray` of [start, end] times in seconds """
        if self._trialdefinition is not None and self._samplerate is not None:
            # trial lengths in samples
            start_end = self.sampleinfo - self.sampleinfo[:, 0][:, None]
            start_end[:, 1] -= 1  # account for last time point
            # add offset and convert to seconds
            start_end = (start_end + self.trialdefinition[:, 2][:, None]) / self._samplerate
            return start_end
        else:
            return None

    @property
    def timepoints(self):
        """len(self.trial_ids) list of lists encoding actual (not sample indices!)
        timing information of unordered `toi` selections"""
        if self._timeShuffle:
            return [
                [
                    (tvec[tp] + self.trialdefinition[tk, 2]) / self._samplerate
                    for tp in range(len(tvec))
                ]
                for tk, tvec in enumerate(self.time)
            ]

    @property
    def freq(self):
        """List or slice encoding frequency-selection"""
        return self._freq

    @freq.setter
    def freq(self, dataselect):

        # Unpack input and perform error-checking
        data, select = dataselect
        freqSpec = select.get("frequency")
        hasFreq = hasattr(data, "freq")
        if freqSpec is not None and hasFreq is False:
            lgl = "Syncopy data object with freq-dimension"
            raise SPYValueError(
                legal=lgl, varname="frequency", actual=data.__class__.__name__
            )

        # If `data` has a `freq` property, fill up `self.freq`
        if hasFreq:
            if isinstance(freqSpec, str):
                if freqSpec == "all":
                    freqSpec = None
                    select["frequency"] = None
                else:
                    raise SPYValueError(
                        legal="'all' or `None` or float or list/array",
                        varname="frequency",
                        actual=freqSpec,
                    )
            if freqSpec is None:
                # select all
                self._freq = data._get_freq()

            else:
                if np.issubdtype(type(freqSpec), np.number):
                    freqSpec = [freqSpec]

                    array_parser(
                        freqSpec,
                        varname="frequency",
                        hasinf=False,
                        hasnan=False,
                        lims=[data.freq.min(), data.freq.max()],
                        dims=(1,),
                    )
                    # single frequency
                    self._freq = data._get_freq(foi=freqSpec)
                # frequency range [fmin, fmax]
                else:
                    array_parser(
                        freqSpec,
                        ntype="numeric",
                        varname="frequency",
                        hasnan=False,
                        lims=[data.freq.min(), data.freq.max()],
                        dims=(2,),
                    )
                    if freqSpec[0] >= freqSpec[1]:
                        lgl = (
                            "`select: frequency` selection with `frequency[0]` < `frequency[1]`"
                        )
                        act = "selection range from {} to {}".format(
                            freqSpec[0], freqSpec[1]
                        )
                        raise SPYValueError(legal=lgl, varname='frequency', actual=act)

                    self._freq = data._get_freq(foi=None, foilim=freqSpec)

    @property
    def taper(self):
        """List or slice encoding taper-selection"""
        return self._taper

    @taper.setter
    def taper(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "taper")

    @property
    def unit(self):
        """len(self.trial_ids) list of lists/slices of by-trial unit-selections"""
        return self._unit

    @unit.setter
    def unit(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "unit")

    @property
    def eventid(self):
        """len(self.trials) list of lists/slices encoding by-trial event-id-selection"""
        return self._eventid

    @eventid.setter
    def eventid(self, dataselect):
        data, select = dataselect
        self._selection_setter(data, select, "eventid")

    # Helper function to process provided selections
    def _selection_setter(self, data, select, selectkey):
        """
        Converts user-provided selection key-words to indexing lists/slices

        Parameters
        ----------
        data : Syncopy data object
            Non-empty Syncopy data object
        select : dict or :class:`StructDict`
            Python dictionary or Syncopy :class:`StructDict` formatted for
            data selection. See :class:`Selector` for a list of valid
            key-value pairs.
        selectkey : str
            Name of key in `select` holding selection pertinent to identically
            named property in `data`

        Returns
        -------
        Nothing : None

        Notes
        -----
        This class method processes and (if necessary converts) user-provided
        selections. Valid selectors are slices, ranges, lists or arrays. If
        possible, all selections are converted to contiguous slices, otherwise
        regular Python lists are used. Selections can be unsorted and may
        include repetitions but must match exactly, be finite and not NaN.
        Converted selections are stored in the respective (hidden) class
        attributes (e.g., ``self._channel``, ``self._unit`` etc.).

        See also
        --------
        syncopy.selectdata : extract data selections from Syncopy objects
        """

        # Unpack input and perform error-checking
        selection = select.get(selectkey)
        target = getattr(data, selectkey, None)
        selector = "_{}".format(selectkey)
        vname = "select: {}".format(selectkey)
        if selection is not None and target is None:
            lgl = "Syncopy data object with {}".format(selectkey)
            raise SPYValueError(
                legal=lgl, varname=vname, actual=data.__class__.__name__
            )

        if target is not None:

            if np.issubdtype(target.dtype, np.dtype("str").type):
                slcLims = [0, target.size]
                arrLims = None
                hasnan = None
                hasinf = None
            else:
                slcLims = [target[0], target[-1] + 1]
                arrLims = [target[0], target[-1]]
                hasnan = False
                hasinf = False

            # Convert 'all' selections to take-all `None` (see next if below) and
            # put single-string selections into a list; same for single-scalar selections
            if isinstance(selection, str):
                if selection == "all":
                    selection = None
                else:
                    selection = [selection]
            elif np.issubdtype(type(selection), np.number):
                selection = [selection]

            # Take entire inventory sitting in `selectkey`
            if selection is None:
                if selectkey in ["unit", "eventid"]:
                    setattr(self, selector, [slice(None, None, 1)] * len(self.trial_ids))
                else:
                    setattr(self, selector, slice(None, None, 1))

            # Check consistency of slice-selections and convert ranges to slices
            elif isinstance(selection, (slice, range)):
                selLims = [-np.inf, np.inf]
                if selection.start is not None:
                    selLims[0] = selection.start
                if selection.stop is not None:
                    selLims[1] = selection.stop
                if selLims[0] >= selLims[1]:
                    lgl = "selection range with min < max"
                    act = "selection range from {} to {}".format(selLims[0], selLims[1])
                    raise SPYValueError(legal=lgl, varname=vname, actual=act)
                # check slice/range boundaries: take care of things like `slice(-10, -3)`
                if np.isfinite(selLims[0]) and (
                    selLims[0] < -slcLims[1] or selLims[0] >= slcLims[1]
                ):
                    lgl = "selection range with min >= {}".format(slcLims[0])
                    act = "selection range starting at {}".format(selLims[0])
                    raise SPYValueError(legal=lgl, varname=vname, actual=act)
                if np.isfinite(selLims[1]) and (
                    selLims[1] > slcLims[1] or selLims[1] < -slcLims[1]
                ):
                    lgl = "selection range with max <= {}".format(slcLims[1])
                    act = "selection range ending at {}".format(selLims[1])
                    raise SPYValueError(legal=lgl, varname=vname, actual=act)

                # The 2d-arrays in `DiscreteData` objects require some additional hand-holding
                # performed by the respective `_get_unit` and `_get_eventid` class methods
                if selectkey in ["unit", "eventid"]:
                    if selection.start is selection.stop is None:
                        setattr(self, selector, [slice(None, None, 1)] * len(self.trial_ids))
                    else:
                        if isinstance(selection, slice):
                            if np.issubdtype(target.dtype, np.dtype("str").type):
                                target = np.arange(target.size)
                            selection = list(target[selection])
                        else:
                            selection = list(selection)
                        setattr(self, selector, getattr(data, "_get_" + selectkey)(self.trial_ids, selection))

                else:
                    if selection.start is selection.stop is None:
                        setattr(self, selector, slice(None, None, 1))
                    else:
                        if selection.step is None:
                            step = 1
                        else:
                            step = selection.step
                        setattr(
                            self, selector, slice(selection.start, selection.stop, step)
                        )

            # Selection is either a valid list/array or bust
            else:
                try:
                    array_parser(
                        selection,
                        varname=vname,
                        hasinf=hasinf,
                        hasnan=hasnan,
                        lims=arrLims,
                        dims=1,
                    )
                except Exception as exc:
                    raise exc
                selection = np.array(selection)
                if np.issubdtype(selection.dtype, np.dtype("str").type):
                    targetArr = target
                else:
                    targetArr = np.arange(target.size)
                if not set(selection).issubset(targetArr):
                    lgl = "list/array of {} existing names or indices".format(selectkey)
                    raise SPYValueError(legal=lgl, varname=vname)

                # Preserve order and duplicates of selection - don't use `np.isin` here!
                idxList = []
                for sel in selection:
                    idxList += list(np.where(targetArr == sel)[0])

                if selectkey in ["unit", "eventid"]:
                    setattr(self, selector, getattr(data, "_get_" + selectkey)(self.trial_ids, idxList))
                else:
                    # if possible, convert range-arrays (`[0, 1, 2, 3]`) to slices for better performance
                    if len(idxList) > 1:
                        steps = np.diff(idxList)
                        if steps.min() == steps.max() == 1:
                            idxList = slice(idxList[0], idxList[-1] + 1, 1)

                    # be careful w/pairwise list-channel selections in `CrossSpectralData` objects
                    # (that could not be converted to slices above)
                    if isinstance(idxList, list) and selectkey in [
                        "channel_i",
                        "channel_j",
                    ]:
                        if len(idxList) > 1:
                            err = "Unordered (low to high) or non-contiguous multi-channel-pair selections not supported"
                            raise NotImplementedError(err)
                        idxList = idxList[0]

                    setattr(self, selector, idxList)

        else:
            return

    # Local helper that converts slice selectors to lists (if necessary)
    def _make_consistent(self, data):
        """
        Consolidate multi-selections

        Parameters
        ----------
        data : Syncopy data object
            Non-empty Syncopy data object

        Returns
        -------
        Nothing : None

        Notes
        -----
        This class method is called after all user-provided selections have
        been (successfully) processed and (if necessary) converted to
        lists/slices.
        For instances of :class:`~syncopy.datatype.continuous_data.ContinuousData`
        child classes (i.e., :class:`~syncopy.AnalogData` and :class:`~syncopy.SpectralData`
        objects) the integrity of conjoint multi-dimensional selections
        is ensured.
        For instances of :class:`~syncopy.datatype.discrete_data.DiscreteData`
        child classes (i.e., :class:`~syncopy.SpikeData` and :class:`~syncopy.EventData`
        objects), any selection (`unit`, `eventid`, `time` and `channel`) operates
        on the rows of the object's underlying `data` array. Thus, multi-selections
        need to be synchronized (e.g., a `unit` selection pointing to rows `[0, 1, 2]`
        and a `time` selection filtering rows `[1, 2, 3]` are combined to `[1, 2]`).

        See also
        --------
        numpy.ix_ : Mesh-construction for array indexing
        """

        # Harmonize selections for `DiscreteData`-children: all selectors are row-
        # indices, go through each trial and combine them
        if self._dataClass in ["SpikeData", "EventData"]:

            # Get relevant selectors (e.g., `self.unit` is `None` for `EventData`)
            actualSelections = []
            for selection in ["time", "eventid", "unit"]:
                if getattr(self, selection) is not None:
                    actualSelections.append(selection)

            # Compute intersection of "time" x "{eventid|unit|channel}" row-indices
            # per trial. BONUS: in `SpikeData` objects, `channels` are **not**
            # the same in all trials - ensure that channel selection propagates
            # correctly. After this step, `self.time` == `self.{unit|eventid}`
            if self._dataClass == "SpikeData":
                chanIdx = data.dimord.index("channel")
                wantedChannels = data.channel_idx[self.channel]
                chanPerTrial = []

            for tk, trialno in enumerate(self.trial_ids):
                trialArr = np.arange(data._trialslice[trialno].stop - data._trialslice[trialno].start)
                byTrialSelections = []
                for selection in actualSelections:
                    byTrialSelections.append(trialArr[getattr(self, selection)[tk]])

                # (try to) preserve unordered selections by processing them first
                areShuffled = [(np.diff(sel) <= 0).any() for sel in byTrialSelections]
                combiOrder = np.argsort(areShuffled)[::-1]
                combinedSelect = byTrialSelections[combiOrder[0]]
                for combIdx in combiOrder:
                    combinedSelect = combinedSelect[np.isin(combinedSelect, byTrialSelections[combIdx])]

                # Keep record of channels present in trials vs. selected channels
                if self._dataClass == "SpikeData":
                    rawChanInTrial = data.trials[trialno][:, chanIdx]
                    chanTrlIdx = np.flatnonzero(np.isin(rawChanInTrial, wantedChannels))
                    combinedSelect = combinedSelect[np.isin(combinedSelect, chanTrlIdx)].tolist()
                    chanPerTrial.append(rawChanInTrial[combinedSelect])
                elif areShuffled:
                    combinedSelect = combinedSelect.tolist()

                # The usual list -> slice conversion (if possible)
                if len(combinedSelect) > 1:
                    selSteps = np.diff(combinedSelect)
                    if selSteps.min() == selSteps.max() == 1:
                        combinedSelect = slice(
                            combinedSelect[0], combinedSelect[-1] + 1, 1
                        )

                # Update selector properties
                for selection in actualSelections:
                    getattr(self, "_{}".format(selection))[tk] = combinedSelect

            # Ensure that `self.channel` is compatible w/provided selections: harmonize
            # `self.channel` with what is actually available in selected trials
            if self._dataClass == "SpikeData":
                availChannels = reduce(np.union1d, chanPerTrial)
                chanSelection = wantedChannels[np.isin(wantedChannels, availChannels)].tolist()
                if len(chanSelection) > 1:
                    selSteps = np.diff(chanSelection)
                    if selSteps.min() == selSteps.max() == 1:
                        chanSelection = slice(
                            chanSelection[0], chanSelection[-1] + 1, 1
                        )
                self._channel = chanSelection

            # Finally, prepare new `trialdefinition` array
            self.trialdefinition = data

            return

        # Count how many lists we got
        listCount = 0
        for prop in self._dimProps:
            if isinstance(getattr(self, prop), list):
                listCount += 1

        # Now go through trial-dependent selectors to see if any by-trial selection is a list
        for prop in self._byTrialProps:
            selList = getattr(self, prop)
            if selList is not None:
                for tsel in selList:
                    if isinstance(tsel, list):
                        listCount += 1
                        break

        # If (on a by-trial basis) we have two or more lists, we need fancy indexing
        if listCount >= 2:
            self._useFancy = True

        # Finally, prepare new `trialdefinition` array for objects with `time` dimensions
        if self.time is not None:
            self.trialdefinition = data

        return

    # Legacy support
    def __repr__(self):
        return self.__str__()

    # Make selection readable from the command line
    def __str__(self):

        # Get list of print-worthy attributes
        ppattrs = [attr for attr in self.__dir__() if not attr.startswith("_")]
        # legacy, we have proper `Selector.trials` now
        ppattrs.remove('trial_ids')
        ppattrs.sort()

        # Construct dict of pretty-printable property info
        ppdict = {}
        for attr in ppattrs:
            val = getattr(self, attr)
            if val is not None and attr in self._byTrialProps:
                val = val[0]
            if isinstance(val, slice):
                if val.start is val.stop is None:
                    ppdict[attr] = "all {}{}, ".format(
                        attr, "s" if not attr.endswith("s") else ""
                    )
                elif val.start is None or val.stop is None:
                    ppdict[attr] = "{}-range, ".format(attr)
                else:
                    ppdict[attr] = "{0:d} {1:s}{2:s}, ".format(
                        int(np.ceil((val.stop - val.start) / val.step)),
                        attr,
                        "s" if not attr.endswith("s") else "",
                    )
            elif isinstance(val, (list, TrialIndexer)):
                ppdict[attr] = "{0:d} {1:s}{2:s}, ".format(
                    len(val), attr, "s" if not attr.endswith("s") else ""
                )
            elif np.issubdtype(type(val), np.number):
                ppdict[attr] = "one {0:s}, ".format(attr)
            else:
                ppdict[attr] = ""

        # Construct string for printing
        msg = "Syncopy {} selector with ".format(self._dataClass)
        for pout in ppdict.values():
            msg += pout

        return msg[:-2]
