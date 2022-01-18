# -*- coding: utf-8 -*-
#
# Test functionality of data selection features in Syncopy
#

# Builtin/3rd party package imports
import pytest
import numpy as np
import inspect

# Local imports
import syncopy.datatype as spd
from syncopy.tests.misc import flush_local_cluster
from syncopy.datatype import AnalogData, SpectralData
from syncopy.datatype.base_data import Selector
from syncopy.datatype.methods.selectdata import selectdata
from syncopy.shared.errors import SPYValueError, SPYTypeError
from syncopy import __acme__
if __acme__:
    import dask.distributed as dd

# Decorator to decide whether or not to run dask-related tests
skip_without_acme = pytest.mark.skipif(
    not __acme__, reason="acme not available")


# The procedure here is:
# (1) test if `Selector` instance was constructed correctly (i.e., indexing tuples
#     look as expected, ordered list -> slice conversion works etc.)
# (2) test if data was correctly selected from source object (i.e., compare shapes,
#     property contents and actual numeric data arrays)
# Multi-selections are not tested here but in the respective class tests (e.g.,
# "time" + "channel" + "trial" `AnalogData` selections etc.)
class TestSelector():

    # Set up "global" parameters for data objects to be tested (we only test
    # equidistant trials here)
    nChannels = 10
    nSamples = 30
    nTrials = 5
    lenTrial = int(nSamples / nTrials) - 1
    nFreqs = 15
    nSpikes = 50
    samplerate = 2.0
    data = {}
    trl = {}

    # Prepare selector results for valid/invalid selections
    selectDict = {}
    selectDict["channel"] = {"valid": (["channel03", "channel01"],
                                       ["channel03", "channel01", "channel01", "channel02"],  # repetition
                                       ["channel01", "channel01", "channel02", "channel03"],  # preserve repetition
                                       [4, 2, 5],
                                       [4, 2, 2, 5, 5],   # repetition
                                       [0, 0, 1, 2, 3],  # preserve repetition, don't convert to slice
                                       range(0, 3),
                                       range(5, 8),
                                       slice(None),
                                       None,
                                       "all",
                                       slice(0, 5),
                                       slice(7, None),
                                       slice(2, 8),
                                       slice(0, 10, 2),
                                       slice(-2, None),
                                       [0, 1, 2, 3],  # contiguous list...
                                       [2, 3, 5]),  # non-contiguous list...
                             "result": ([2, 0],
                                        [2, 0, 0, 1],
                                        [0, 0, 1, 2],
                                        [4, 2, 5],
                                        [4, 2, 2, 5, 5],
                                        [0, 0, 1, 2, 3],
                                        slice(0, 3, 1),
                                        slice(5, 8, 1),
                                        slice(None, None, 1),
                                        slice(None, None, 1),
                                        slice(None, None, 1),
                                        slice(0, 5, 1),
                                        slice(7, None, 1),
                                        slice(2, 8, 1),
                                        slice(0, 10, 2),
                                        slice(-2, None, 1),
                                        slice(0, 4, 1),  # ...gets converted to slice
                                        [2, 3, 5]),  # stays as is
                             "invalid": (["channel200", "channel400"],
                                         ["invalid"],
                                         tuple("wrongtype"),
                                         "notall",
                                         range(0, 100),
                                         slice(80, None),
                                         slice(-20, None),
                                         slice(-15, -2),
                                         slice(5, 1),
                                         [40, 60, 80]),
                             "errors": (SPYValueError,
                                        SPYValueError,
                                        SPYTypeError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError)}

    selectDict["taper"] = {"valid": ([4, 2, 3],
                                     [4, 2, 2, 3],  # repetition
                                     [0, 1, 1, 2, 3],  # preserve repetition, don't convert to slice
                                     range(0, 3),
                                     range(2, 5),
                                     slice(None),
                                     None,
                                     "all",
                                     slice(0, 5),
                                     slice(3, None),
                                     slice(2, 4),
                                     slice(0, 5, 2),
                                     slice(-2, None),
                                     [0, 1, 2, 3],  # contiguous list...
                                     [1, 3, 4]),  # non-contiguous list...
                           "result": ([4, 2, 3],
                                      [4, 2, 2, 3],
                                      [0, 1, 1, 2, 3],
                                      slice(0, 3, 1),
                                      slice(2, 5, 1),
                                      slice(None, None, 1),
                                      slice(None, None, 1),
                                      slice(None, None, 1),
                                      slice(0, 5, 1),
                                      slice(3, None, 1),
                                      slice(2, 4, 1),
                                      slice(0, 5, 2),
                                      slice(-2, None, 1),
                                      slice(0, 4, 1),  # ...gets converted to slice
                                      [1, 3, 4]),  # stays as is
                           "invalid": (["taper_typo", "channel400"],
                                       tuple("wrongtype"),
                                       "notall",
                                       range(0, 100),
                                       slice(80, None),
                                       slice(-20, None),
                                       slice(-15, -2),
                                       slice(5, 1),
                                       [40, 60, 80]),
                           "errors": (SPYValueError,
                                      SPYTypeError,
                                      SPYValueError,
                                      SPYValueError,
                                      SPYValueError,
                                      SPYValueError,
                                      SPYValueError,
                                      SPYValueError,
                                      SPYValueError)}

    # only define valid inputs, the expected (trial-dependent) results are computed below
    selectDict["unit"] = {"valid": (["unit3", "unit1"],
                                    ["unit3", "unit1", "unit1", "unit2"],  # repetition
                                    ["unit1", "unit1", "unit2", "unit3"],  # preserve repetition
                                    [4, 2, 3],
                                    [4, 2, 2, 3],  # repetition
                                    [0, 0, 2, 3],  # preserve repetition, don't convert to slice
                                    range(0, 3),
                                    range(2, 5),
                                    slice(None),
                                    None,
                                    "all",
                                    slice(0, 5),
                                    slice(3, None),
                                    slice(2, 4),
                                    slice(0, 5, 2),
                                    slice(-2, None),
                                    [0, 1, 2, 3],  # contiguous list...
                                    [1, 3, 4]),  # non-contiguous list...
                          "invalid": (["unit7", "unit77"],
                                      tuple("wrongtype"),
                                      "notall",
                                      range(0, 100),
                                      slice(80, None),
                                      slice(-20, None),
                                      slice(-15, -2),
                                      slice(5, 1),
                                      [40, 60, 80]),
                          "errors": (SPYValueError,
                                     SPYTypeError,
                                     SPYValueError,
                                     SPYValueError,
                                     SPYValueError,
                                     SPYValueError,
                                     SPYValueError,
                                     SPYValueError,
                                     SPYValueError)}

    # only define valid inputs, the expected (trial-dependent) results are computed below
    selectDict["eventid"] = {"valid": ([1, 0],
                                       [1, 1, 0],  # repetition
                                       [0, 0, 1, 2],  # preserve repetition, don't convert to slice
                                       range(0, 2),
                                       range(1, 2),
                                       slice(None),
                                       None,
                                       "all",
                                       slice(0, 2),
                                       slice(1, None),
                                       slice(0, 1),
                                       slice(-1, None),
                                       [0, 1]),  # contiguous list...
                             "invalid": (["eventid", "eventid"],
                                         tuple("wrongtype"),
                                         "notall",
                                         range(0, 100),
                                         slice(80, None),
                                         slice(-20, None),
                                         slice(-15, -2),
                                         slice(5, 1),
                                         [40, 60, 80]),
                             "errors": (SPYValueError,
                                        SPYTypeError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError,
                                        SPYValueError)}

    # in the general test routine, only check correct handling of invalid `toi`/`toilim`
    # and `foi`/`foilim` selections - valid selectors are strongly object-dependent
    # and thus tested in separate methods below
    selectDict["toi"] = {"invalid": (["notnumeric", "stillnotnumeric"],
                                     tuple("wrongtype"),
                                     "notall",
                                     range(0, 10),
                                     slice(0, 5),
                                     [0, np.inf],
                                     [np.nan, 1]),
                         "errors": (SPYValueError,
                                    SPYTypeError,
                                    SPYValueError,
                                    SPYTypeError,
                                    SPYTypeError,
                                    SPYValueError,
                                    SPYValueError)}
    selectDict["toilim"] = {"invalid": (["notnumeric", "stillnotnumeric"],
                                        tuple("wrongtype"),
                                        "notall",
                                        range(0, 10),
                                        slice(0, 5),
                                        [np.nan, 1],
                                        [0.5, 1.5 , 2.0],  # more than 2 components
                                        [2.0, 1.5]),  # lower bound > upper bound
                            "errors": (SPYValueError,
                                       SPYTypeError,
                                       SPYValueError,
                                       SPYTypeError,
                                       SPYTypeError,
                                       SPYValueError,
                                       SPYValueError,
                                       SPYValueError)}
    selectDict["foi"] = {"invalid": (["notnumeric", "stillnotnumeric"],
                                     tuple("wrongtype"),
                                     "notall",
                                     range(0, 10),
                                     slice(0, 5),
                                     [0, np.inf],
                                     [np.nan, 1],
                                     [-1, 2],  # out of bounds
                                     [2, 900]),  # out of bounds
                         "errors": (SPYValueError,
                                    SPYTypeError,
                                    SPYValueError,
                                    SPYTypeError,
                                    SPYTypeError,
                                    SPYValueError,
                                    SPYValueError,
                                    SPYValueError,
                                    SPYValueError)}
    selectDict["foilim"] = {"invalid": (["notnumeric", "stillnotnumeric"],
                                        tuple("wrongtype"),
                                        "notall",
                                        range(0, 10),
                                        slice(0, 5),
                                        [np.nan, 1],
                                        [-1, 2],  # lower limit out of bounds
                                        [2, 900],  # upper limit out of bounds
                                        [2, 7, 6],  # more than 2 components
                                        [9, 2]),  # lower bound > upper bound
                            "errors": (SPYValueError,
                                       SPYTypeError,
                                       SPYValueError,
                                       SPYTypeError,
                                       SPYTypeError,
                                       SPYValueError,
                                       SPYValueError,
                                       SPYValueError,
                                       SPYValueError,
                                       SPYValueError)}

    # Generate 2D array simulating an AnalogData array
    data["AnalogData"] = np.arange(1, nChannels * nSamples + 1).reshape(nSamples, nChannels)
    trl["AnalogData"] = np.vstack([np.arange(0, nSamples, nTrials),
                                   np.arange(lenTrial, nSamples + nTrials, nTrials),
                                   np.ones((lenTrial + 1, )),
                                   np.arange(1, lenTrial + 2) * np.pi]).T

    # Generate a 4D array simulating a SpectralData array
    data["SpectralData"] = np.arange(1, nChannels * nSamples * nTrials * nFreqs + 1).reshape(nSamples, nTrials, nFreqs, nChannels)
    trl["SpectralData"] = trl["AnalogData"]

    # Use a fixed random number generator seed to simulate a 2D SpikeData array
    seed = np.random.RandomState(13)
    data["SpikeData"] = np.vstack([seed.choice(nSamples, size=nSpikes),
                                   seed.choice(np.arange(0, nChannels), size=nSpikes),
                                   seed.choice(int(nChannels/2), size=nSpikes)]).T
    trl["SpikeData"] = trl["AnalogData"]

    # Use a triple-trigger pattern to simulate EventData w/non-uniform trials
    data["EventData"] = np.vstack([np.arange(0, nSamples, 2),
                                   np.zeros((int(nSamples / 2), ))]).T
    data["EventData"][1::3, 1] = 1
    data["EventData"][2::3, 1] = 2
    trl["EventData"] = trl["AnalogData"]

    # Define data classes to be used in tests below
    classes = ["AnalogData", "SpectralData", "SpikeData", "EventData"]

    # test `Selector` constructor w/all data classes
    def test_general(self):

        # construct expected results for `DiscreteData` objects defined above
        mapDict = {"unit": "SpikeData", "eventid": "EventData"}
        for prop, dclass in mapDict.items():
            discrete = getattr(spd, dclass)(data=self.data[dclass],
                                            trialdefinition=self.trl[dclass],
                                            samplerate=self.samplerate)
            propIdx = discrete.dimord.index(prop)

            # convert selection from `selectDict` to a usable integer-list
            allResults = []
            for selection in self.selectDict[prop]["valid"]:
                if isinstance(selection, slice):
                    if selection.start is selection.stop is None:
                        selects = [None]
                    else:
                        selects = list(range(getattr(discrete, prop).size))[selection]
                elif isinstance(selection, range):
                    selects = list(selection)
                elif isinstance(selection, str) or selection is None:
                    selects = [None]
                else: # selection is list/ndarray
                    if isinstance(selection[0], str):
                        avail = getattr(discrete, prop)
                    else:
                        avail = np.arange(getattr(discrete, prop).size)
                    selects = []
                    for sel in selection:
                        selects += list(np.where(avail == sel)[0])

                # alternate (expensive) way to get by-trial selection indices
                result = []
                for trial in discrete.trials:
                    if selects[0] is None:
                        res = slice(0, trial.shape[0], 1)
                    else:
                        res = []
                        for sel in selects:
                            res += list(np.where(trial[:, propIdx] == sel)[0])
                        if len(res) > 1:
                            steps = np.diff(res)
                            if steps.min() == steps.max() == 1:
                                res = slice(res[0], res[-1] + 1, 1)
                    result.append(res)
                allResults.append(result)

            self.selectDict[prop]["result"] = tuple(allResults)

        # wrong type of data and/or selector
        with pytest.raises(SPYTypeError):
            Selector(np.empty((3,)), {})
        with pytest.raises(SPYValueError):
            Selector(spd.AnalogData(), {})
        ang = AnalogData(data=self.data["AnalogData"],
                         trialdefinition=self.trl["AnalogData"],
                         samplerate=self.samplerate)
        with pytest.raises(SPYTypeError):
            Selector(ang, ())
        with pytest.raises(SPYValueError):
            Selector(ang, {"wrongkey": [1]})

        # set/clear in-place data selection (both setting and clearing are idempotent,
        # i.e., repeated execution must work, hence the double whammy)
        ang.selectdata(trials=[3, 1])
        ang.selectdata(trials=[3, 1])
        ang.selectdata(clear=True)
        ang.selectdata(clear=True)
        with pytest.raises(SPYValueError) as spyval:
            ang.selectdata(trials=[3, 1], clear=True)
            assert "no data selectors if `clear = True`" in str(spyval.value)

        # go through all data-classes defined above
        for dclass in self.classes:
            dummy = getattr(spd, dclass)(data=self.data[dclass],
                                         trialdefinition=self.trl[dclass],
                                         samplerate=self.samplerate)

            # test trial selection
            selection = Selector(dummy, {"trials": [3, 1]})
            assert selection.trials == [3, 1]
            selected = selectdata(dummy, trials=[3, 1])
            assert np.array_equal(selected.trials[0], dummy.trials[3])
            assert np.array_equal(selected.trials[1], dummy.trials[1])
            assert selected.trialdefinition.shape == (2, 4)
            assert np.array_equal(selected.trialdefinition[:, -1], dummy.trialdefinition[[3, 1], -1])

            for trlSec in [None, "all"]:
                selection = Selector(dummy, {"trials": trlSec})
                assert selection.trials == list(range(len(dummy.trials)))
                selected = selectdata(dummy, trials=trlSec)
                for tk, trl in enumerate(selected.trials):
                    assert np.array_equal(trl, dummy.trials[tk])
                assert np.array_equal(selected.trialdefinition, dummy.trialdefinition)

            with pytest.raises(SPYValueError):
                Selector(dummy, {"trials": [-1, 9]})

            # test "simple" property setters handled by `_selection_setter`
            # for prop in ["eventid"]:
            for prop in ["channel", "taper", "unit", "eventid"]:
                if hasattr(dummy, prop):
                    expected = self.selectDict[prop]["result"]
                    for sk, sel in enumerate(self.selectDict[prop]["valid"]):
                        solution = expected[sk]
                        if dclass == "SpikeData" and prop == "channel":
                            if isinstance(solution, slice):
                                start, stop, step = solution.start, solution.stop, solution.step
                                if start is None:
                                    start = 0
                                elif start < 0:
                                    start = len(dummy.channel) + start
                                if stop is None:
                                    stop = len(dummy.channel)
                                elif stop < 0:
                                    stop = len(dummy.channel) + stop
                                if step not in [None, 1]:
                                    solution = list(range(start, stop))[solution]
                                else:
                                    solution = slice(start, stop, step)

                        # once we're sure `Selector` works, actually select data
                        selection = Selector(dummy, {prop + "s": sel})
                        assert getattr(selection, prop) == solution
                        selected = selectdata(dummy, {prop + "s": sel})

                        # process `unit` and `enventid`
                        if prop in selection._byTrialProps:
                            propIdx = selected.dimord.index(prop)
                            propArr = np.unique(selected.data[:, propIdx]).astype(np.intp)
                            assert set(getattr(selected, prop)) == set(getattr(dummy, prop)[propArr])
                            tk = 0
                            for trialno in range(len(dummy.trials)):
                                if solution[trialno]: # do not try to compare empty selections
                                    assert np.array_equal(selected.trials[tk],
                                                          dummy.trials[trialno][solution[trialno], :])
                                    tk += 1

                        # `channel` is a special case for `SpikeData` objects
                        elif dclass == "SpikeData" and prop == "channel":
                            chanIdx = selected.dimord.index("channel")
                            chanArr = np.arange(dummy.channel.size)
                            assert set(selected.data[:, chanIdx]).issubset(chanArr[solution])
                            assert set(selected.channel) == set(dummy.channel[solution])

                        # everything else (that is not a `DiscreteData` child)
                        else:
                            idx = [slice(None)] * len(dummy.dimord)
                            idx[dummy.dimord.index(prop)] = solution
                            assert np.array_equal(np.array(dummy.data)[tuple(idx)],
                                                  selected.data)
                            assert np.array_equal(getattr(selected, prop),
                                                  getattr(dummy, prop)[solution])

                    # ensure invalid selection trigger expected errors
                    for ik, isel in enumerate(self.selectDict[prop]["invalid"]):
                        with pytest.raises(self.selectDict[prop]["errors"][ik]):
                            Selector(dummy, {prop + "s": isel})
                else:

                    # ensure objects that don't have a `prop` attribute complain
                    with pytest.raises(SPYValueError):
                        Selector(dummy, {prop + "s": [0]})

            # ensure invalid `toi` + `toilim` specifications trigger expected errors
            if hasattr(dummy, "time") or hasattr(dummy, "trialtime"):
                for selection in ["toi", "toilim"]:
                    for ik, isel in enumerate(self.selectDict[selection]["invalid"]):
                        with pytest.raises(self.selectDict[selection]["errors"][ik]):
                            Selector(dummy, {selection: isel})
                # provide both `toi` and `toilim`
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"toi": [0], "toilim": [0, 1]})
            else:
                # ensure objects that don't have `time` props complain properly
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"toi": [0]})
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"toilim": [0]})

            # ensure invalid `foi` + `foilim` specifications trigger expected errors
            if hasattr(dummy, "freq"):
                for selection in ["foi", "foilim"]:
                    for ik, isel in enumerate(self.selectDict[selection]["invalid"]):
                        with pytest.raises(self.selectDict[selection]["errors"][ik]):
                            Selector(dummy, {selection: isel})
                # provide both `foi` and `foilim`
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"foi": [0], "foilim": [0, 1]})
            else:
                # ensure objects without `freq` property complain properly
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"foi": [0]})
                with pytest.raises(SPYValueError):
                    Selector(dummy, {"foilim": [0]})

    def test_continuous_toitoilim(self):

        # this only works w/the equidistant trials constructed above!!!
        selDict = {"toi": (None,  # trivial "selection" of entire contents
                           "all", # trivial "selection" of entire contents
                           [0.5],  # single entry lists
                           [0.6],  # inexact match
                           [1.0, 2.5],  # two disjoint time-points
                           [1.2, 2.7],  # inexact from above
                           [1.9, 2.4],  # inexact from below
                           [0.4, 2.1],  # inexact from below, inexact from above
                           [1.6, 1.9],  # inexact from above, inexact from below
                           [-0.2, 0.6, 0.9, 1.1, 1.3, 1.6, 1.8, 2.2, 2.45, 3.],  # alternating madness
                           [2.0, 0.5, 2.5],  # unsorted list
                           [1.0, 0.5, 0.5, 1.5],  # repetition
                           [0.5, 0.5, 1.0, 1.5], # preserve repetition, don't convert to slice
                           [0.5, 1.0, 1.5]),  # sorted list (should be converted to slice-selection)
                   "toilim": (None,  # trivial "selection" of entire contents
                              "all",  # trivial "selection" of entire contents
                              [0.5, 1.5],  # regular range
                              [1.5, 2.0],  # minimal range (just two-time points)
                              [1.0, np.inf],  # unbounded from above
                              [-np.inf, 1.0])}  # unbounded from below

        # all trials have same time-scale: take 1st one as reference
        trlTime = (np.arange(0, self.trl["AnalogData"][0, 1] - self.trl["AnalogData"][0, 0])
                        + self.trl["AnalogData"][0, 2]) / self.samplerate

        ang = AnalogData(data=self.data["AnalogData"],
                         trialdefinition=self.trl["AnalogData"],
                         samplerate=self.samplerate)
        angIdx = [slice(None)] * len(ang.dimord)
        timeIdx = ang.dimord.index("time")

        # the below check only works for equidistant trials!
        for tselect in ["toi", "toilim"]:
            for timeSel in selDict[tselect]:
                sel = Selector(ang, {tselect: timeSel}).time
                if timeSel is None or timeSel == "all":
                    idx = slice(None)
                else:
                    if tselect == "toi":
                        idx = []
                        for tp in timeSel:
                            idx.append(np.abs(trlTime - tp).argmin())
                    else:
                        idx = np.intersect1d(np.where(trlTime >= timeSel[0])[0],
                                             np.where(trlTime <= timeSel[1])[0])

                # check that correct data was selected (all trials identical, just take 1st one)
                assert np.array_equal(ang.trials[0][idx, :],
                                      ang.trials[0][sel[0], :])
                if not isinstance(idx, slice) and len(idx) > 1:
                    timeSteps = np.diff(idx)
                    if timeSteps.min() == timeSteps.max() == 1:
                        idx = slice(idx[0], idx[-1] + 1, 1)
                result = [idx] * len(ang.trials)

                # check correct format of selector (list -> slice etc.)
                assert np.array_equal(result, sel)

                # perform actual data-selection and ensure identity of results
                selected = selectdata(ang, {tselect: timeSel})
                for trialno in range(len(ang.trials)):
                    angIdx[timeIdx] = result[trialno]
                    assert np.array_equal(selected.trials[trialno],
                                          ang.trials[trialno][tuple(angIdx)])

    # test `toi`/`toilim` selection w/`SpikeData` and `EventData`
    def test_discrete_toitoilim(self):

        # this only works w/the equidistant trials constructed above!!!
        selDict = {"toi": (None,  # trivial "selection" of entire contents
                           "all",  # trivial "selection" of entire contents
                           [0.5],  # single entry lists
                           [0.6],  # inexact match
                           [1.0, 2.5],  # two disjoint time-points
                           [1.2, 2.7],  # inexact from above
                           [1.9, 2.4],  # inexact from below
                           [0.4, 2.1],  # inexact from below, inexact from above
                           [1.6, 1.9],  # inexact from above, inexact from below
                           [-0.2, 0.6, 0.9, 1.1, 1.3, 1.6, 1.8, 2.2, 2.45, 3.],  # alternating madness
                           [2.0, 0.5, 2.5],  # unsorted list
                           [1.0, 0.5, 0.5, 1.5],  # repetition
                           [0.5, 0.5, 1.0, 1.5], # preserve repetition, don't convert to slice
                           [0.5, 1.0, 1.5]),  # sorted list (should be converted to slice-selection)
                   "toilim": (None,  # trivial "selection" of entire contents
                              "all",  # trivial "selection" of entire contents
                              [0.5, 1.5],  # regular range
                              [1.5, 2.0],  # minimal range (just two-time points)
                              [1.0, np.inf],  # unbounded from above
                              [-np.inf, 1.0])}  # unbounded from below

        # all trials have same time-scale for both `EventData` and `SpikeData`: take 1st one as reference
        trlTime = list((np.arange(0, self.trl["SpikeData"][0, 1] - self.trl["SpikeData"][0, 0])
                        + self.trl["SpikeData"][0, 2])/2 )

        # the below method of extracting spikes satisfying `toi`/`toilim` only works w/equidistant trials!
        for dclass in ["SpikeData", "EventData"]:
            discrete = getattr(spd, dclass)(data=self.data[dclass],
                                            trialdefinition=self.trl[dclass],
                                            samplerate=self.samplerate)
            for tselect in ["toi", "toilim"]:
                for timeSel in selDict[tselect]:
                    if isinstance(timeSel, list):
                        smpIdx = []
                        for tp in timeSel:
                            if np.isfinite(tp):
                                smpIdx.append(np.abs(np.array(trlTime) - tp).argmin())
                            else:
                                smpIdx.append(tp)
                    result = []
                    sel = Selector(discrete, {tselect: timeSel}).time
                    selected = selectdata(discrete, {tselect: timeSel})
                    tk = 0
                    for trlno in range(len(discrete.trials)):
                        thisTrial = discrete.trials[trlno][:, 0]
                        if isinstance(timeSel, list):
                            if tselect == "toi":
                                trlRes = []
                                for idx in smpIdx:
                                    trlRes += list(np.where(thisTrial == idx + trlno * self.lenTrial)[0])
                            else:
                                start = smpIdx[0] + trlno * self.lenTrial
                                stop = smpIdx[1] + trlno * self.lenTrial
                                candidates = np.intersect1d(thisTrial[thisTrial >= start],
                                                            thisTrial[thisTrial <= stop])
                                trlRes = []
                                for cand in candidates:
                                    trlRes += list(np.where(thisTrial == cand)[0])
                        else:
                            trlRes = slice(0, thisTrial.size, 1)

                        # ensure that actually selected data is correct
                        assert np.array_equal(discrete.trials[trlno][trlRes, :],
                                              discrete.trials[trlno][sel[trlno], :])
                        if sel[trlno]:
                            assert np.array_equal(selected.trials[tk],
                                                  discrete.trials[trlno][sel[trlno], :])
                            tk += 1

                        if not isinstance(trlRes, slice) and len(trlRes) > 1:
                            sampSteps = np.diff(trlRes)
                            if sampSteps.min() == sampSteps.max() == 1:
                                trlRes = slice(trlRes[0], trlRes[-1] + 1, 1)
                        result.append(trlRes)

                    # check correct format of selector (list -> slice etc.)
                    assert result == sel

    def test_spectral_foifoilim(self):

        # this selection only works w/the dummy frequency data constructed above!!!
        selDict = {"foi": (None,  # trivial "selection" of entire contents,
                           "all",  # trivial "selection" of entire contents
                           [1],  # single entry lists
                           [2.6],  # inexact match
                           [2, 9],  # two disjoint frequencies
                           [7.2, 8.3],  # inexact from above
                           [6.8, 11.9],  # inexact from below
                           [0.4, 13.1],  # inexact from below, inexact from above
                           [1.2, 2.9],  # inexact from above, inexact from below
                           [1.1, 1.9, 2.1, 3.9, 9.2, 11.8, 12.9, 5.1, 13.8],  # alternating madness
                           [2, 1, 11],  # unsorted list
                           [5, 2, 2, 3],  # repetition
                           [1, 1, 2, 3], # preserve repetition, don't convert to slice
                           [2, 3, 4]),  # sorted list (should be converted to slice-selection)
                   "foilim": (None,  # trivial "selection" of entire contents,
                              "all",  # trivial "selection" of entire contents
                              [2, 11],  # regular range
                              [1, 2],  # minimal range (just two-time points)
                              [1.0, np.inf],  # unbounded from above
                              [-np.inf, 12])}  # unbounded from below

        spc = SpectralData(data=self.data['SpectralData'],
                           trialdefinition=self.trl['SpectralData'],
                           samplerate=self.samplerate)
        allFreqs = spc.freq
        spcIdx = [slice(None)] * len(spc.dimord)
        freqIdx = spc.dimord.index("freq")

        for fselect in ["foi", "foilim"]:
            for freqSel in selDict[fselect]:
                sel = Selector(spc, {fselect: freqSel}).freq
                if freqSel is None or freqSel == "all":
                    idx = slice(None)
                else:
                    if fselect == "foi":
                        idx = []
                        for fq in freqSel:
                            idx.append(np.abs(allFreqs - fq).argmin())
                    else:
                        idx = np.intersect1d(np.where(allFreqs >= freqSel[0])[0],
                                             np.where(allFreqs <= freqSel[1])[0])

                # check that correct data was selected (all trials identical, just take 1st one)
                assert np.array_equal(spc.freq[idx], spc.freq[sel])
                if not isinstance(idx, slice) and len(idx) > 1:
                    freqSteps = np.diff(idx)
                    if freqSteps.min() == freqSteps.max() == 1:
                        idx = slice(idx[0], idx[-1] + 1, 1)

                # check correct format of selector (list -> slice etc.)
                assert np.array_equal(idx, sel)

                # perform actual data-selection and ensure identity of results
                selected = selectdata(spc, {fselect: freqSel})
                spcIdx[freqIdx] = idx
                assert np.array_equal(selected.freq, spc.freq[sel])
                for trialno in range(len(spc.trials)):
                    assert np.array_equal(selected.trials[trialno],
                                          spc.trials[trialno][tuple(spcIdx)])

    @skip_without_acme
    def test_parallel(self, testcluster):
        # collect all tests of current class and repeat them in parallel
        client = dd.Client(testcluster)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr != "test_parallel")]
        for test in all_tests:
            getattr(self, test)()
            flush_local_cluster(testcluster)
        client.close()



