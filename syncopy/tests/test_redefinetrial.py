# -*- coding: utf-8 -*-
#
# Test resampledata
#

# 3rd party imports
import pytest
import numpy as np

# Local imports
import dask.distributed as dd

from syncopy import redefinetrial
from syncopy.shared.errors import SPYValueError, SPYError, SPYTypeError
from syncopy.datatype import CrossSpectralData


class TestRedefinetrial:

    nTrials = 10
    samplerate = 10

    # equally sized trials, use CrossSpectralData with time axis
    reg_data = CrossSpectralData(data=[np.ones((10, 7, 2, 2)) for _ in range(nTrials)],
                                 samplerate=samplerate)

    irreg_data = reg_data.copy()
    trldef = irreg_data.trialdefinition
    # short 3rd trial
    trldef[2] = [23, 26, -10]
    # longer trial 6 (with overlap)
    trldef[5] = [48, 65, -10]
    irreg_data.trialdefinition = trldef

    # check trial lengths in seconds
    assert np.all(np.diff(reg_data.trialintervals) == 0.9)
    assert not np.all(np.diff(irreg_data.trialintervals) == 0.9)
    # one short trial
    assert np.sum(np.diff(irreg_data.trialintervals) < 0.9) == 1
    # one long trial
    assert np.sum(np.diff(irreg_data.trialintervals) > 0.9) == 1

    def test_user_input(self):
        """ check non compatible arguments are catched """

        # offset and trialdef
        with pytest.raises(SPYError, match='Incompatible input arguments'):
            redefinetrial(self.reg_data, offset=-2, trl=3)
        # relative begin sample and trialdef
        with pytest.raises(SPYError, match='Incompatible input arguments'):
            redefinetrial(self.reg_data, begsample=2000, trl=3)
        # toilim and trialdef
        with pytest.raises(SPYError, match='Incompatible input arguments'):
            redefinetrial(self.reg_data, trl=3, toilim=[1, 2])
        # minlength and toilim
        with pytest.raises(SPYError, match='Incompatible input arguments'):
            redefinetrial(self.reg_data, minlength=2, toilim=[1, 2])
        # relative begin sample and minlength
        with pytest.raises(SPYError, match='Incompatible input arguments'):
            redefinetrial(self.reg_data, begsample=2000, minlength=2)

    def test_offset(self):

        # only pre-stimulus with default offset (-1s or -samplerate)
        assert np.all([self.reg_data.time[i] < 0 for i in range(self.nTrials)])

        # set offset far into positive time
        dummy = redefinetrial(self.reg_data, offset=12)
        assert np.all(dummy.trialdefinition[:, 2] == 12)

        # now only post-stimulus
        assert np.all([dummy.time[i] > 0 for i in range(self.nTrials)])

        # test vector valued offset
        dummy = redefinetrial(self.reg_data, offset=np.arange(self.nTrials) - self.nTrials // 2)
        # now halft the trials should have only post stimulus time axes
        assert np.sum(np.all([dummy.time[i] >= 0 for i in range(self.nTrials)], axis=0)) == 5

        # test exceptions
        with pytest.raises(SPYValueError, match='expected array of length'):
            redefinetrial(self.reg_data, offset=np.arange(self.nTrials - 1))

        with pytest.raises(SPYTypeError, match='expected scalar, array'):
            redefinetrial(self.reg_data, offset='no-number')

        # with trial selection
        dummy = redefinetrial(self.reg_data, trials=np.arange(9),
                              offset=np.arange(1, self.nTrials))
        assert len(dummy.trials) == self.nTrials - 1
        # again only post-stimulus with positive offsets
        assert np.all([dummy.time[i] > 0 for i in range(self.nTrials - 1)])

    def test_minlength(self):
        """ select trials via total lengths in seconds """

        # regular trial length is 10 samples, with samplerate=10 this is 1 second
        # so nothing gets thrown out here
        dummy = redefinetrial(self.reg_data, minlength=1)
        assert len(dummy.trials) == self.nTrials

        # here we should get an empty object as no trial is 2 seconds long
        dummy = redefinetrial(self.reg_data, minlength=2)
        assert dummy.data is None

        # irregular dataset
        # only 1 trial is shorter than 1s
        dummy = redefinetrial(self.irreg_data, minlength=1)
        assert len(dummy.trials) == self.nTrials - 1

        # only 1 trial is longer than 1s
        dummy = redefinetrial(self.irreg_data, minlength=1.5)
        assert len(dummy.trials) == 1

        # if we deselect the longer trial we get an empty object
        dummy = redefinetrial(self.irreg_data, trials=[0, 1, 8], minlength=1.5)
        assert dummy.data is None

        # for very short minimal length all trials get selected
        dummy = redefinetrial(self.irreg_data, minlength=.1)
        assert len(dummy.trials) == self.nTrials

        # test exceptions
        with pytest.raises(SPYTypeError, match='expected scalar'):
            redefinetrial(self.reg_data, minlength='no-number')

        with pytest.raises(SPYTypeError, match='expected scalar'):
            redefinetrial(self.reg_data, minlength=np.arange(10))

        with pytest.raises(SPYValueError, match='expected value to be greater'):
            redefinetrial(self.reg_data, minlength=-.1)

    def test_toilim(self):
        """ select/cut trials via latency window """

        # with default offset all time axes are negative/pre-stimulus [-1, ..., -.1]
        dummy = redefinetrial(self.reg_data, toilim=[-.8, -.2])
        # nothing gets lost here
        assert len(dummy.trials) == self.nTrials
        # now only selected time window
        assert np.all([dummy.time[i] > -1 for i in range(self.nTrials)])
        assert np.all([dummy.time[i] < -.1 for i in range(self.nTrials)])
        # all trials got shortened to 7 samples
        assert np.all([len(trl) == 7 for trl in dummy.trials])

        # sanity check without toilim
        assert not np.all([self.reg_data.time[i] > -1 for i in range(self.nTrials)])

        # positive/post-stimulus latencies don't work as there is no data
        # FIXME: Maybe better to also return empty objects here?!
        with pytest.raises(SPYValueError, match='expected start of latency window < -0.1s'):
            redefinetrial(self.reg_data, toilim=[.2, .8])

        # with new offset the time axis changes into positive/post-stimulus domain
        dummy = redefinetrial(self.reg_data, offset=0)
        # negative/pre-stimulus latency window is now no longer possible
        with pytest.raises(SPYValueError, match='expected end of latency window > 0.0s'):
            redefinetrial(dummy, toilim=[-.8, -.2])
        # but positive latencies now work
        dummy2 = redefinetrial(dummy, toilim=[.2, .8])
        assert np.all([dummy2.time[i] >= .2 for i in range(self.nTrials)])
        assert np.all([dummy2.time[i] <= .8 for i in range(self.nTrials)])
        # again all trials got shortened to 7 samples
        assert np.all([len(trl) == 7 for trl in dummy2.trials])

        # irregular dataset, only longest trial reaches into post-stimulus/positive time
        dummy = redefinetrial(self.irreg_data, toilim=[0.1, .4])
        assert len(dummy.trials) == 1

        # here the short trial gets kicked out
        dummy = redefinetrial(self.irreg_data, toilim=[-.8, -.2])
        assert len(dummy.trials) == self.nTrials - 1

        # test exceptions, array_parser gets hit
        with pytest.raises(SPYValueError, match='expected array of shape'):
            redefinetrial(self.reg_data, toilim=[2, ])
        with pytest.raises(SPYTypeError, match='expected array'):
            redefinetrial(self.reg_data, toilim=-2)

    def test_begin_end_sample(self):
        """ test cutting trials via relative sample numbers """

        # every trial gets shortened by 3 samples (2-9)
        dummy = redefinetrial(self.reg_data, begsample=2, endsample=9)
        assert np.all([len(trl) == 7 for trl in dummy.trials])

        # the irregular trials now all get cut to 10 samples
        dummy = redefinetrial(self.irreg_data, begsample=0, endsample=10)
        assert np.all([len(trl) == 10 for trl in dummy.trials])
        # sanity check
        assert not np.all([len(trl) == 10 for trl in self.irreg_data.trials])

        # can also set a new offset at the same time
        dummy = redefinetrial(self.irreg_data, begsample=0, endsample=10, offset=1)
        assert np.all([len(trl) == 10 for trl in dummy.trials])
        assert np.all([dummy.time[i] > 0 for i in range(self.nTrials)])

        # test exceptions

        # begsample missing
        with pytest.raises(SPYValueError, match='expected both'):
            redefinetrial(self.reg_data, endsample=8)

        # endsample missing
        with pytest.raises(SPYValueError, match='expected both'):
            redefinetrial(self.reg_data, begsample=8)

        # sample numberes are relative to trial start, so negative values are invalid
        with pytest.raises(SPYValueError, match='expected integers >= 0'):
            redefinetrial(self.reg_data, begsample=-2, endsample=2)

        # end > begin
        with pytest.raises(SPYValueError, match='expected endsample > begsample'):
            redefinetrial(self.reg_data, begsample=8, endsample=2)

        # wrong type
        with pytest.raises(SPYTypeError, match='expected scalar or array'):
            redefinetrial(self.reg_data, begsample=2, endsample='d')

        # wrong length
        with pytest.raises(SPYValueError, match='expected same sizes'):
            redefinetrial(self.reg_data, begsample=2, endsample=[2,3])

        # endsample reaches outside data range
        with pytest.raises(SPYValueError, match='expected integers < 10'):
            redefinetrial(self.reg_data, begsample=2, endsample=20)

    def test_trl(self):
        """ setting trialdefinition directly """

        # check that single trial selection works
        dummy = redefinetrial(self.reg_data, trl=[20, 42, 0])
        assert len(dummy.trials) == 1
        assert len(dummy.time[0]) == 22

        # redefine to only 3 irreg trials
        trldef = np.zeros((3, 3))
        trldef[0] = [5, 15, 0]
        trldef[1] = [18, 25, 0]
        trldef[2] = [20, 35, 0]
        dummy = redefinetrial(self.reg_data, trl=trldef)
        assert len(dummy.trials) == 3

        # test invalid input
        with pytest.raises(SPYError, match='Incompatible input arguments'):
            redefinetrial(self.reg_data, trl=trldef, endsample=4)

        # wrong shape
        with pytest.raises(SPYError, match='expected 2-dimensional array'):
            redefinetrial(self.reg_data, trl=trldef[..., None])

        # rest of the exceptions get catched via spy.definetrial

    def test_cfg(self):

        dummy = redefinetrial(self.irreg_data, offset=-5, trials=[0, 5, 7], begsample=0, endsample=6)
        # use cfg from dummy
        dummy2 = redefinetrial(self.irreg_data, dummy.cfg)
        assert dummy == dummy2
        assert np.all([len(trl) == 6 for trl in dummy2.trials])


if __name__ == '__main__':
    T1 = TestRedefinetrial()
