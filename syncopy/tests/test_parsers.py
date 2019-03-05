# -*- coding: utf-8 -*-
#
# Created: 2019-03-05 16:22:56
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2019-03-05 17:14:14>

from syncopy import (spy_io_parser, spy_scalar_parser, spy_array_parser,
                     spy_data_parser, spy_json_parser, spy_get_defaults)
from syncopy import SPYValueError, SPYTypeError
from syncopy import AnalogData, SpectralData

import pytest
import numpy as np


def test_spy_io_parser():
    # FIXME: Write tests for io_parser
    assert True


class TestSpyScalarParser(object):
    def test_none(self):
        with pytest.raises(SPYTypeError):
            spy_scalar_parser(None, varname="value",
                              ntype="int_like", lims=[10, 1000])

    def test_within_limits(self):
        value = 440
        spy_scalar_parser(value, varname="value",
                          ntype="int_like", lims=[10, 1000])

        freq = 2        # outside bounds
        with pytest.raises(SPYValueError):
            spy_scalar_parser(freq, varname="freq",
                              ntype="int_like", lims=[10, 1000])

    def test_integer_like(self):
        freq = 440.0
        spy_scalar_parser(freq, varname="freq",
                          ntype="int_like", lims=[10, 1000])

        # not integer-like
        freq = 440.5
        with pytest.raises(SPYValueError):
            spy_scalar_parser(freq, varname="freq",
                              ntype="int_like", lims=[10, 1000])

    def test_string(self):
        freq = '440'
        with pytest.raises(SPYTypeError):
            spy_scalar_parser(freq, varname="freq",
                              ntype="int_like", lims=[10, 1000])

    def test_complex_valid(self):
        value = complex(2, -1)
        spy_scalar_parser(value, lims=[-3, 5])  # valid

    def test_complex_invalid(self):
        value = complex(2, -1)
        with pytest.raises(SPYValueError):
            spy_scalar_parser(value, lims=[-3, 1])


class TestArrayParser(object):
    time = np.linspace(0, 10, 100)

    def test_none(self):
        with pytest.raises(SPYTypeError):
            spy_array_parser(None, varname="time")

    def test_1d_ndims(self):
        # valid ndims
        spy_array_parser(self.time, varname="time", dims=1)

        # invalid ndims
        with pytest.raises(SPYValueError):
            spy_array_parser(self.time, varname="time", dims=2)

    def test_1d_shape(self):
        # valid shape
        spy_array_parser(self.time, varname="time", dims=(100,))

        # invalid shape
        with pytest.raises(SPYValueError):
            spy_array_parser(self.time, varname="time", dims=(100, 1))

    def test_1d_newaxis(self):
        # Artificially appending a singleton dimension to `time` does not affect
        # parsing:
        time = self.time[:, np.newaxis]
        assert time.shape == (100, 1)
        spy_array_parser(time, varname="time", dims=1)
        spy_array_parser(time, varname="time", dims=(100,))

    def test_1d_lims(self):
        # valid lims
        spy_array_parser(self.time, varname="time", lims=[0, 10])
        # invalid lims
        with pytest.raises(SPYValueError):
            spy_array_parser(self.time, varname="time", lims=[0, 5])

    def test_ntype(self):
        # string
        with pytest.raises(SPYTypeError):
            spy_array_parser(str(self.time), varname="time", ntype="numeric")
        # float32 instead of expected float64
        with pytest.raises(SPYValueError):
            spy_array_parser(np.float32(self.time), varname="time",
                             ntype='float64')

    def test_character_list(self):
        channels = np.array(["channel1", "channel2", "channel3"])
        spy_array_parser(channels, varname="channels", dims=1)
        spy_array_parser(channels, varname="channels", dims=(3,))
        with pytest.raises(SPYValueError):
            spy_array_parser(channels, varname="channels", dims=(4,))


class TestDataParser(object):
    data = AnalogData()

    def test_none(self):
        with pytest.raises(SPYTypeError):
            spy_data_parser(None)

    def test_dataclass(self):
        # valid
        spy_data_parser(self.data, dataclass=AnalogData)

        # invalid
        with pytest.raises(SPYTypeError):
            spy_data_parser(self.data, dataclass=SpectralData)

    def test_empty(self):
        with pytest.raises(SPYValueError):
            spy_data_parser(self.data, empty=False)

        # FIXME: fill in some data and test for non-emptiness

    def test_dimord(self):
        # FIXME: fill in some data and test dimord
        assert True


def func(input, keyword=None):
    """ Test function for get_defaults test """
    pass


def test_spy_get_defaults():
    assert(spy_get_defaults(func) == {"keyword": None})


def test_spy_json_parser():
    # FIXME: implement testing for json parser
    assert True
