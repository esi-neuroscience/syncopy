# -*- coding: utf-8 -*-
#
# Test .info property of BaseData
#

import pytest
import numpy as np
import tempfile
import os

# Local imports
import syncopy as spy
from syncopy.shared.tools import SerializableDict
from syncopy.shared.errors import SPYTypeError


class TestInfo:

    # serializable dict
    ok_dict = {'sth': 4, 'important': [1, 2],
               'to': {'v1': 2}, 'remember': 'need more coffe'}
    # non-serializable dict
    ns_dict = {'sth': 4, 'not_serializable': {'v1': range(2)}}
    # dict with non-serializable keys
    ns_dict2 = {range(2) : 'small_range', range(1000) : 'large_range'}

    # test setter
    def test_property(self):

        # as .info is a basedata property,
        # testing for one derived class should suffice
        adata = spy.AnalogData([np.ones((3, 1))], samplerate=1)

        # attach some aux. info
        adata.info = self.ok_dict

        # got converted into SerializableDict
        # so testing this makes sense
        assert isinstance(adata.info, SerializableDict)
        assert adata.info == self.ok_dict

        # that is not allowed (akin to cfg)
        with pytest.raises(SPYTypeError, match="expected dictionary-like"):
            adata.info = None

        # clear with empty dict
        adata.info = {}
        assert len(adata.info) == 0
        assert len(self.ok_dict) != 0

        # test we're catching non-serializable dictionary entries
        with pytest.raises(SPYTypeError, match="expected serializable data type"):
            adata.info['new-var'] = np.arange(3)
        with pytest.raises(SPYTypeError, match="expected serializable data type"):
            adata.info = self.ns_dict

        # test that we also catch non-serializable keys
        with pytest.raises(SPYTypeError, match="expected serializable data type"):
            adata.info = self.ns_dict2

        # this interestingly still does NOT work (numbers are np.float64):
        with pytest.raises(SPYTypeError, match="expected serializable data type"):
            adata.info['new-var'] = list(np.arange(3))

        # even this.. numbers are still np.int64
        with pytest.raises(SPYTypeError, match="expected serializable data type"):
            adata.info['new-var'] = list(np.arange(3, dtype=int))

        # this then works, hope is that users don't abuse it
        adata.info['new-var'] = list(np.arange(3, dtype=float))
        assert np.allclose(adata.info['new-var'], np.arange(3))

    # test aux. info dict saving and loading
    def test_io(self):
        with tempfile.TemporaryDirectory() as tdir:

            fname = os.path.join(tdir, "dummy")
            dummy = spy.AnalogData([np.ones((3, 1))], samplerate=1)

            # attach some aux. info
            dummy.info = self.ok_dict
            spy.save(dummy, fname)
            del dummy

            dummy2 = spy.load(fname)
            assert dummy2.info == self.ok_dict


if __name__ == '__main__':
    T1 = TestInfo()
