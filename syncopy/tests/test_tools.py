# -*- coding: utf-8 -*-
#
# Test shared tools.
#

import pytest
import numpy as np
import json
import copy

import syncopy as spy
from syncopy.shared.tools import SerializableDict
from syncopy.shared.errors import SPYError



class TestTools:
    def test_structdict_shallow_copy(self):
        """Test for fix of issue #394: 'Copying a spy.StructDict returns a dict'."""
        cfg = spy.StructDict()
        cfg.a = 0.5
        cfg.b = "test"
        cfg.c = [1, 2, 3]
        assert type(cfg) == spy.shared.tools.StructDict

        cfg2 = cfg.copy(deep=False)
        _ = spy.StructDict()

        assert type(cfg2) == spy.shared.tools.StructDict
        assert cfg.a == 0.5
        assert cfg.b == "test"
        assert cfg.c == [1, 2, 3]

        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.c == cfg.c

        # Check the list was shallow-copied.
        cfg.c.append(4)
        assert cfg2.c == cfg.c

    def test_structdict_from_dict(self):
        my_dict = {"a": 0.5, "b": "test", "c": [1, 2, 3]}
        cfg = spy.StructDict(my_dict)
        assert type(cfg) == spy.shared.tools.StructDict
        assert cfg.a == 0.5
        assert cfg.b == "test"
        assert cfg.c == [1, 2, 3]

    def test_copy_Welch_cfg(self):
        from syncopy.tests.test_welch import TestWelch

        cfg = TestWelch.get_welch_cfg()
        cfg.method = "abs"
        cfg2 = cfg.copy()
        assert type(cfg2) == spy.shared.tools.StructDict
        assert cfg2.method == "abs"

    def test_structdict_shallow_copy_ext(self):
        """Test for fix of issue #394: 'Copying a spy.StructDict returns a dict'."""
        cfg = spy.StructDict()
        cfg.a = 0.5
        cfg.b = "test"
        cfg.c = [1, 2, 3]
        assert type(cfg) == spy.shared.tools.StructDict

        cfg2 = copy.copy(cfg)

        assert type(cfg2) == spy.shared.tools.StructDict
        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.c == cfg.c

        # Check the list was shallow-copied.
        cfg.c.append(4)
        assert cfg2.c == cfg.c

    def test_structdict_deepcopy(self):
        cfg = spy.StructDict()
        cfg.a = 0.5
        cfg.b = "test"
        cfg.c = [1, 2, 3]
        assert type(cfg) == spy.shared.tools.StructDict

        cfg2 = cfg.deepcopy()

        assert type(cfg2) == spy.shared.tools.StructDict
        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.c == cfg.c

        # Check the list was deep-copied.
        cfg.c.append(4)
        assert cfg2.c != cfg.c
        assert 4 in cfg.c
        assert 4 not in cfg2.c
        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.a == 0.5
        assert cfg2.b == "test"
        assert cfg2.c == [1, 2, 3]
        assert cfg.c == [1, 2, 3, 4]

    def test_structdict_deepcopy_ext(self):
        cfg = spy.StructDict()
        cfg.a = 0.5
        cfg.b = "test"
        cfg.c = [1, 2, 3]
        assert type(cfg) == spy.shared.tools.StructDict

        cfg2 = copy.deepcopy(cfg)  # Use copy.deepcopy instead of cfg.deepcopy() this time.

        assert type(cfg2) == spy.shared.tools.StructDict
        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.c == cfg.c

        # Check the list was deep-copied.
        cfg.c.append(4)
        assert cfg2.c != cfg.c
        assert 4 in cfg.c
        assert 4 not in cfg2.c
        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.a == 0.5
        assert cfg2.b == "test"
        assert cfg2.c == [1, 2, 3]
        assert cfg.c == [1, 2, 3, 4]

    def test_structdict_deepcopy_ext_with_explicit_deep(self):
        cfg = spy.StructDict()
        cfg.a = 0.5
        cfg.b = "test"
        cfg.c = [1, 2, 3]
        assert type(cfg) == spy.shared.tools.StructDict

        cfg2 = cfg.copy(deep=True)  # Use copy.deepcopy instead of cfg.deepcopy() this time.

        assert type(cfg2) == spy.shared.tools.StructDict
        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.c == cfg.c

        # Check the list was deep-copied.
        cfg.c.append(4)
        assert cfg2.c != cfg.c
        assert 4 in cfg.c
        assert 4 not in cfg2.c
        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.a == 0.5
        assert cfg2.b == "test"
        assert cfg2.c == [1, 2, 3]
        assert cfg.c == [1, 2, 3, 4]

    def test_structdict_deepcopy_ext_with_implicit_deep(self):
        cfg = spy.StructDict()
        cfg.a = 0.5
        cfg.b = "test"
        cfg.c = [1, 2, 3]
        assert type(cfg) == spy.shared.tools.StructDict

        cfg2 = cfg.copy()  # Use copy.deepcopy instead of cfg.deepcopy() this time.

        assert type(cfg2) == spy.shared.tools.StructDict
        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.c == cfg.c

        # Check the list was deep-copied.
        cfg.c.append(4)
        assert cfg2.c != cfg.c
        assert 4 in cfg.c
        assert 4 not in cfg2.c
        assert cfg2.a == cfg.a
        assert cfg2.b == cfg.b
        assert cfg2.a == 0.5
        assert cfg2.b == "test"
        assert cfg2.c == [1, 2, 3]
        assert cfg.c == [1, 2, 3, 4]


def test_serializable_dict():

    dct = SerializableDict()

    # check standard types
    dct['list'] = [str(i) + 'a' for i in range(10)]
    assert len(dct['list']) == 10
    assert dct['list'][9] == '9a'

    dct['tuple'] = tuple(dct['list'])
    assert len(dct['tuple']) == 10
    assert dct['tuple'][9] == '9a'

    dct['str'] = 'some_string'
    assert dct['str'] == 'some_string'

    dct['list2'] = [i / 239 for i in range(10)]
    assert len(dct['list2']) == 10

    dct['dict'] = {'d1': [1, 2, 3], 'd2': 'g'}
    assert dct['dict']['d1'] == [1, 2, 3]
    assert dct['dict']['d2'] == 'g'

    # not serializable scalars

    num1 = np.int64(12)

    with pytest.raises(TypeError, match="not JSON"):
        json.dumps(num1)

    # gets converted to Python scalars
    dct['int64'] = num1
    assert isinstance(dct['int64'], int)

    # now 1-level deep sequences which are NOT
    # directly serializable

    arr = np.arange(10)

    with pytest.raises(TypeError, match="not JSON"):
        json.dumps(arr)

    # converted to list of python scalars
    dct['arr'] = arr
    assert isinstance(dct['arr'], list)

    lst = [np.int64(12), np.int64(20)]

    with pytest.raises(TypeError, match="not JSON"):
        json.dumps(lst)

    dct['list64'] = lst
    assert isinstance(dct['list64'][1], int)

    # 2 level deep nesting not supported

    lst2 = [[np.int64(12)]]

    with pytest.raises(TypeError, match="not JSON"):
        json.dumps(lst2)

    with pytest.raises(SPYError, match="expected serializable data type"):
        dct['nested2'] = lst2

    # keys also need to be serializable

    with pytest.raises(SPYError, match="Wrong type of key"):
        dct[np.int64(12)] = 'some_string'

if __name__ == "__main__":
    T1 = TestTools()
