# -*- coding: utf-8 -*-
#
# Test shared tools.
#

import syncopy as spy
import copy

class TestTools:

    def test_structdict_shallow_copy(self):
        """Test for fix of issue #394: 'Copying a spy.StructDict returns a dict'."""
        cfg = spy.StructDict()
        cfg.a = 0.5
        cfg.b = "test"
        cfg.c = [1, 2, 3]
        assert type(cfg) == spy.shared.tools.StructDict

        cfg2 = cfg.copy()
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
        my_dict = {'a': 0.5, 'b': 'test', 'c' : [1, 2, 3]}
        cfg = spy.StructDict(my_dict)
        assert type(cfg) == spy.shared.tools.StructDict
        assert cfg.a == 0.5
        assert cfg.b == "test"
        assert cfg.c == [1, 2, 3]

    def test_structdict_from_dict_copy(self):
        my_dict = {'a': 0.5, 'b': 'test', 'c' : [1, 2, 3]}
        cfg = spy.StructDict(my_dict)
        assert type(cfg) == spy.shared.tools.StructDict
        assert cfg.a == 0.5
        assert cfg.b == "test"
        assert cfg.c == [1, 2, 3]

        cfg2 = cfg.copy()
        assert type(cfg2) == spy.shared.tools.StructDict
        assert cfg2.b == "test"

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


if __name__ == '__main__':
    T1 = TestTools()

