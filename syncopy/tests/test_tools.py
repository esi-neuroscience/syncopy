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
        assert type(cfg) == spy.shared.tools.StructDict
        assert type(cfg.copy()) == spy.shared.tools.StructDict

    def test_structdict_deep_copy(self):
        """Test for fix of issue #394: 'Copying a spy.StructDict returns a dict'."""
        cfg = spy.StructDict()
        assert type(cfg) == spy.shared.tools.StructDict
        assert type(copy.deepcopy(cfg)) == spy.shared.tools.StructDict


if __name__ == '__main__':
    T1 = TestTools()

