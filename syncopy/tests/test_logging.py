# -*- coding: utf-8 -*-
#
# Test logging.
#

import os

# Local imports
import syncopy as spy


class TestLogging:

    def test_logfile_exists(self):
        logfile = os.path.join(spy.__logdir__, "syncopy.log")
        assert os.path.isfile(logfile)


