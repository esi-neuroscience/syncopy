# -*- coding: utf-8 -*-
#
# Test logging.
#

import os

# Local imports
import syncopy as spy
from syncopy.shared.log import get_logger
from syncopy.shared.errors import SPYLog


class TestLogging:

    def test_logfile_exists(self):
        logfile = os.path.join(spy.__logdir__, "syncopy.log")
        assert os.path.isfile(logfile)

    def test_default_log_level_is_warning(self):
        logfile = os.path.join(spy.__logdir__, "syncopy.log")
        assert os.path.isfile(logfile)
        num_lines_bofore = sum(1 for line in open(logfile))

        # Log something with log level info and DEBUG, which should not affect the logfile.
        logger = get_logger()
        logger.info("I am adding an INFO level log entry.")
        SPYLog("I am adding a DEBUG level log entry.", loglevel="DEBUG")

        num_lines_after_info_debug = sum(1 for line in open(logfile))

        assert num_lines_bofore == num_lines_after_info_debug

        # Now log something with log level WARNING
        SPYLog("I am adding a WARNING level log entry.", loglevel="WARNING")

        num_lines_after_warning = sum(1 for line in open(logfile))
        assert num_lines_after_info_debug + 1 == num_lines_after_warning





