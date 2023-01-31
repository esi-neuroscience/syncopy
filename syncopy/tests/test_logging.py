# -*- coding: utf-8 -*-
#
# Test logging.
#

import os
import platform

# Local imports
import syncopy as spy
from syncopy.shared.log import get_logger, get_parallel_logger
from syncopy.shared.errors import SPYLog


class TestLogging:

    def test_seq_logfile_exists(self):
        logfile = os.path.join(spy.__logdir__, "syncopy.log")
        assert os.path.isfile(logfile)

    def test_par_logfile_exists(self):
        par_logfile = os.path.join(spy.__logdir__, f"syncopy_{platform.node()}.log")
        assert os.path.isfile(par_logfile)

    def test_default_log_level_is_important(self):
        # Ensure the log level is at default (that user did not change SPYLOGLEVEL on test system)
        assert os.getenv("SPYLOGLEVEL", "IMPORTANT") == "IMPORTANT"

        logfile = os.path.join(spy.__logdir__, "syncopy.log")
        assert os.path.isfile(logfile)
        num_lines_initial = sum(1 for line in open(logfile)) # The log file gets appended, so it will most likely *not* be empty.

        # Log something with log level info and DEBUG, which should not affect the logfile.
        logger = get_logger()
        logger.info("I am adding an INFO level log entry.")
        SPYLog("I am adding a DEBUG level log entry.", loglevel="DEBUG")

        num_lines_after_info_debug = sum(1 for line in open(logfile))

        assert num_lines_initial == num_lines_after_info_debug

        # Now log something with log level WARNING
        SPYLog("I am adding a WARNING level log entry.", loglevel="WARNING")

        num_lines_after_warning = sum(1 for line in open(logfile))
        assert num_lines_after_warning > num_lines_after_info_debug

    def test_default_parellel_log_level_is_important(self):
        # Ensure the log level is at default (that user did not change SPYLOGLEVEL on test system)
        assert os.getenv("SPYLOGLEVEL", "IMPORTANT") == "IMPORTANT"
        assert os.getenv("SPYPARLOGLEVEL", "IMPORTANT") == "IMPORTANT"

        par_logfile = os.path.join(spy.__logdir__, f"syncopy_{platform.node()}.log")
        assert os.path.isfile(par_logfile)
        num_lines_initial = sum(1 for line in open(par_logfile)) # The log file gets appended, so it will most likely *not* be empty.

        # Log something with log level info and DEBUG, which should not affect the logfile.
        par_logger = get_parallel_logger()
        par_logger.info("I am adding an INFO level log entry.")
        par_logger.debug("I am adding a DEBUG level log entry.")

        num_lines_after_info_debug = sum(1 for line in open(par_logfile))

        assert num_lines_initial == num_lines_after_info_debug

        # Now log something with log level WARNING
        par_logger.important("I am adding a IMPORTANT level log entry.")
        par_logger.warning("This is the last warning.")

        num_lines_after_warning = sum(1 for line in open(par_logfile))
        assert num_lines_after_warning > num_lines_after_info_debug

    def test_log_function_is_in_root_namespace_with_seq(self):
        """Tests sequential logger via spy.log function."""
        assert os.getenv("SPYLOGLEVEL", "IMPORTANT") == "IMPORTANT"

        logfile = os.path.join(spy.__logdir__, "syncopy.log")
        assert os.path.isfile(logfile)
        num_lines_initial = sum(1 for line in open(logfile)) # The log file gets appended, so it will most likely *not* be empty.

        # Log something with log level info and DEBUG, which should not affect the logfile.
        spy.log("I am adding an INFO level log entry.", level="INFO")

        num_lines_after_info_debug = sum(1 for line in open(logfile))
        assert num_lines_initial == num_lines_after_info_debug

        # Now log something with log level WARNING
        spy.log("I am adding a IMPORTANT level log entry.", level="IMPORTANT", par=False)
        spy.log("This is the last warning.", level="IMPORTANT")

        num_lines_after_warning = sum(1 for line in open(logfile))
        assert num_lines_after_warning > num_lines_after_info_debug

    def test_log_function_is_in_root_namespace_with_par(self):
        """Tests parallel logger via spy.log function."""
        assert os.getenv("SPYPARLOGLEVEL", "IMPORTANT") == "IMPORTANT"

        par_logfile = os.path.join(spy.__logdir__, f"syncopy_{platform.node()}.log")
        assert os.path.isfile(par_logfile)
        num_lines_initial = sum(1 for line in open(par_logfile)) # The log file gets appended, so it will most likely *not* be empty.

        # Log something with log level info and DEBUG, which should not affect the logfile.
        spy.log("I am adding an INFO level log entry.", level="INFO", par=True)

        num_lines_after_info_debug = sum(1 for line in open(par_logfile))
        assert num_lines_initial == num_lines_after_info_debug

        # Now log something with log level WARNING
        spy.log("I am adding a IMPORTANT level log entry.", level="IMPORTANT", par=True)
        spy.log("This is the last warning.", level="IMPORTANT", par=True)

        num_lines_after_warning = sum(1 for line in open(par_logfile))
        assert num_lines_after_warning > num_lines_after_info_debug








