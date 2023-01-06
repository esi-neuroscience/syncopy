# -*- coding: utf-8 -*-
#
# Logging functions for Syncopy.
#
# Note: The logging setup is done in the top-level `__init.py__` file.

import logging
import socket


loggername = "syncopy"  # Since this is a library, we should not use the root logger (see Python logging docs).

def get_logger():
    """Get the syncopy root logger.

    Logs to console by default. To be used in everything that runs on the local computer."""
    return logging.getLogger(loggername)

def get_parallel_logger():
    """
    Get a logger for stuff that is run in parallel.

    Logs to a machine-specific file in the SPYLOGDIR by default. To be used in computational routines.
    """
    host = socket.gethostname()
    return logging.getLogger(loggername + "_" + host)

