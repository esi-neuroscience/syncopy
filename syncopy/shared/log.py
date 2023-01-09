# -*- coding: utf-8 -*-
#
# Logging functions for Syncopy.
#
# Note: The logging setup is done in the top-level `__init.py__` file.

import os
import logging
import socket
import syncopy
import warnings


loggername = "syncopy"  # Since this is a library, we should not use the root logger (see Python logging docs).
loglevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

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


def delete_all_logfiles(silent=True):
    """Delete all '.log' files in the Syncopy logging directory.

    The log directory that will be emptied is `syncopy.__logdir__`.
    """
    logdir = syncopy.__logdir__
    num_deleted = 0
    if os.path.isdir(logdir):
        filelist = [ f for f in os.listdir(logdir) if f.endswith(".log") ]
        for f in filelist:
            logfile = os.path.join(logdir, f)
            try:
                os.remove(logfile)
                num_deleted += 1
            except Exception as ex:
                warnings.warn(f"Could not delete log file '{logfile}': {str(ex)}")
    if not silent:
        print(f"Deleted {num_deleted} log files from directory '{logdir}'.")


