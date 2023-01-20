# -*- coding: utf-8 -*-
#
# Logging functions for Syncopy.
#
# Note: The logging setup is done in the top-level `__init.py__` file.

import os
import sys
import logging
import socket
import syncopy
import warnings
import datetime
import platform
import getpass


loggername = "syncopy"  # Since this is a library, we should not use the root logger (see Python logging docs).
loglevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']


def setup_logging():

    # Setup logging.

    # default path ONLY relevant for ESI Frankfurt
    csHome = "/cs/home/{}".format(getpass.getuser())
    if os.environ.get("SPYLOGDIR"):
        syncopy.__logdir__ = os.path.abspath(os.path.expanduser(os.environ["SPYLOGDIR"]))
    else:
        if os.path.exists(csHome):
            syncopy.__logdir__ = os.path.join(csHome, ".spy", "logs")
        else:
            syncopy.__logdir__ = os.path.join(os.path.expanduser("~"), ".spy", "logs")

    if not os.path.exists(syncopy.__logdir__):
        os.makedirs(syncopy.__logdir__, exist_ok=True)

    loglevel = os.getenv("SPYLOGLEVEL", "WARNING")
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):  # An invalid string was set as the env variable, default to WARNING.
        warnings.warn("Invalid log level set in environment variable 'SPYLOGLEVEL', ignoring and using WARNING instead. Hint: Set one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.")
        loglevel = "WARNING"

    # The logger for local/sequential stuff -- goes to terminal and to a file.
    spy_logger = logging.getLogger('syncopy')
    fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    spy_logger.addHandler(sh)

    logfile = os.path.join(syncopy.__logdir__, f'syncopy.log')
    fh = logging.FileHandler(logfile)  # The default mode is 'append'.
    fh.setFormatter(fmt)
    spy_logger.addHandler(fh)

    spy_logger.setLevel(loglevel)
    spy_logger.debug(f"Starting Syncopy session at {datetime.datetime.now().astimezone().isoformat()}.")
    spy_logger.info(f"Syncopy log level set to: {loglevel}.")

    # Log to per-host files in parallel code by default.
    # Note that this setup handles only the logger of the current host.
    parloglevel = os.getenv("SPYPARLOGLEVEL", loglevel)
    numeric_level = getattr(logging, parloglevel.upper(), None)
    if not isinstance(numeric_level, int):  # An invalid string was set as the env variable, use default.
        warnings.warn("Invalid log level set in environment variable 'SPYPARLOGLEVEL', ignoring and using WARNING instead. Hint: Set one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.")
        parloglevel = "WARNING"
    host = platform.node()
    parallel_logger_name = "syncopy_" + host
    spy_parallel_logger = logging.getLogger(parallel_logger_name)

    class HostnameFilter(logging.Filter):
        hostname = platform.node()

        def filter(self, record):
            record.hostname = HostnameFilter.hostname
            return True

    logfile_par = os.path.join(syncopy.__logdir__, f'syncopy_{host}.log')
    fhp = logging.FileHandler(logfile_par)  # The default mode is 'append'.
    fhp.addFilter(HostnameFilter())
    spy_parallel_logger.setLevel(parloglevel)
    fmt_with_hostname = logging.Formatter('%(asctime)s - %(levelname)s - %(hostname)s: %(message)s')
    fhp.setFormatter(fmt_with_hostname)
    spy_parallel_logger.addHandler(fhp)
    sh = logging.StreamHandler(sys.stdout)
    spy_parallel_logger.addHandler(sh)
    spy_parallel_logger.info(f"Syncopy parallel logger '{parallel_logger_name}' setup to log to file '{logfile_par}' at level {loglevel}.")


def get_logger():
    """Get the syncopy root logger.

    Logs to console by default. To be used in everything that runs on the local computer."""
    return logging.getLogger(loggername)


def get_parallel_logger():
    """
    Get a logger for stuff that is run in parallel.

    Logs to a machine-specific file in the SPYLOGDIR by default. To be used in computational routines.

    The log directory used is `syncopy.__logdir__`. It can be changed by setting the environment variable SPYLOGDIR before running an application that uses Syncopy.
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


