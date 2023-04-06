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


def setup_logging(spydir=None, session=""):
    """Setup logging on module initialization (in the module root level '__init__.py' file). Should not be called elsewhere."""

    _addLoggingLevel('IMPORTANT', logging.WARNING - 5)  # Add a new custom log level named 'IMPORTANT' between DEBUG and INFO (int value = 25).

    if os.environ.get("SPYLOGDIR"):
        syncopy.__logdir__ = os.path.abspath(os.path.expanduser(os.environ["SPYLOGDIR"]))
    else:
        if spydir is not None:
            syncopy.__logdir__ = os.path.join(spydir, "logs")
        else:
            syncopy.__logdir__ = os.path.join(os.path.expanduser("~"), ".spy", "logs")

    if not os.path.exists(syncopy.__logdir__):
        os.makedirs(syncopy.__logdir__, exist_ok=True)

    loglevel = os.getenv("SPYLOGLEVEL", "IMPORTANT")
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):  # An invalid string was set as the env variable, default to IMPORTANT.
        warnings.warn("Invalid log level set in environment variable 'SPYLOGLEVEL', ignoring and using IMPORTANT instead. Hint: Set one of 'DEBUG', 'IMPORTANT', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.")
        loglevel = "IMPORTANT"

    class HostnameFilter(logging.Filter):
        hostname = platform.node()

        def filter(self, record):
            record.hostname = HostnameFilter.hostname
            return True

    class SessionFilter(logging.Filter):
        def filter(self, record):
            record.session = session
            return True

    # The logger for local/sequential stuff -- goes to terminal and to a file.
    spy_logger = logging.getLogger(loggername)

    datefmt_interactive = '%H:%M:%S'
    datefmt_file = "%Y-%m-%d %H:%M:%S"

    # Interactive formatter: no hostname and session info (less clutter on terminal).
    fmt_interactive = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt_interactive)
    # Log file formatter: with hostname and session info.
    fmt_with_hostname = logging.Formatter('%(asctime)s - %(levelname)s - %(hostname)s - %(session)s: %(message)s',
                                          datefmt_file)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt_interactive)
    spy_logger.addHandler(sh)

    logfile = os.path.join(syncopy.__logdir__, f'syncopy.log')
    fh = logging.FileHandler(logfile)  # The default mode is 'append'.
    fh.addFilter(HostnameFilter())
    fh.addFilter(SessionFilter())
    fh.setFormatter(fmt_with_hostname)
    spy_logger.addHandler(fh)

    spy_logger.setLevel(loglevel)
    spy_logger.info(f"Starting Syncopy session at {datetime.datetime.now().astimezone().isoformat()}.")
    spy_logger.debug(f"Syncopy logger '{loggername}' setup to log to file '{logfile}' at level {loglevel}.")

    # Log to per-host files in parallel code by default.
    # Note that this setup handles only the logger of the current host.
    parloglevel = os.getenv("SPYPARLOGLEVEL", "IMPORTANT")
    numeric_level = getattr(logging, parloglevel.upper(), None)
    if not isinstance(numeric_level, int):  # An invalid string was set as the env variable, use default.
        warnings.warn("Invalid log level set in environment variable 'SPYPARLOGLEVEL', ignoring and using IMPORTANT instead. Hint: Set one of 'DEBUG', 'IMPORTANT', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.")
        parloglevel = "IMPORTANT"
    host = platform.node()
    parallel_logger_name = "syncopy_" + host
    spy_parallel_logger = logging.getLogger(parallel_logger_name)

    logfile_par = os.path.join(syncopy.__logdir__, f'syncopy_{host}.log')
    fhp = logging.FileHandler(logfile_par)  # The default mode is 'append'.
    fhp.addFilter(HostnameFilter())
    fhp.addFilter(SessionFilter())
    spy_parallel_logger.setLevel(parloglevel)

    fhp.setFormatter(fmt_with_hostname)
    spy_parallel_logger.addHandler(fhp)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt_interactive)

    spy_parallel_logger.addHandler(sh)
    spy_parallel_logger.debug(f"Syncopy parallel logger '{parallel_logger_name}' setup to log to file '{logfile_par}' at level {parloglevel}.")


# See https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945#35804945
def _addLoggingLevel(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName) and hasattr(logging, methodName) and hasattr(logging.getLoggerClass(), methodName):
        return  # Setup already complete.

    if hasattr(logging, levelName):
        raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


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


def set_loglevel(level, parallel_level=None):
    """
    Set log level for the loggers.

    Parameters
    ----------
    level: str, one of 'DEBUG', 'IMPORTANT', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
    parallel_level: optional str (same as for 'level' above) of None. If None, the log level of the sequential logger is also used for the parallel logger.
    """
    if parallel_level is None:
        parallel_level = level
    get_logger().setLevel(level)
    get_parallel_logger().setLevel(parallel_level)


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
