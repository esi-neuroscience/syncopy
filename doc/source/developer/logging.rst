.. _syncopy-logging:

Controlling Logging in Syncopy
===============================

Syncopy uses the `Python logging module <https://docs.python.org/3/library/logging.html>`_ for logging. It uses two different loggers:
one for code that runs on the local machine, and another one for logging the parallelelized code that
is run by the remote workers in a high performance computing (HPC) cluster environment.


Log levels
-----------

The default log level is for the Syncopy logger is `'logging.WARNING'` (from now on referred to as `'WARNING'`). This means that you will not see any Syncopy messages below that threshold, i.e., messages printed with log levels `'DEBUG'` and `'INFO'`. To change the log level, you can either use the logging API in your application code as explained below, or set the environment variable `'SPYLOGLEVEL'` to one of the values supported by the logging module, e.g., 'CRITICAL', 'WARNING', 'INFO', or 'DEBUG'. See the `official docs of the logging module <https://docs.python.org/3/library/logging.html#levels>`_ for details on the supported log levels.


Log file location
-----------------

All Syncopy log files are saved in a configurable directory which we refer to as `SPYLOGDIR`. By default, `SPYLOGDIR` is set to the directory `.spy/logs/` in your home directory (accessible as `~/.spy/logs/` under Linux and Mac OS), and it can be adapted by setting the environment variable `SPYLOGDIR` before running your application.

E.g., if your Python script using Syncopy is `~/neuro/paperfig1.py`, you can set the log level and log directory on the command line like this in the Bash shell:

.. code-block:: shell
   export SPYLOGDIR=/tmp/spy
   export SPYLOGLEVEL=DEBUG
   ~/neuro/paperfig1.py


Logging code that runs locally
-------------------------------

For all code that is run on the local machine, Syncopy logs to a logger named `'syncopy'` which is handled by both the console and the logfile `'SPYLOGDIR/syncopy.log'`.

To adapt the local logging behaviour of Syncopy, one can configure the logger as explained in the documentation for the logging module, e.g., in your application that uses Syncopy:

.. code-block:: python

   import syncopy
   import logging
   # Get the logger used by syncopy
   logger = logging.getLogger('syncopy')

   # Change the log level:
   logger.setLevel(logging.DEBUG)

   # Add another handler that logs to a file:
   fh = logging.FileHandler('syncopy_debug_log.log')
   logger.addHandler(fh)

   logger.info("My app starts now.")
   # The rest of your application code goes here.


Logging code that potentially runs remotely
--------------------------------------------

The parallel code that performs the heavy lifting on the Syncopy data (i.e., what we call `compute functions`) will be executed on remote machines when Syncopy is run in an HPC environment. Therefore,
special handling is required for these parts of the code, and we need to log to one log file per remote machine.

Syncopy automatically configures a suitable logger named `syncopy_<host>` on each host, where `<host>` is the hostname. Each of these loggers is attached to the respective logfile `'SPYLOGDIR/syncopy_<host>.log'`, where `<host>` is the hostname, which ensures that logging works properly even if you log into the same directory on all remote machines (e.g., a home directory that is mounted on all machines via a network file system).

Here is how to log with the remote logger:

.. code-block:: python

   import syncopy
   import logging, platform

   # ...
   # In some cF or backend function:
   par_logger = logging.getLogger("syncopy_" + platform.node())
   par_logger.info("Code run on remote machine is being run.")

This is all you need to do. If you want to configure different log levels for the remote logger and the local one, you can configure the environment variable `SPYPARLOGLEVEL` in addition to `SPYLOGLEVEL`.
