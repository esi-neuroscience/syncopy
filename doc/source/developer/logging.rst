.. _syncopy-logging:

Controlling Logging in Syncopy
===============================

Syncopy uses the `Python logging module <https://docs.python.org/3/library/logging.html>`_ for logging. It uses two different loggers:
one for code that runs on the local machine, and another one for logging the parallelelized code that
is run by the remote workers in a high performance computing (HPC) cluster environment.

Logging code that runs locally
-------------------------------

For all code that is run on the local machine, Syncopy logs to a logger named `'syncopy'` which is handled by the console.

To adapt the local logging behaviour of Syncopy, one can configure the logger as explained in the documentation for the logging module, e.g., in your application that uses Syncopy:

.. code-block:: python

   import syncopy
   import logging
   # Get the logger used by syncopy
   logger = logging.getLogger('syncopy')

   # Change the log level:
   logger.setLevel(logging.DEBUG)

   # Make it log to a file:
   fh = logging.FileHandler('syncopy_log_within_my_app.log')
   logger.addHandler(fh)

   # The rest of your application code goes here.


Logging code that potentially runs remotely
--------------------------------------------

The parallel code that performs the heavy lifting on the Syncopy data will be executed on remote machines (cluster nodes) when Syncopy is run in an HPC environment. Therefore,
special handling is required for these parts of the code, and we need to log to one log file per remote machine to avoid race conditions and



Log levels
-----------

The default log level is for the Syncopy logger is `'logging.WARNING'` (from now on referred to as `'WARNING'`). This means that you will not see any Syncopy messages below that threshold, i.e., messages printed with log levels `'DEBUG'` and `'INFO'`. To change the log level, you can either use the logging API in your application code as explained above, or set the environment variable `'SPYLOGLEVEL'` to one of the values supported by the logging module, e.g., 'CRITICAL', 'WARNING', 'INFO', or 'DEBUG'. See the `official docs of the logging module <https://docs.python.org/3/library/logging.html#levels>`_ for details on the supported log levels.
