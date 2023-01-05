.. _syncopy-logging:

Controlling Logging in Syncopy
===============================

Syncopy uses the `Python logging module <https://docs.python.org/3/library/logging.html>`_ for logging, and logs to a logger named `'syncopy'`.

To adapt the logging behaviour of Syncopy, one can configure the logger as explained in the documentation for the logging module.


The default log level is `'WARNING'`. To change the log level, one can either use the logging API (see above), or set the environment variable `'SYNCOPY_LOGLEVEL'` to one of the values supported by the logging module, e.g., 'CRITICAL', 'WARNING', 'INFO', or 'DEBUG'. See the `official docs <https://docs.python.org/3/library/logging.html#levels>`_ for details on the supported log levels.
