.. _syncopy-logging:

Controlling Logging in Syncopy
===============================

Syncopy uses the `Python logging module <https://docs.python.org/3/library/logging.html>`_ for logging, and logs to a logger named `'syncopy'` which is handled by the console.

To adapt the logging behaviour of Syncopy, one can configure the logger as explained in the documentation for the logging module. E.g.:

.. code-block:: python

   import syncopy
   import logging
   # Get the logger used by syncopy
   logger = logging.getLogger('syncopy')

   # Change the log level:
   logger.setLevel(logging.DEBUG)

   # Make it log to a file instead of the console:
   fh = logging.FileHandler('syncopy_log_within_my_app.log')
   logger.addHandler(fh)



The default log level is for the Syncopy logger is `'WARNING'`. To change the log level, you can either use the logging API in your application code as explained above, or set the environment variable `'SPYLOGLEVEL'` to one of the values supported by the logging module, e.g., 'CRITICAL', 'WARNING', 'INFO', or 'DEBUG'. See the `official docs <https://docs.python.org/3/library/logging.html#levels>`_ for details on the supported log levels.
