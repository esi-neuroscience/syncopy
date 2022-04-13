.. image:: https://raw.githubusercontent.com/esi-neuroscience/syncopy/master/doc/source/_static/syncopy_logo.png
   :alt: Syncopy-Logo

Systems Neuroscience Computing in Python
========================================


|Conda Version| |PyPi Version| |License|

.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/esi-syncopy.svg
   :target: https://anaconda.org/conda-forge/esi-syncopy
.. |PyPI version| image:: https://badge.fury.io/py/esi-syncopy.svg
   :target: https://badge.fury.io/py/esi-syncopy
.. |License| image:: https://img.shields.io/github/license/esi-neuroscience/syncopy

master branch status: |Master Tests| |Master Coverage|

.. |Master Tests| image:: https://github.com/esi-neuroscience/syncopy/actions/workflows/cov_test_workflow.yml/badge.svg?branch=master
   :target: https://github.com/esi-neuroscience/syncopy/actions/workflows/cov_test_workflow.yml
.. |Master Coverage| image:: https://codecov.io/gh/esi-neuroscience/syncopy/branch/master/graph/badge.svg?token=JEI3QQGNBQ
   :target: https://codecov.io/gh/esi-neuroscience/syncopy

dev branch status: |Dev Tests| |Dev Coverage|

.. |Dev Tests| image:: https://github.com/esi-neuroscience/syncopy/actions/workflows/cov_test_workflow.yml/badge.svg?branch=dev
   :target: https://github.com/esi-neuroscience/syncopy/actions/workflows/cov_test_workflow.yml
.. |Dev Coverage| image:: https://codecov.io/gh/esi-neuroscience/syncopy/branch/dev/graph/badge.svg?token=JEI3QQGNBQ
   :target: https://codecov.io/gh/esi-neuroscience/syncopy

Syncopy aims to be a user-friendly toolkit for *large-scale*
electrophysiology data-analysis in Python. We strive to achieve the following goals:

1. Syncopy is a *fully open source Python* environment for electrophysiology
   data analysis.
2. Syncopy is *scalable* and built for *very large datasets*. It automatically
   makes use of available computing resources and is developed with built-in
   parallelism in mind.
3. Syncopy is *compatible with FieldTrip*. Data and results can be loaded into
   MATLAB and Python, parameter names and function call syntax are as similar as possible

Syncopy is developed at the
`Ernst Strüngmann Institute (ESI) gGmbH for Neuroscience in Cooperation with Max Planck Society <https://www.esi-frankfurt.de/>`_
and released free of charge under the
`BSD 3-Clause "New" or "Revised" License <https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_(%22BSD_License_2.0%22,_%22Revised_BSD_License%22,_%22New_BSD_License%22,_or_%22Modified_BSD_License%22)>`_.

Contact
-------
To report bugs or ask questions please use our `GitHub issue tracker <https://github.com/esi-neuroscience/syncopy/issues>`_.
For general inquiries please contact syncopy (at) esi-frankfurt.de.

Installation
============
Syncopy is available on pip

.. code-block:: bash

   pip install esi-syncopy

For using SynCoPy's parallel processing capabilities, `ACME <https://github.com/esi-neuroscience/acme>`_ is required

.. code-block:: bash

   conda install -c conda-forge esi-acme

To get the latest development version, please clone our GitHub repository:

.. code-block:: bash

   git clone https://github.com/esi-neuroscience/syncopy.git
   cd syncopy/
   pip install -e .

Getting Started
===============
Please visit our `online documentation <http://syncopy.org>`_.
