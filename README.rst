.. image:: https://raw.githubusercontent.com/esi-neuroscience/syncopy/master/doc/source/_static/syncopy_logo_small.png
	   :alt: Syncopy-Logo

Systems Neuroscience Computing in Python
========================================


|Conda Version| |PyPi Version| |License| |DOI|

.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/esi-syncopy.svg
   :target: https://anaconda.org/conda-forge/esi-syncopy
.. |PyPI version| image:: https://badge.fury.io/py/esi-syncopy.svg
   :target: https://badge.fury.io/py/esi-syncopy
.. |License| image:: https://img.shields.io/github/license/esi-neuroscience/syncopy
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.8191941.svg
   :target: https://doi.org/10.5281/zenodo.8191941

|Master Tests| |Master Coverage|

.. |Master Tests| image:: https://github.com/esi-neuroscience/syncopy/actions/workflows/cov_test_workflow.yml/badge.svg?branch=master
   :target: https://github.com/esi-neuroscience/syncopy/actions/workflows/cov_test_workflow.yml
.. |Master Coverage| image:: https://codecov.io/gh/esi-neuroscience/syncopy/branch/master/graph/badge.svg?token=JEI3QQGNBQ
   :target: https://codecov.io/gh/esi-neuroscience/syncopy

Syncopy aims to be a user-friendly toolkit for *large-scale*
electrophysiology data-analysis in Python. We strive to achieve the following goals:

1. Syncopy is a *fully open source Python* environment for electrophysiology
   data analysis.
2. Syncopy is *scalable* and built for *very large datasets*. It automatically
   makes use of available computing resources and is developed with built-in
   parallelism in mind.
3. Syncopy is *compatible with FieldTrip*. Data and results can be loaded into
   MATLAB and Python, and parameter names and function call syntax are as similar as possible.

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

We recommend to install SynCoPy into a new conda environment:

#. Install the `Anaconda Distribution for your Operating System <https://www.anaconda.com/products/distribution>`_ if you do not yet have it.
#. Start a new terminal.

   * You can do this by starting ```Anaconda navigator```, selecting ```Environments``` in the left tab, selecting the ```base (root)``` environment, and clicking the green play button and then ```Open Terminal```.
   * Alternatively, under Linux, you can just type ```bash``` in your active terminal to start a new session.

You should see a terminal with a command prompt that starts with ```(base)```, indicating that you are
in the conda ```base``` environment.

Now we create a new environment named ```syncopy``` and install syncopy into this environment:

.. code-block:: bash

   conda create -y --name syncopy
   conda activate syncopy
   conda install -y -c conda-forge esi-syncopy

Getting Started
===============
Please visit our `online documentation <http://syncopy.org>`_.

Developer Installation
-----------------------

To get the latest development version, please clone our GitHub repository and change to the `dev` branch. We highly recommend to install into a new conda virtual environment, so that this development version does not interfere with your existing installation.

.. code-block:: bash

   git clone https://github.com/esi-neuroscience/syncopy.git
   cd syncopy/
   conda env create --name syncopy-dev --file syncopy.yml
   conda activate syncopy-dev
   pip install -e .


We recommend to verify your development installation by running the unit tests. You can skip the parallel tests to save some time, the tests should run in about 5 minutes then:


.. code-block:: bash

   python -m pytest -k "not parallel"


You now have a verified developer installation of Syncopy. Please refert to our `contributing guide <https://github.com/esi-neuroscience/syncopy/blob/master/CONTRIBUTING.md>`_ if you want to contribute to Syncopy.

