Syncopy for FieldTrip users
===========================

Syncopy is written in the `Python programming language
<https://www.python.org/>`_ using the `NumPy <https://www.numpy.org/>`_ and
`SciPy <https://scipy.org/>`_ libraries for computing as well as `Dask
<https://dask.org>`_ for parallelization. However, it's call signatures and
parameter names are designed to mimick the `MATLAB <https://mathworks.com>`_
analysis toolbox `FieldTrip <http://www.fieldtriptoolbox.org>`_.

The scope of Syncopy is limited to only parts of FieldTrip, in particular
spectral analysis of electrophysiology data. Therefore, M/EEG-specific routines
such as loading M/EEG file types, source localization, ..., are currently not
covered by Syncopy. For a Python toolbox tailored to M/EEG data analysis, see
for example the `MNE Project <https://www.martinos.org/mne/>`_.

.. contents::
    Contents
    :local:

Translating MATLAB code to Python
---------------------------------

For translating code from MATLAB to Python there are several guides, e.g.

* the `Mathesaurus <http://mathesaurus.sourceforge.net/matlab-numpy.html>`_
* `NumPy for Matlab users <https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html>`_
* `MATLAB to Python - A Migration Guide by Enthought <https://www.enthought.com/white-paper-matlab-to-python>`_

Key differences between Python and MATLAB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the above links cover differences between Python and MATLAB to a great
extent, we highlight here what we think are the most important differences:

* Indexing is different: Python array indexing starts at 0. The end of a range
  in Python is not included
* Data in Python is not necessarily copied and may be manipulated in-place.
* The powerful `import system of Python <https://docs.python.org/3/reference/import.html>`_
  allows simple function names (e.g., :func:`~syncopy.load`) without worrying
  about overwriting built-in functions.
* `Project-specific environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
  allow reproducable and customizable working environments.

Translating FieldTrip calls to Syncopy
--------------------------------------

Using a FieldTrip function in MATLAB usually works via constructing a ``cfg``
``struct`` that contains all configured parameters:

.. code-block:: matlab

    ft_defaults
    cfg = [];
    cfg.option1 = 'yes';
    cfg.option2 = [10, 20];
    result = ft_something(cfg);

In Syncopy this struct is a Python dictionary that can automatically be filled
with the defaults of any function.

.. code-block:: python

    import syncopy as spy
    cfg = spy.get_defaults(spy.something)
    cfg.option1 = 'yes'
    # or
    cfg.option1 = True
    cfg.option2 = [10, 20]
    result = spy.something(cfg=cfg)

A FieldTrip power spectrum in Syncopy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For example, a power spectrum calculated with FieldTrip via

.. code-block:: matlab
      
    cfg = [];
    cfg.method = 'mtmfft';
    cfg.foilim = [1 150];
    cfg.output = 'pow';
    cfg.taper = 'dpss';
    cfg.tapsmofrq = 10;
    spec = ft_freqanalysis(cfg, data)

can be computed in Syncopy with

.. code-block:: matlab
      
    cfg = spy.get_defaults(spy.freqanalysis)
    cfg.method = 'mtmfft';
    cfg.foilim = [1, 150];
    cfg.output = 'pow';
    cfg.taper = 'dpss';
    cfg.tapsmofrq = 10;
    spec = spy.freqanalysis(cfg, data)


Key differences between FieldTrip and Syncopy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* FieldTrip has more features. Syncopy is still in early development and will
  never cover the rich featureset of FieldTrip.
* FieldTrip supports many data formats. Syncopy
* Syncopy data objects are never fully loaded into memory.



Exchanging data between FieldTrip and Syncopy
---------------------------------------------

Data created with Syncopy can be loaded into MATLAB using the `matlab-syncopy
<http://git.esi.local/it/matlab-syncopy>`_ interface. It's still in early
development and supports only a subset of data classes. Also, the MATLAB
interface does not support loading data larger than local memory.
