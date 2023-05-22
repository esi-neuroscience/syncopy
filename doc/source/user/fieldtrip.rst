Syncopy for FieldTrip Users
===========================

Syncopy is written in the `Python programming language
<https://www.python.org/>`_ using the `NumPy <https://www.numpy.org/>`_ and
`SciPy <https://scipy.org/>`_ libraries for computing as well as `Dask
<https://dask.org>`_ for parallelization. However, its call signatures and
parameter names are designed to mimic the `MATLAB <https://mathworks.com>`_
analysis toolbox `FieldTrip <http://www.fieldtriptoolbox.org>`_.

The scope of Syncopy is limited to emulate parts of FieldTrip, in particular
spectral analysis of electrophysiology data. Therefore, M/EEG-specific routines
such as loading M/EEG file types, source localization, etc. are currently not
included in Syncopy. For a Python toolbox tailored to M/EEG data analysis, see
for example the `MNE Project <https://www.martinos.org/mne/>`_.

.. contents::
    Contents
    :local:


Exchanging Data between FieldTrip and Syncopy
---------------------------------------------

MAT-Files can be imported directly into Syncopy via :func:`~syncopy.load_ft_raw`, at the moment only the `ft_datatype_raw <https://github.com/fieldtrip/fieldtrip/blob/release/utilities/ft_datatype_raw.m>`_ is supported:

.. autosummary::
   syncopy.load_ft_raw


Key Differences between Syncopy and FieldTrip
---------------------------------------------

Have a look at the :ref:`quick_start` page to quickly walk through a few Syncopy examples.

Data types and handling
^^^^^^^^^^^^^^^^^^^^^^^^

The data in Syncopy is represented as `Python objects <https://python.swaroopch.com/oop.html>`_. So it has **methods** (functions) and **attributes** (data) attached, accessible via the ``.`` operator. Let's have a look at an :class:`~syncopy.AnalogData` example::

  import syncopy as spy

  # red noise AR(1) process with 10 trials and 250 samples
  adata = spy.synthdata.red_noise(alpha=0.9, nTrials=10, nSamples=250)

  # access the filename attribute
  adata.filename

this will print something like:

.. code-block:: bash

   /path/to/.spy/tmp_storage/spy_fe2c_493b3197.analog

Every Syncopy data object has the following attributes:

- ``trials``: returns a **single trial** as :class:`numpy.ndarray` or an **iterable**
- ``channel``: string :class:`numpy.ndarray` of **channel labels**
- ``trialdefinition``: :class:`numpy.ndarray` representing `start`, `stop` and `offset` off each trial
- ``samplerate``: the samplerate in Hz
- ``filename``: the path to the data file on disc
- ``data``: the backing hdf5 dataset. You should not need to interact with this directly.

Each data class can have special `attributes`, for example ``freq`` for :class:`~syncopy.SpectralData`. An extensive overview over all data classes can be found here: :ref:`syncopy-data-classes`.

Functions and methods operating on data, like I/O and plotting can be found at :ref:`data_basics`.

Changing Attributes
~~~~~~~~~~~~~~~~~~~

The attributes of Syncopy data objects typically mirror the `fields` of MatLab `structures`, however they cannot be simply overwritten::

  adata.channel = 3

this gives::

   SPYTypeError: Wrong type of `channel`: expected array_like found int

Syncopy has detailed error handling, and tries to tell you what exactly is wrong. So here, an **array_like** was expected, but a single **int** was the input. **array_like** basically means a sequence type, so :class:`numpy.ndarray` or Python ``list``. Let's try again::

  adata.channel = ['c1', 'c2', 'c3']

Still no good::

  SPYValueError: Invalid value of `channel`: 'shape = (3,)'; expected array of shape (2,)

So in NumPy language that tells us, that Syncopy expected an array with two elements instead of three. Inspecting the ``channel`` attribute::

  adata.channel

.. code-block:: python

   array(['channel1', 'channel2'], dtype='<U8')

we see that we have only two channels in this case, so setting three channel labels indeed makes no sense. Finally with::

  adata.channel = ['c1', 'c2']

we can change the channel labels.



Translating MATLAB Code to Python
---------------------------------
For translating code from MATLAB to Python there are several guides, e.g.

* the `Mathesaurus <http://mathesaurus.sourceforge.net/matlab-numpy.html>`_
* `NumPy for Matlab users <https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html>`_
* `MATLAB to Python - A Migration Guide by Enthought <https://www.enthought.com/white-paper-matlab-to-python>`_

Key Differences between Python and MATLAB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While the above links cover differences between Python and MATLAB to a great
extent, we highlight here what we think are the most important differences:

* Indexing is different - Python array indexing starts at 0:

  >>> x = [1, 2, 3, 4]
  >>> x[0]
  1

  Python ranges are half-open intervals ``[left, right)``, i.e., the right boundary
  is not included:

  >>> list(range(1, 4))
  [1, 2, 3]

* Data in Python is not necessarily copied and may be manipulated in-place:

  >>> x = [1, 2, 3, 4]
  >>> y = x
  >>> x[0] = -1
  >>> y
  [-1, 2, 3, 4]

  To prevent this an explicit copy of a `list`, `numpy.array`, etc. can be requested:

  >>> x = [1, 2,3 ,4]
  >>> y = list(x)
  >>> x[0] = -1
  >>> y
  [1, 2, 3, 4]

* Python's powerful `import system <https://docs.python.org/3/reference/import.html>`_
  allows simple function names (e.g., :func:`~syncopy.load`) without worrying
  about overwriting built-in functions

  >>> import syncopy as spy
  >>> import numpy as np
  >>> spy.load
  <function syncopy.io.load_spy_container.load(filename, tag=None, dataclass=None, checksum=False, mode='r+', out=None)
  >>> np.load
  <function numpy.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')>

* `Project-specific environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
  allow reproducible and customizable work setups.

  .. code-block:: bash

      $ conda activate np17
      $ python -c "import numpy; print(numpy.version.version)"
      1.17.2
      $ conda activate np15
      $ python -c "import numpy; print(numpy.version.version)"
      1.15.4


Translating FieldTrip Calls to Syncopy
--------------------------------------
Using a FieldTrip function in MATLAB usually works via constructing a ``cfg``
``struct`` that contains all necessary configuration parameters:

.. code-block:: matlab

    ft_defaults
    cfg = [];
    cfg.option1 = 'yes';
    cfg.option2 = [10, 20];
    result = ft_something(cfg);

Syncopy emulates this concept using a :class:`syncopy.StructDict` (really just a
slightly modified Python dictionary) that can automatically be filled with
default settings of any function.

.. code-block:: python

    import syncopy as spy
    cfg = spy.get_defaults(spy.something)
    cfg.option1 = 'yes'
    # or
    cfg.option1 = True
    cfg.option2 = [10, 20]
    result = spy.something(cfg)

A FieldTrip Power Spectrum in Syncopy
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

.. code-block:: python

    cfg = spy.get_defaults(spy.freqanalysis)
    cfg.method = 'mtmfft'
    cfg.foilim = [1, 150]
    cfg.output = 'pow'
    cfg.tapsmofrq = 10
    spec = spy.freqanalysis(cfg, data)


Key Differences between FieldTrip and Syncopy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* FieldTrip has **a lot** more features. Syncopy is still in early development and will
  never cover the rich feature-set of FieldTrip.
* FieldTrip supports **many** data formats. Syncopy currently only supports data import
  from FieldTrip (see below).
* Syncopy data objects use disk-streaming and are thus never fully loaded into memory.

Experimental import/export from MatLab
--------------------------------------

See :ref:`matlab_io` for an example.
