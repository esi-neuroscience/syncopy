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
    cfg.taper = 'dpss'
    cfg.tapsmofrq = 10
    spec = spy.freqanalysis(cfg, data)


Key Differences between FieldTrip and Syncopy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* FieldTrip has **a lot** more features. Syncopy is still in early development and will
  never cover the rich feature-set of FieldTrip.
* FieldTrip supports **many** data formats. Syncopy currently only supports data import 
  from FieldTrip (see below). 
* Syncopy data objects use disk-streaming and are thus never fully loaded into memory.

Exchanging Data between FieldTrip and Syncopy
---------------------------------------------
Data created with Syncopy can be loaded into MATLAB using the `matlab-syncopy
<https://github.com/esi-neuroscience/syncopy-matlab>`_ interface. It's still in early
development and supports only a subset of data classes. Also, the MATLAB
interface does not support loading data that do not fit into local memory.

MAT-Files can also be imported directly into Syncopy via :func:`~syncopy.load_ft_raw`, at the moment only the ``ft_datatype_raw`` is supported.

Exemplary Workflow: Roundtrip - FieldTrip to Syncopy and Back
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For this illustrative example we start by generating synthetic data in FieldTrip

.. code-block:: matlab

    cfg = [];
    cfg.method  = 'superimposed';
    cfg.fsample = 1000;
    cfg.numtrl  = 13;
    cfg.trllen  = 7;
    cfg.s1.freq = 50;
    cfg.s1.ampl = 1;
    cfg.s1.phase = 0;
    cfg.noise.ampl = 0;
    data = ft_freqsimulation(cfg);
    data.dimord = '{rpt}_label_time';

Next, `download the latest release <https://github.com/esi-neuroscience/syncopy-matlab/releases>`_ 
of Syncopy's MATLAB interface and add the folder containing the `+spy` directory to your 
MATLAB path.  

.. code-block:: matlab

    addpath('/path/to/syncopy-matlab/')

Now, we save the synthetic dataset as Syncopy :class:`~syncopy.AnalogData` dataset in the 
respective user home

.. code-block:: matlab

    cfg = []; cfg.filename = '~/syn_data.analog';
    spy.ft_save_spy(cfg, data)

The previous call generated two files: an HDF5 data-file ``~/syn_data.analog``
and the accompanying JSON meta-data ``~/syn_data.analog.info`` (please refer to 
:ref:`syncopy-data-format` for more information about Syncopy's file format). 

We start an (i)Python session, import Syncopy and use :func:`~syncopy.load` to read the 
data from disk:

.. code-block:: python
      
    import syncopy as spy 
    data = spy.load('~/syn_data.analog')

Now, let's compute a power-spectrum using Syncopy's parallel computing engine:

.. code-block:: python
      
    cfg = spy.get_defaults(spy.freqanalysis)
    cfg.method = 'mtmfft'
    cfg.output = 'pow'
    cfg.parallel = True
    spec = spy.freqanalysis(cfg, data)

.. note::

    Using SLURM on the ESI HPC cluster for datasets this small usually does not 
    yield any performance gain due to the comparatively large overhead of starting 
    a SLURM worker pool compared to the total computation time. 

We save the resulting :class:`~syncopy.SpectralData` object alongside the corresponding 
:class:`~syncopy.AnalogData` source:

.. code-block:: python
      
    spy.save(spec, filename='~/syn_data')

Note that :func:`syncopy.save` automatically appends the appropriate filename 
extension (``.spectral`` in this case). 

Back in MATLAB, we can import the computed spectrum using:

.. code-block:: matlab

    spec = spy.ft_load_spy('~/syn_data.spectral')
