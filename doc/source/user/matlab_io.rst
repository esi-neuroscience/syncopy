.. _matlab_io:

Roundtrip from MatLab - FieldTrip to Syncopy and Back
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Data created with Syncopy can be loaded into MATLAB using the `matlab-syncopy
<https://github.com/esi-neuroscience/syncopy-matlab>`_ interface. It's still in early
development and supports only a subset of data classes. Also, the MATLAB
interface does not support loading data that do not fit into local memory.


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
