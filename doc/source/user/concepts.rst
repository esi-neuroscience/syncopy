**************
Basic Concepts
**************

Using Syncopy usually entails writing Python analysis scripts operating on a given list of data files. For new users we prepared a :ref:`quick_start`. Here we want to present the general concepts behind Syncopy.

Data analysis pipelines are inspired by the well established and feature-rich 
`MATLAB <https://mathworks.com>`_ toolbox `FieldTrip <http://www.fieldtriptoolbox.org>`_.
Syncopy aims to emulate FieldTrip's basic usage concepts.

.. contents:: Topics covered
   :local:

.. _workflow:

General Workflow
----------------

A typical analysis workflow with Syncopy might look like this:

.. image:: workFlow.png

	  
We start with data import (or simply loading if already in ``.spy`` format) which will create one of Syncopy's dataypes like :class:`~syncopy.AnalogData`. Then actual (parallel) processing of the data is triggered by calling a *meta-function* (see also below), for example :func:`~syncopy.freqanalysis`. An analysis output often results in a different datatype, e.g. :class:`~syncopy.SpectralData`. All indicated methods (:func:`~syncopy.show`, :func:`~syncopy.singlepanelplot` and :func:`~syncopy.save`) for data access are available for all of Syncopy's datatypes. Hence, at any processing step the data can be plotted, NumPy :class:`~numpy.ndarray`'s extracted or (intermediate) results saved to disc as ``.spy`` containers. 

.. note::
   Have a look at :doc:`Data Basics <data_basics>` for further details about Syncopy's data formats and interfaces


Memory Management
~~~~~~~~~~~~~~~~~

One of the key concepts of Syncopy is mindful computing resource management, especially keeping a low **memory footprint**. In the depicted workflow, data processed :blue:`on disc` is indicated in :blue:`blue`, whereas potentially :red:`memory exhausting operations` are indicated in :red:`red`. So care has to be taken when using :func:`~syncopy.show` or the plotting routines :func:`~syncopy.singlepanelplot` and :func:`~syncopy.multipanelplot`, as these potentially pipe the whole dataset into the systems memory. It is advised to either perform some averaging beforehand, or cautiously only selecting a few channels/trials for these operations.

.. _meta_functions:
      
Syncopy Meta-Functions
----------------------
All of Syncopy's computing managers (like :func:`~syncopy.freqanalysis`) can be 
either called using positional/keyword arguments following standard Python syntax, 
e.g., 

.. code-block:: python
      
    spec = spy.freqanalysis(data, method="mtmfft", foilim=[1, 150], output="pow", taper="dpss", tapsmofrq=10)

or using a ``cfg`` configuration structure:

.. code-block:: python
      
    cfg = spy.get_defaults(spy.freqanalysis)
    cfg.method = 'mtmfft';
    cfg.foilim = [1, 150];
    cfg.output = 'pow';
    cfg.taper = 'dpss';
    cfg.tapsmofrq = 10;
    spec = spy.freqanalysis(cfg, data)
    
