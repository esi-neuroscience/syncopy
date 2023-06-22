.. _data_basics:

Syncopy Data Basics
===================

Syncopy utilizes a simple data format based on `HDF5
<https://portal.hdfgroup.org/display/HDF5/HDF5>`_ and `JSON
<https://en.wikipedia.org/wiki/JSON>`_ (see :doc:`../developer/io` for details).
These formats were chosen for their *ubiquity* as they can be handled well in
virtually all popular programming languages, and for allowing *streaming,
parallel access* enabling computing on parallel architectures.

.. contents:: Topics covered
   :local:


Loading and Saving Syncopy (``*.spy``) Data
-------------------------------------------
Reading and writing data with Syncopy

.. autosummary::

    syncopy.load
    syncopy.save

Functions for Inspecting/Editing Syncopy Data Objects
-----------------------------------------------------
Defining trials, data selection and NumPy :class:`~numpy.ndarray` interface

.. autosummary::

    syncopy.definetrial
    syncopy.selectdata
    syncopy.show

Plotting Functions
------------------

.. autosummary::

   syncopy.singlepanelplot
   syncopy.multipanelplot

.. hint::
   The :ref:`selections` section details how :func:`~syncopy.singlepanelplot` and :func:`~syncopy.show` all work based on the same :func:`~syncopy.selectdata` API.

   
Importing Data into Syncopy
---------------------------

Importing Data from different file formats into Syncopy
-------------------------------------------------------

Currently, Syncopy supports importing data from `FieldTrip raw data <https://www.fieldtriptoolbox.org/development/datastructure/>`_ format, from `NWB <https://www.nwb.org/>`_ and `TDT <https://www.tdt.com/>`_:

.. autosummary::

    syncopy.io.load_ft_raw
    syncopy.io.load_nwb
    syncopy.io.load_tdt


Importing Data from NumPy
-------------------------

If you have an electrical time series as a :class:`~numpy.ndarray` and want to import it into Syncopy, you can initialize an :class:`~syncopy.AnalogData` object directly:

.. autosummary::

    syncopy.AnalogData


Creating Synthetic Example Data
-------------------------------

Syncopy contains the `synthdata` module, which can be used to create synthetic data for testing and demonstration purposes.


.. autosummary::

    syncopy.synthdata



Exporting Data from Syncopy
---------------------------

Syncopy supports export of data to `NWB <https://www.nwb.org/>`_ format for objects of type :class:`~syncopy.AnalogData`, :class:`~syncopy.TimeLockData` and :class:`~syncopy.SpikeData`.


.. autosummary::

    syncopy.AnalogData.save_nwb
    syncopy.TimeLockData.save_nwb
    syncopy.SpikeData.save_nwb

Here is a little example::

  import syncopy as spy

  raw_data = spy.synthdata.red_noise(alpha=0.9)
  
  # some processing, bandpass filter and (here meaningless) phase extraction
  processed_data = spy.preprocessing(raw_data, filter_type='bp', freq=[35, 40], hilbert='angle')

  # save raw data to NWB
  nwb_path = 'test.nwb'
  nwbfile = raw_data.save_nwb(nwb_path)
  
  # save processed data into same NWB file
  processed_data.save_nwb(nwb_path, nwbfile=nwbfile, is_raw=False)
  
Note that NWB is a very general container format, and thus loading an NWB container created in one software package into the internal data structures used by another software package requires some interpretation of the fields, which users many need to do manually. One can inspect NWB files online using tools like the `NWB Explorer <https://nwbexplorer.opensourcebrain.org>`_.

