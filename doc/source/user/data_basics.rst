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

Importing Data into Syncopy
---------------------------

Currently, Syncopy supports importing data from `FieldTrip raw data <https://www.fieldtriptoolbox.org/development/datastructure/>`_ format and from `NWB <https://www.nwb.org/>`_

.. autosummary::

    syncopy.io.load_ft_raw
    syncopy.io.load_nwb
   
