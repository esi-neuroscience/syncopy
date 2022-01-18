Handling Data in Syncopy
========================
Syncopy utilizes a simple data format based on `HDF5
<https://portal.hdfgroup.org/display/HDF5/HDF5>`_ and `JSON
<https://en.wikipedia.org/wiki/JSON>`_ (see :doc:`../developer/io` for details).
These formats were chosen for their *ubiquity* as they can be handled well in
virtually all popular programming languages, and for allowing *streaming,
parallel access* enabling computing on parallel architectures.

Currently, data in other formats (e.g. from  a recording system) have to be
converted before use with Syncopy. For this purpose, later versions of Syncopy will include
importing and exporting engines, for example based on `Neo
<https://neo.readthedocs.io/en/latest/>`_ or `NWB <https://www.nwb.org/>`_.


Loading and Saving Syncopy (``*.spy``) Data
-------------------------------------------
Reading and writing data with Syncopy

.. autosummary::

    syncopy.load
    syncopy.save

Functions for Inspecting/Editing Syncopy Data Objects
-----------------------------------------------------
Defining trials, data selection and padding.

.. autosummary::

    syncopy.definetrial
    syncopy.show
    syncopy.selectdata
    syncopy.padding

Advanced Topics
---------------
More information about Syncopy's data class structure and file format.

.. toctree::

    ../developer/datatype
    ../developer/io
