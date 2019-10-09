Handling data in Syncopy
========================

Syncopy utilizes a simple data format based on `HDF5
<https://portal.hdfgroup.org/display/HDF5/HDF5>`_ and `JSON
<https://en.wikipedia.org/wiki/JSON>`_ (see :doc:`../developer/io` for details).
These formats were chosen for their *ubiquity* as they can be handled well in
virtually all popular programming languages, and for allowing *streaming,
parallel access* enabling computing on parallel architecures.

Currently, data in other formats (e.g. from  a recording system) have to be
converted first. For this purpose, later versions of Syncopy will include
importing and exporting engines, for example based on `Neo
<https://neo.readthedocs.io/en/latest/>`_ or `NWB <https://www.nwb.org/>`_.


Reading and saving Syncopy (``*.spy``) data
-------------------------------------------
.. autosummary::

    syncopy.load
    syncopy.save



Functions for editing data in memory
------------------------------------
These functions are useful for editing and slicing data:

.. autosummary::

    syncopy.definetrial
    syncopy.padding
