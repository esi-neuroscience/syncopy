.. _syncopy-data-classes:

Syncopy Data Classes
====================

The data structure in Syncopy is based around the idea that all
electrophysiology data can be represented as multidimensional arrays. For
example, a multi-channel local field potential can be stored as a
two-dimensional `float` array with the dimensions being time (sample) and
channel. This array is always stored in the :attr:`data` property and can be
indexed using `NumPy indexing
<https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays>`_. 

.. note:: Each Syncopy data object is simply an annotated multi-dimensional array.

Different types of electrophysiology data often share common properties (e.g.
having channel/electrode labels, having a time axis, etc.). An efficient way of
organizing these different data types are `classes
<https://en.wikipedia.org/wiki/Class_(computer_programming)>`_ organized in a
hierarchy, with shared properties inherited from the top level to the bottom
classes (see also `Wikipedia
<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)>`_).

.. inheritance-diagram:: syncopy.AnalogData syncopy.SpectralData syncopy.SpikeData syncopy.EventData
   :top-classes: BaseData
   :parts: 1

The bottom classes in the class tree are for active use in analyses.

Usable Syncopy Data Classes
----------------------------
The following classes can be instanced at the package-level (``spy.AnalogData(...)`` etc.)

.. autosummary::

    syncopy.AnalogData
    syncopy.SpectralData
    syncopy.SpikeData
    syncopy.EventData




