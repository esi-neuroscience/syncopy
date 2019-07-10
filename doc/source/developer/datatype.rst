
Syncopy data classes 
====================
 
The data structure in Syncopy is based around the idea that all
electrophysiology data can be represented as multidimensional arrays. For
example, a multi-channel local field potential can be stored as a
two-dimensional `float` array with the dimensions being time (sample) and
channel. Hence, 

.. note:: Each Syncopy data object is simply an anotated multi-dimensional array.

This array is always stored in the :attr:`data` property and can be
indexed using `NumPy indexing
<https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays>`_. 

Different types electrophysiology data often share different properties (e.g.
having channel/electrode labels, having a time axis, etc.). An efficient way of
organizing these different data types are `classes
<https://en.wikipedia.org/wiki/Class_(computer_programming)>`_ organized in a
hierarchy, with shared properties inherited from the top levels to bottom
classes (see also `Wikipedia
<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)>`_).

.. inheritance-diagram:: syncopy.AnalogData syncopy.SpectralData syncopy.SpikeData syncopy.EventData
   :top-classes: BaseData
   :parts: 1

The bottom classes in the class tree are for active use in analyses.

The usable Syncopy data classes
-------------------------------

The classes that

.. autosummary::
    :toctree: _stubs
    :template: syncopy_class.rst

    syncopy.AnalogData
    syncopy.SpectralData
    syncopy.SpikeData
    syncopy.EventData




