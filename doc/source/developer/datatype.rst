
Syncopy data classes (:mod:`syncopy.datatype`)
 

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

Some info highlighting the boundless wisdom underlying the class design...

.. autosummary::
    :toctree: _stubs
    :template: syncopy_class.rst

    syncopy.AnalogData
    syncopy.SpectralData
    syncopy.SpikeData
    syncopy.EventData



