API for Users
=============

This page gives an overview over all public functions and classes of Syncopy.

.. contents:: Sections
   :local:

High-level functions
--------------------
These *meta-functions* bundle many related analysis methods into one high-level function.

.. autosummary::

   syncopy.preprocessing   
   syncopy.resampledata
   syncopy.freqanalysis   
   syncopy.connectivityanalysis
   syncopy.timelockanalysis

Descriptive Statistics
----------------------
.. autosummary::

   syncopy.mean
   syncopy.var
   syncopy.std
   syncopy.median
   syncopy.itc
   syncopy.spike_psth

Utility
--------

.. autosummary::

   syncopy.definetrial
   syncopy.selectdata
   syncopy.redefinetrial   
   syncopy.show
   syncopy.cleanup
   
I/O
--------------------
Functions to import and export data in Syncopy

.. autosummary::

   syncopy.load
   syncopy.save
   syncopy.load_ft_raw
   syncopy.load_tdt
   syncopy.load_nwb
   syncopy.copy


Plotting
-----------

These convenience function are intended to be used for a quick visual inspection of data and results.

.. autosummary::

   syncopy.singlepanelplot
   syncopy.multipanelplot

Data Types
--------------------

Syncopy data types are Python classes, which offer convenient ways for data access and manipulation.

.. autosummary::

   syncopy.AnalogData
   syncopy.SpectralData
   syncopy.CrossSpectralData
   syncopy.SpikeData
   syncopy.EventData
