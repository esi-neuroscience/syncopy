.. Syncopy documentation master file

.. title:: Syncopy Documentation

.. image:: _static/syncopy_logo.png
    :alt: Syncopy logo
    :height: 200px
    :align: center


Welcome to the Documentation of SyNCoPy!
========================================

SyNCoPy (**Sy**\stems **N**\euroscience **Co**\mputing in **Py**\thon, spelled Syncopy in the following)
is a Python toolkit for user-friendly, large-scale electrophysiology data analysis.
We strive to achieve the following goals:

1. Syncopy provides a full *open source* Python environment for reproducible
   electrophysiology data analysis.
2. Syncopy is *scalable* to accommodate *very large* datasets. It automatically
   makes use of available computing resources and is developed with built-in
   parallelism in mind.
3. Syncopy is *compatible* with the MATLAB toolbox `FieldTrip <http://www.fieldtriptoolbox.org/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Install Syncopy </setup>
   Quickstart Guide <quickstart/quickstart>

Want to contribute or just curious how the sausage
is made? Take a look at our :doc:`Developer Guide <developer/developers>`.


Tutorials and in depth Guides
-----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   Resampling <tutorials/resampling>
   Spectral Analysis <tutorials/freqanalysis>


.. toctree::
   :maxdepth: 2
   :caption: Guides	      
   
   Basic Concepts <user/concepts>
   Syncopy for FieldTrip Users <user/fieldtrip>
   Handling Data <user/data>
   Parallel Processing <user/parallel>
   Selections <user/selectdata>

API
---

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   User API <user/user_api>


Contact
-------
To report bugs or ask questions please use our `GitHub issue tracker <https://github.com/esi-neuroscience/syncopy/issues>`_.
For general inquiries please contact syncopy (at) esi-frankfurt.de.
