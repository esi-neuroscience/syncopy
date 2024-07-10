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

Citing Syncopy
-----------------

A pre-print paper on Syncopy is available `here on arxiv, with DOI 10.1101/2024.04.15.589590 <https://doi.org/10.1101/2024.04.15.589590>`_. Please cite this pre-print if you use Syncopy. In APA style, the citation is: Mönke, G., Schäfer, T., Parto-Dezfouli, M., Kajal, D. S., Fürtinger, S., Schmiedt, J. T., & Fries, P. (2024). *Systems Neuroscience Computing in Python (SyNCoPy): A Python Package for Large-scale Analysis of Electrophysiological Data.* bioRxiv, 2024-04.


Tutorials and in depth Guides
-----------------------------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   Preprocessing <tutorials/preprocessing>
   Resampling <tutorials/resampling>
   Spectral Analysis <tutorials/freqanalysis>
   Connectivity Analysis <tutorials/connectivity>

.. toctree::
   :maxdepth: 1
   :caption: Guides

   Basic Concepts <user/concepts>
   Syncopy for FieldTrip Users <user/fieldtrip>
   Handling Data <user/data>
   Parallel Processing <user/parallel>

API
---

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   User API <user/user_api>
   Complete API <user/complete_api>

Contact
-------
To report bugs or ask questions please use our `GitHub issue tracker <https://github.com/esi-neuroscience/syncopy/issues>`_.
For general inquiries please contact syncopy (at) esi-frankfurt.de.
