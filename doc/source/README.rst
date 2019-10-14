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

.. contents:: Contents
    :local:
    :depth: 1

Getting Started
---------------
Our :doc:`Quickstart Guide <quickstart>` covers installation and basic usage. 
More in-depth information relevant to every user of Syncopy can be found in our 
:doc:`User Guide <user/users>`. Want to contribute or just curious how the sausage 
is made? Take a look at our :doc:`Developer Guide <developer/developers>`. Once again
in order of brevity:

* :doc:`Quickstart Guide <quickstart>`
* :doc:`User Guide <user/users>`
* :doc:`Developer Guide <developer/developers>`

Resources by Topic 
^^^^^^^^^^^^^^^^^^
Looking for information regarding a specific analysis method? The table below 
might help. 

.. cssclass:: table-hover

+-------------------+-----------------------+---------------------------+
| **Topic**         | **Resources**         | **Description**           |
+-------------------+-----------------------+---------------------------+
| |TnW|             | |Spy4FT|              | |Spy4FTDesc|              |
|                   +-----------------------+---------------------------+
|                   | |SpyData|             | |SpyDataDesc|             |
|                   +-----------------------+---------------------------+
|                   | |UG|                  | |UGDesc|                  |
+-------------------+-----------------------+---------------------------+
| |RDoc|            | |UsrAPI|              | |UsrAPIDesc|              |
|                   +-----------------------+---------------------------+
|                   | |DevAPI|              | |DevAPIDesc|              |
|                   +-----------------------+---------------------------+
|                   | |DevTools|            | |DevToolsDesc|            |
|                   +-----------------------+---------------------------+
|                   | |Indx|                | |IndxDesc|                |
+-------------------+-----------------------+---------------------------+
| |Spec|            | |SpecTut|             | |SpecTutDesc|             |
|                   +-----------------------+---------------------------+
|                   | |SpecEx|              | |SpecExDesc|              |
|                   +-----------------------+---------------------------+
|                   | |SpecAdv|             | |SpecAdvDesc|             |
+-------------------+-----------------------+---------------------------+
| |Con|             | |ConTut|              | |ConTutDesc|              |
|                   +-----------------------+---------------------------+
|                   | |ConEx|               | |ConExDesc|               |
|                   +-----------------------+---------------------------+
|                   | |ConAdv|              | |ConAdvDesc|              |
+-------------------+-----------------------+---------------------------+

.. |TnW| replace:: *Tutorials & Walkthroughs*
.. |RDoc| replace:: *Reference Documentation*
.. |Spec| replace:: *Spectral Estimation*
.. |Con| replace:: *Connectivity*

.. |Spy4FT| replace:: :doc:`Syncopy for FieldTrip Users <user/fieldtrip>`
.. |Spy4FTDesc| replace:: Quick introduction to Syncopy from a FieldTrip user's perspective
.. |SpyData| replace:: :doc:`Data Handling in Syncopy <user/data_handling>`
.. |SpyDataDesc| replace:: Overview of Syncopy's data management
.. |UG| replace:: :doc:`Syncopy User Guide <user/users>`
.. |UGDesc| replace:: Syncopy's user manual

.. |UsrAPI| replace:: :doc:`User API <user/user_api>`
.. |UsrAPIDesc| replace:: The subset of Syncopy's interface relevant to users
.. |DevAPI| replace:: :doc:`Developer API <developer/developer_api>`
.. |DevAPIDesc| replace:: The parts of Syncopy mostly interesting for developers
.. |Indx| replace:: :ref:`Package Index <genindex>`
.. |IndxDesc| replace:: Index of all functions/classes
.. |DevTools| replace:: :doc:`Syncopy Developer Tools <developer/tools>`
.. |DevToolsDesc| replace:: Tools for contributing new functionality to Syncopy

.. |SpecTut| replace:: Spectral Estimation Tutorial
.. |SpecTutDesc| replace:: An introduction to the available spectral estimation methods in Syncopy
.. |SpecEx| replace:: Spectral Estimation Examples
.. |SpecExDesc| replace:: Example scripts and notebooks illustrating spectral estimation in Syncopy
.. |SpecAdv| replace:: Advanced Topics in Spectral Estimation
.. |SpecAdvDesc| replace:: Technical details and notes for advanced users/developers

.. |ConTut| replace:: Connectivity Tutorial
.. |ConTutDesc| replace:: An introduction to connectivity estimation in Syncopy
.. |ConEx| replace:: Connectivity Examples
.. |ConExDesc| replace:: Example scripts and notebooks illustrating the use of connectivity metrics in Syncopy
.. |ConAdv| replace:: Advanced Topics in Connectivity 
.. |ConAdvDesc| replace:: Technical details and notes for advanced users/developers

Still no luck finding what you're looking for? Try using the :ref:`search <search>` function. 

Sitemap
-------
.. toctree::
   :maxdepth: 2

   quickstart
   user/users.rst    
   developer/developers.rst   

Indices and tables
^^^^^^^^^^^^^^^^^^
* :ref:`genindex`
* :ref:`search`

Contact
-------
To report bugs or ask questions please use our GitHub issue tracker. For
general inquiries please contact syncopy (at) esi-frankfurt.de. 
