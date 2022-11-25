***********
Selections
***********

In every practical data analysis project working on subsets of a dataset is a very common scenario. Syncopy offers the powerful :func:`syncopy.selectdata` function to achieve exactly that. There are two distinct ways to apply a selection, either *in place* or the chosen subset gets copied and returned as a new Syncopy data object. Additionally, every Syncopy *meta-function* (see :ref:`meta_functions`) supports the ``select`` keyword, which applies an in place selection on the fly when processing data.

.. contents:: Topics covered
   :local:

.. _workflow:

Creating Selections
===================

A new selection can be created by calling :func:`syncopy.selectdata`::

  trial10 = AData.selectdata(trials=10, channel=["channel11", "channel17"])
  # alternatively
  trial10 = spy.selectdata(AData, trials=10, channel=["channel02", "channel06"])

In this case if ``AData`` is an :class:`syncopy.AnalogData` object, the resulting
``trial10`` data object will also be of that same data type. Inspecting the original
dataset by simpling typing its name into the Python interpreter::

  AData

we see that we have 100 trials and 10 channels:
  
.. code-block:: bash


   Syncopy AnalogData object with fields

            cfg : dictionary with keys ''
        channel : [10] element <class 'numpy.ndarray'>
      container : None
           data : 100 trials of length 500.0 defined on [50000 x 10] float32 Dataset of size 1.91 MB
       ...

If we now inspect our selection results::

  trial10

we see that we are left with 1 trial and 2 channels:

.. code-block:: bash


   Syncopy AnalogData object with fields

            cfg : dictionary with keys ''
        channel : [2] element <class 'numpy.ndarray'>
      container : None
           data : 1 trials of length 500.0 defined on [500 x 2] float32 Dataset of size 0.00 MB
       ...


Moreover by inspecting the ``.log`` (see also :ref:`logging`) we can see the selection settings used to create this dataset::

  trials10.log

.. code-block:: bash
  
	write_log: computed _selectdata with settings
	inplace = False
	clear = False
	trials = 10
	channel = ['channel02', 'channel06']
	channel_i = None
	channel_j = None
	toi = None
	toilim = None
	foi = None
	foilim = None
	taper = None
	unit = None
	eventid = None
	parallel = False

This log is persistent, meaning that when saving and later loading this reduced dataset the settings used for this selection can still be recovered.

Selection Parameters
====================

There are various selection parameters
available, which each can accept a variety of Python datatypes like ``int``, ``str`` or ``list``:

+-------------------+-----------------------+-----------------------+-----------------------+-------------------------------------+
| **Parameter**     |   **Description**     |   **Accepted Values** |   **Examples**        |   **Availability**                  |
+===================+=======================+=======================+=======================+=====================================+
|    trials         |     |trialsDesc|      |     |trialsVals|      |     |trialsEx1|       |                                     |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |trialsEx2|       |     *all data types*                |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |trialsEx3|       |                                     |
+-------------------+-----------------------+-----------------------+-----------------------+-------------------------------------+
|    channel        |     |channelDesc|     |     |channelVals|     |     |channelEx1|      | :class:`syncopy.AnalogData`         |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |channelEx2|      | :class:`syncopy.SpectralData`       |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |channelEx3|      | :class:`syncopy.CrossSpectralData`  |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |channelEx4|      | :class:`syncopy.SpikeData`          |
+-------------------+-----------------------+-----------------------+-----------------------+-------------------------------------+
|    toi            |     |toiDesc|         |     |toiVals|         |     |toiEx1|          | :class:`syncopy.AnalogData`         |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |toiEx2|          | :class:`syncopy.SpectralData`       |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |toiEx3|          | :class:`syncopy.CrossSpectralData`  |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |                       | :class:`syncopy.SpikeData`          |
+-------------------+-----------------------+-----------------------+-----------------------+-------------------------------------+	
|    toilim         |     |toilimDesc|      |     |toilimVals|      |     |toilimEx1|       |   *as above*                        |
+-------------------+-----------------------+-----------------------+-----------------------+-------------------------------------+
|    foi            |     |foiDesc|         |     |foiVals|         |     |foiEx1|          |                                     |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |foiEx2|          | :class:`syncopy.SpectralData`       |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |foiEx3|          | :class:`syncopy.CrossSpectralData`  |
+-------------------+-----------------------+-----------------------+-----------------------+-------------------------------------+
|    foilim         |     |foilimDesc|      |     |foilimVals|      |     |foilimEx1|       |   *as above*                        |
+-------------------+-----------------------+-----------------------+-----------------------+-------------------------------------+
|    unit           |     |unitDesc|        |     |unitVals|        |     |unitEx1|         | :class:`syncopy.SpikeData`          |
|                   |                       |                       |                       |                                     |
|                   |                       |                       |     |unitEx2|         |                                     |
|                   |                       |                       |                       |                                     | 
|                   |                       |                       |     |unitEx3|         |                                     |
+-------------------+-----------------------+-----------------------+-----------------------+-------------------------------------+
|    eventid        |     |eventidDesc|     |     |eventidVals|     |     |eventidEx1|      |                                     |
|                   |                       |                       |                       | :class:`syncopy.EventData`          |
|                   |                       |                       |     |eventidEx2|      |                                     |
+-------------------+-----------------------+-----------------------+-----------------------+-------------------------------------+

.. |trialsVals| replace:: *int, array, list*
.. |trialsDesc| replace:: *trial selection*
.. |trialsEx1| replace::  ``selectdata(trials=7)`` 
.. |trialsEx2| replace::  ``selectdata(trials=[2, 9, 21])`` 
.. |trialsEx3| replace::  ``selectdata(trials=np.arange(2, 10))`` 

			  
.. |channelDesc| replace:: *channel selection*
.. |channelVals| replace:: *int, str, array, list, slice*
.. |channelEx1| replace:: ``selectdata(channel=7)`` 
.. |channelEx2| replace:: ``selectdata(channel=[11, 16])``			  
.. |channelEx3| replace:: ``selectdata(channel=np.arange(2, 10))``
.. |channelEx4| replace:: ``selectdata(channel=["V1-11, "V2-12"])``

.. |toiDesc| replace:: *time points of interest in seconds*
.. |toiVals| replace:: *float, array, list*
.. |toiEx1| replace:: ``selectdata(toi=0.2)`` 
.. |toiEx2| replace:: ``selectdata(toi=[0, 0.1, 0.2, 0.5])`` 
.. |toiEx3| replace:: ``selectdata(toi=np.linspace(0, 1, 100))`` 

.. |toilimDesc| replace:: *time interval of interest in seconds*
.. |toilimVals| replace:: *float [tmin, tmax]*
.. |toilimEx1| replace:: ``selectdata(toilim=[-.1, 1])`` 

.. |foiDesc| replace:: *frequencies of interest in Hz*
.. |foiVals| replace:: *float, array, list*
.. |foiEx1| replace:: ``selectdata(foi=20)`` 
.. |foiEx2| replace:: ``selectdata(foi=[5, 10, 15])`` 
.. |foiEx3| replace:: ``selectdata(foi=np.linspace(1, 60, 100))`` 

.. |foilimDesc| replace:: *frequency interval of interest in Hz*
.. |foilimVals| replace:: *float [fmin, fmax]*
.. |foilimEx1| replace:: ``selectdata(foilim=[10, 60])`` 

.. |unitDesc| replace:: *unit selection*
.. |unitVals| replace:: *int, str, list, slice*
.. |unitEx1| replace:: ``selectdata(unit=7)``
.. |unitEx2| replace:: ``selectdata(unit=[11, 16, 32])``			  
.. |unitEx3| replace:: ``selectdata(unit=["unit17", "unit3"])``

.. |eventidDesc| replace:: *eventid selection*
.. |eventidVals| replace:: *int, list, slice*
.. |eventidEx1| replace:: ``selectdata(eventid=2)``
.. |eventidEx2| replace:: ``selectdata(eventid=[2, 0, 1])``

Parameters per Data Type
------------------------
Which of the selection parameters is available for a specfific dataset depends on the data type we are working with:


:class:`syncopy.AnalogData`:

* **trials**
* **channel**
* **toi / toilim**

:class:`syncopy.SpectralData`:

* **trials**
* **channel**
* **foi / foilim**
* **toi / toilim** (if time-frequency spectrum)

:class:`syncopy.CrossSpectralData`:

* **trials**
* **channel_i, channel_j** (use like **channel**)
* **foi / foilim**
* **toi / toilim** (if time-frequency spectrum)

:class:`syncopy.SpikeData`:

* **trials**
* **channel**
* **unit**
* **toi / toilim**

:class:`syncopy.EventData`:

* **trials**
* **eventid**
* **toi / toilim**

.. note::
   Have a look at :doc:`Data Basics <data_basics>` for further details about Syncopy's data classes and interfaces

