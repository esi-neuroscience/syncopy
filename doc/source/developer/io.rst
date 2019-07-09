Reading from and writing data to disk
=====================================

.. contents::
    Contents
    :local:


The Syncopy data format (``*.spy``)
-----------------------------------

As each Syncopy data object is not more than an anotated multi-dimensional
array each object is usually stored in 

1. a binary file for the data arrays and
2. a human-readable file for metadata.

Syncopy aims at being scalable for very large files that don't fit in memory. To
cope with those kind of files, it is usually necessary to stream data from and
to disk only on demand. A file format that is well-established for this 
purpose is `HDF5 <https://www.hdfgroup.org/>`_, which is therefore the default
storage backend of Syncopy. In addition, the metadata are stored in `JSON
<https://en.wikipedia.org/wiki/JSON>`_, which is both easily human-readable 
and machine-readable.

The data files are usually stored in a folder called ``<basename>.spy``, which
can contain multiple data of different data classes that have been recorded
simulatenously, e.g. spikes and local field potentials. The standard naming
pattern of the data files is the following

:: 

    <basename>.spy
      └── <basename>_<tag1>.<dataclass>
      └── <basename>_<tag1>.<dataclass>.info
      └── <basename>_<tag2>.<dataclass>
      └── <basename>_<tag2>.<dataclass>.info
           ...

The ``<dataclass>`` specifies the type of data that is stored in the file, i.e.
one of the :ref:`Syncopy data classes`. The ``<tag>`` part of the filename is
user-defined to distinguish data of the same data class, that should be kept
separate, e.g. data from separate electrode arrays. The data can be loaded into
Python using the :func:`syncopy.load` function.


**Example folder**

:: 

    monkeyB_20190709_rfmapping_1.spy
      └── monkeyB_20190709_rfmapping_1_amua-stimon.analog
      └── monkeyB_20190709_rfmapping_1_amua-stimon.analog.info
      └── monkeyB_20190709_rfmapping_1_amua-cueon.analog
      └── monkeyB_20190709_rfmapping_1_amua-cueon.analog.info
      └── monkeyB_20190709_rfmapping_1_eyes.analog
      └── monkeyB_20190709_rfmapping_1_eyes.analog.info
      └── monkeyB_20190709_rfmapping_1_lfp.analog
      └── monkeyB_20190709_rfmapping_1_lfp.analog.info
      └── monkeyB_20190709_rfmapping_1_vprobe.spike
      └── monkeyB_20190709_rfmapping_1_vprobe.spike.info
      └── monkeyB_20190709_rfmapping_1_marker.event
      └── monkeyB_20190709_rfmapping_1_marker.event.info
      └── monkeyB_20190709_rfmapping_1_stimon-wavelet.spectral
      └── monkeyB_20190709_rfmapping_1_stimon-wavelet.spectral.info



Structure of the data file (HDF5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The HDF5 file contains some metadata (`HDF5 attributes
<http://docs.h5py.org/en/stable/high/attr.html>`_) in its header (partially
redundant with JSON file), the ``data`` array in binary form (`HDF5 dataset
<http://docs.h5py.org/en/stable/high/dataset.html>`_), and a ``[nTrials x
3+k]``-sized ``trialdefinition`` array containing information about the trials
defined on the data (trial_start, trial_stop, trial_triggeroffset, trialinfo_1,
trialinfo_2, ..., trialinfo_k).

::

    bof | ---- header ---- | ---------- data ---------- | -- trialdefinition --| eof


The shape, data type, and offsets of the ``data`` and ``trialdefinition`` arrays
are stored in both the header and the metadata JSON file. The format is
therefore simple enough to be read either with an HDF5 library, e.g. `h5py
<https://www.h5py.org/>`_, or directly as binary arrays, e.g. with a
:class:`numpy.memmap` or :func:`numpy.fromfile` (numpy >= 1.17). Also with other
programming languages such as C/C++ or MATLAB, it is feasible to read and write
such data files using the HDF5 library (`C/C++
<https://portal.hdfgroup.org/display/HDF5/Examples+from+Learning+the+Basics>`_ ,
`MATLAB
<https://de.mathworks.com/help/matlab/high-level-functions.html?s_tid=CRUX_lftnav>`_)
or more low-level functions (`fread
<https://de.mathworks.com/help/matlab/ref/fread.html>`_). GUIs for inspecting
the data directly include `HDFView
<https://www.hdfgroup.org/downloads/hdfview/>`_ and the `HDFCompass
<https://github.com/HDFGroup/hdf-compass>`_.


Structure of the metadata file (JSON)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The JSON file contains all metadata about the data object. The required fields
in the JSON file are:

=============  =====  ===========
name           type   description
=============  =====  ===========
"dimord"       list   labels of array dimensions
"version"      str    version of spy format
"log"          str    "prosaic" history of data
"cfg"          dict   "rigorous" history of data
"data"         str    filename of HDF5 file
"data_dtype"   str    NumPy datatype of data array
"data_shape"   list   shape of data array in indices
"data_offset"  int    offset from begin of file of data array (bytes)
"trl_dtype"    str    NumPy datatype of trialdata array
"trl_shape"    list   shape of trialdata array in indices
"trl_offset"   int    offset from begin of file of trialdata array (bytes)
=============  =====  ===========

.. warning:: 
    As Syncopy is still in early development, the definition of the required
    JSON fields may change in the future.


Example JSON file:

.. code-block:: javascript

    {
        "type": "AnalogData",
        "dimord": [
            "time",
            "channel"
        ],
        "version": "0.1a",
        "data": "example.c1a8.dat",
        "data_dtype": "float32",
        "data_shape": [
            406680,
            560
        ],
        "data_offset": 2048,
        "trl_dtype": "int64",
        "trl_shape": [
            219,
            3
        ],
        "trl_offset": 910965248,
        "samplerate": 1000.0,
        "data_checksum": "074602b93ef237b9831fe8ee7ea59b4f8b2ce3614338d65c88081dc9eaddd098964fb68e6061b940de599ab966c3b242e27bd522f80779b1794c3dc3cc518c8e",
        "log": "...",
        "hdr": [
            {
                "version": 1,
                "length": 128,
                "dtype": "float32",
                "M": 406680,
                "N": 256,
                "tSample": 1000000,
                "file": "MT_RFmapping_session-168a1_xWav.lfp"            
            }
        ],
        "channel": [
            "ecogLfp_000",
            "ecogLfp_001",
            "..."
            
        ],
        "cfg": {
            "...": "..."
        }
    }

    

Reading other data formats
--------------------------

Reading and writing other data formats are currently not supported. Getting your
data into Syncopy is however relatively straightforward, if you're able to read
your data into Python, e.g. by using `NEO <http://neuralensemble.org/neo/>`_.

Similar to :func:`syncopy.load` you'll have to write a function that creates an
empty data object (e.g. :class:`syncopy.AnalogData`) and fills the ``data``
property with an index-able array as well as the metadata properties.

In future releases of Syncopy, example reading routines and/or exporting
functions may be provided.