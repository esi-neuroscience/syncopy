.. _syncopy-data-format:

Reading and Writing Data
=========================

.. contents::
    Contents
    :local:


The Syncopy Data Format (``*.spy``)
-----------------------------------
As each Syncopy data object is simply an annotated multi-dimensional
array every object is stored as

1. a binary file holding data arrays and
2. a human-readable file for metadata.

Syncopy aims to be scalable to process small experimental data-sets to very large
files that don't fit into memory. To meet these requirements, Syncopy performs
on-demand streaming of data from and to disk. A file format that is well-established
for this purpose is `HDF5 <https://www.hdfgroup.org/>`_, which is, therefore,
the default storage backend of Syncopy. In addition, metadata are stored in `JSON
<https://en.wikipedia.org/wiki/JSON>`_, which is both easily human-
and machine-readable.

By default, Syncopy's data files are stored in a folder called ``<basename>.spy``, which
can contain the on-disk representations of multiple objects of different classes
(e.g., spikes and local field potentials that have been recorded simultaneously).
The standard naming pattern of Syncopy's data files is as follows:

::

    <basename>.spy
      └── <basename>_<tag1>.<dataclass>
      └── <basename>_<tag1>.<dataclass>.info
      └── <basename>_<tag2>.<dataclass>
      └── <basename>_<tag2>.<dataclass>.info
           ...

The ``<dataclass>`` specifies the type of data that is stored in the file, i.e.
one of the :ref:`syncopy-data-classes`. The ``<tag>`` part of the filename is
user-defined to distinguish data of the same data class, that should be kept
separate, e.g. data from separate electrode arrays. Data can be loaded using
the :func:`syncopy.load` function.


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



Structure of the Data File (HDF5)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The HDF5 file contains some metadata (`HDF5 attributes
<http://docs.h5py.org/en/stable/high/attr.html>`_) in its header (partially
redundant with the corresponding JSON file), the ``data`` array in binary form
(`HDF5 dataset <http://docs.h5py.org/en/stable/high/dataset.html>`_), and a ``[nTrials x
3+k]``-sized ``trialdefinition`` array containing information about the trials
defined on the data (`trial_start`, `trial_stop`, `trial_triggeroffset`, `trialinfo_1`,
`trialinfo_2`, ..., `trialinfo_k`).

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
or low-level functions (`fread
<https://de.mathworks.com/help/matlab/ref/fread.html>`_). GUIs for inspecting
the data directly include `HDFView
<https://www.hdfgroup.org/downloads/hdfview/>`_ and `HDFCompass
<https://github.com/HDFGroup/hdf-compass>`_.


Structure of the Metadata File (JSON)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The JSON file contains all metadata relevant to the data object. All Syncopy data
objects need to specify (at least) the fields set in ``startInfoDict`` defined
in ``syncopy.io.utils``:

====================    =====  ===========
name                    type   description
====================    =====  ===========
"filename"              str    filename of HDF5 file
"dataclass"             str    one of :ref:`syncopy-data-classes`
"data_dtype"            str    NumPy datatype of data array
"data_shape"            list   shape of data array
"data_offset"           int    offset from begin of file of data array (bytes)
"trl_dtype"             str    NumPy datatype of trialdata array
"trl_shape"             list   shape of trialdata array
"trl_offset"            int    offset from begin of file of trialdata array (bytes)
"file_checksum"         str    checksum value of HDF5 file on disk
"order"                 str    either "C" or "F" (row-major or column-major order of array on disk)
"checksum_algorithm"    str    employed checksum algorithm
"_version"              str    Syncopy package version
"_log"                  str    "prosaic" history of data
"cfg"                   dict   "rigorous" history of data
====================    =====  ===========

.. warning::
    As Syncopy is still in early development, the definition of the required
    JSON fields may change in the future.


Example JSON file:

.. code-block:: javascript

    {
        "filename": "example.c1a8.analog",
        "dataclass": "AnalogData",
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
        "file_checksum": "074602b93ef237b9831fe8ee7ea59b4f8b2ce3614338d65c88081dc9eaddd098964fb68e6061b940de599ab966c3b242e27bd522f80779b1794c3dc3cc518c8e",
        "order": "C",
        "checksum_algorithm": "openssl_sha1",
        "dimord": [
            "time",
            "channel"
        ],
        "_version": "0.1a",
        "_log": "...",
        "cfg": {
            "...": "..."
        }
        "samplerate": 1000.0,
        "channel": [
            "ecogLfp_000",
            "ecogLfp_001",
            "..."

        ],
    }



Reading Other File Formats
--------------------------

Reading and writing other data formats is currently not supported. Getting your
data into Syncopy is, however, relatively straightforward, if you can access
your data in Python, e.g. by using `NEO <http://neuralensemble.org/neo/>`_.

Similar to :func:`syncopy.load` you'll have to write a function that creates an
empty data object (e.g. :class:`syncopy.AnalogData`) and fills the ``data``
property with an index-able array as well as all relevant metadata properties.

In future releases of Syncopy, example reading routines and/or exporting
functions will be provided.
