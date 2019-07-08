Reading from and writing data to disk
=====================================

The Syncopy data format (``*.spy``)
-----------------------------------

As each Syncopy data object is not more than an anotated multi-dimensional
array each object is usually stored in 

1. a binary file for the data arrays and
2. a human-readable file for metadata.

Syncopy aims at being scalable for very large files that don't fit in memory. To
cope with those kind of files, it is usually necessary to stream data from and
to disk only on demand. A file format that is well-established for this kind of
purpose is `HDF5 <https://www.hdfgroup.org/>`_, which is therefore the default
storage backend of Syncopy. In addition, the metadata are store in `JSON
<https://en.wikipedia.org/wiki/JSON>`_, which is both easily human-readable 
and machine-readable.

These files are usually stored in a folder called ``<basename>.spy``, which can
contain multiple data of different classes that have been recorded
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
one of :ref:`Syncopy data classes`. The ``<tag>`` part of the filename is
user-defined to distinguish data of the same data class, that should be kept
separate, e.g. data from separate electrode arrays.

The HDF5 file contains some metadata (HDF attributes) in its header (redundant
with JSON file), the data array in binary form (HDF dataset), and a [nTrials x
3+k]-sized `trialdata` array containing information about the trials defined on
the data (trial_start, trial_stop, trial_triggeroffset, trialinfo_1,
trialinfo_2, ..., trialinfo_k).

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

Similar to :func:`syncopy.load_spy` you'll have to write a function that creates
an empty data object (e.g. `syncopy.AnalogData`) and fill the ``data`` property
with an index-able array as well as the annotation fields.