# -*- coding: utf-8 -*-
#
# Supported Syncopy classes and file extensions
#

def _data_classname_to_extension(name):
    return "." + name.split('Data')[0].lower()

# data file extensions are first word of data class name in lower-case
supportedClasses = ('AnalogData', 'SpectralData', 'CrossSpectralData', # ContinousData
                    'SpikeData', 'EventData',  # DiscreteData
                    'TimelockData', ) # StatisticalData

supportedDataExtensions = tuple([_data_classname_to_extension(cls)
                                 for cls in supportedClasses])

# Define SynCoPy's general file-/directory-naming conventions
FILE_EXT = {"dir" : ".spy",
            "info" : ".info",
            "data" : supportedDataExtensions}
