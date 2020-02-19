# -*- coding: utf-8 -*-
# 
# Supported Syncopy classes and file extensions
# 
# Created: 2020-01-27 13:23:10
# Last modified by: Joscha Schmiedt [joscha.schmiedt@esi-frankfurt.de]
# Last modification time: <2020-01-27 13:27:26>

def _data_classname_to_extension(name):
    return "." + name.split('Data')[0].lower()

# data file extensions are first word of data class name in lower-case
supportedClasses = ('AnalogData', 'SpectralData', # ContinousData
                    'SpikeData', 'EventData',  # DiscreteData
                    'TimelockData', ) # StatisticalData

supportedDataExtensions = tuple([_data_classname_to_extension(cls)
                                 for cls in supportedClasses])

# Define SynCoPy's general file-/directory-naming conventions
FILE_EXT = {"dir" : ".spy",
            "info" : ".info",
            "data" : supportedDataExtensions}
