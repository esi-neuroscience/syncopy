import numpy as np
import scipy.signal as sci
from syncopy.connectivity.single_trial_compRoutines import cross_spectra_cF
from syncopy.connectivity.single_trial_compRoutines import cross_covariance_cF
from syncopy.connectivity import connectivityanalysis
import matplotlib.pyplot as ppl

from syncopy.shared.parsers import data_parser, scalar_parser, array_parser    
from syncopy.shared.tools import get_defaults 
from syncopy.datatype import SpectralData, padding

from syncopy.tests.misc import generate_artificial_data
tdat = generate_artificial_data(inmemory=True)

sdict = {"trials": [0], 'channels' : ['channel1']}
connectivityanalysis(data=tdat, select=sdict, pad_to_length=4200)
connectivityanalysis(data=tdat, select=sdict, pad_to_length=None)

