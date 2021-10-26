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

# this still gives type(tsel) = slice :)
sdict1 = {"trials": [0], 'channels' : ['channel1'], 'toi': np.arange(-1, 1, 0.001)}
# this gives type(tsel) = list
# sdict1 = {"trials": [0], 'channels' : ['channel1'], 'toi': np.array([0, 0.3, 1])}
sdict2 = {"trials": [0], 'channels' : ['channel1'], 'toilim' : [-1, 0]}

print('sdict1')
connectivityanalysis(data=tdat, select=sdict1, pad_to_length=4200)
#connectivityanalysis(data=tdat, select=sdict1, pad_to_length='nextpow2')
#connectivityanalysis(data=tdat, select=sdict1, pad_to_length=None)
print('sdict2')
#connectivityanalysis(data=tdat, select=sdict2)
print('no selection')
#connectivityanalysis(data=tdat, foi = np.arange(20, 80))
#connectivityanalysis(data=tdat, foilim = [20, 80])


