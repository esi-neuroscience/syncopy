# -*- coding: utf-8 -*-
# 
# Ensure tooling functions shared across the package work as intended
# 

# Builtin/3rd party package imports
import numpy as np
import pytest

# Local imports
from syncopy.shared.tools import best_match
from syncopy.shared.errors import SPYValueError


class TestBestMatch():

    # Source-arrays with integer elements
    intSource = np.arange(10)
    randIntSource = np.random.choice(intSource, size=intSource.size, replace=False)

    # Source-arrays with floating point elements
    floatSource = np.array([1.5, 1.5, 2.2, 6.2, 8.8])
    randFloatSource = np.random.choice(floatSource, size=floatSource.size, replace=False)

    # Selections defined by ordered/unordered int/float arrays
    intSelection = intSource[:4]
    randIntSelection = np.random.choice(intSelection, size=intSelection.size, replace=False)
    floatSelection = np.array([1.9, 9., 1., -0.4, 1.2, 0.2, 9.3])
    sortFloatSelection = np.sort(floatSelection)
    
    def test_intsource(self):
        
        for source in [self.intSource, self.randIntSource]:
            for selection in [self.intSelection, self.randIntSelection, 
                              self.floatSelection, self.sortFloatSelection]:
                expectedVal = np.round(selection)
                expectedIdx = np.array([np.where(source == elem)[0][0] for elem in expectedVal])
                val, idx = best_match(source, selection)
                assert np.array_equal(val, expectedVal)
                assert np.array_equal(idx, expectedIdx)
                
                val, idx = best_match(source, selection, squash_duplicates=True)
                _, sidx = np.unique(expectedVal, return_index=True)
                sidx.sort()
                assert np.array_equal(val, expectedVal[sidx])
                assert np.array_equal(idx, expectedIdx[sidx])
                
                with pytest.raises(SPYValueError):
                    best_match(source, selection, tol=1e-6)
                
                val, idx = best_match(source, [selection.min(), selection.max()], span=True)
                expectedVal = np.array([elem for elem in source 
                                        if selection.min() <= elem <= selection.max()])
                expectedIdx = np.array([np.where(source == elem)[0][0] for elem in expectedVal])
        

    def test_floatsource(self):
        for source in [self.floatSource, self.randFloatSource]:
            for selection in [self.intSelection, self.randIntSelection, 
                              self.floatSelection, self.sortFloatSelection]:
                expectedVal = np.array([source[np.argmin(np.abs(source - elem))] for elem in selection])
                expectedIdx = np.array([np.where(source == elem)[0][0] for elem in expectedVal])
                val, idx = best_match(source, selection)
                assert np.array_equal(val, expectedVal)
                assert np.array_equal(idx, expectedIdx)

                val, idx = best_match(source, selection, squash_duplicates=True)
                _, sidx = np.unique(expectedVal, return_index=True)
                sidx.sort()
                assert np.array_equal(val, expectedVal[sidx])
                assert np.array_equal(idx, expectedIdx[sidx])

                with pytest.raises(SPYValueError):
                    best_match(source, selection, tol=1e-6)

                val, idx = best_match(source, [selection.min(), selection.max()], span=True)
                expectedVal = np.array([elem for elem in source 
                                        if selection.min() <= elem <= selection.max()])
                expectedIdx = np.array([np.where(source == elem)[0][0] for elem in expectedVal])
                