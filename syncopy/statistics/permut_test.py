# -*- coding: utf-8 -*-
#
# Routines to set up permutation testing for multiple comparison
#

import numpy as np
import logging
import platform

from syncopy.shared.errors import SPYValueError
from syncopy import synthdata

nTrials = 30
ad = synthdata.white_noise(nTrials)
# put 3 different labels
trldef = ad.trialdefinition
trldef = np.column_stack([trldef, np.random.randint(3, size=nTrials)])
ad.trialdefinition = trldef


def create_permutation(trialinfo, label1, label2, label_col=0):
    """
    Given a `trialinfo` and two labels,
    returns a new `trialdefinition` array with those two labels randomized.
    """

    # TODO warn if unequal label distribution

    # get location of labels
    loc1 = np.where(trialinfo[:, label_col] == label1)[0]
    loc2 = np.where(trialinfo[:, label_col] == label2)[0]

    # maybe better check somewhere else beforehand
    if loc1.size == 0:
        raise SPYValueError(f"at least one oocurence of label {label1}",
                            'label1', f"label {label1} not found in trialinfo")
    if loc2.size == 0:
        raise SPYValueError(f"at least one oocurence of label {label1}",
                            'label1', f"label {label1} not found in trialinfo")
    
    # now randomly exchange the positions ONLY of these two labels
    all_pos = np.r_[loc1, loc2]
    np.random.shuffle(all_pos)
    loc1 = all_pos[:len(loc1)]
    loc2 = all_pos[len(loc1):]

    # and create a new trialinfo array
    new_trialinfo = trialinfo.copy()
    new_trialinfo[loc1, label_col] = label1
    new_trialinfo[loc2, label_col] = label2

    return new_trialinfo
    
