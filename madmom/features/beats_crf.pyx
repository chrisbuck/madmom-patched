# encoding: utf-8
from __future__ import absolute_import, division, print_function
import numpy as np
cimport numpy as np
cimport cython
from numpy.math cimport INFINITY
from scipy.ndimage import correlate1d

def initial_distribution(num_states, interval):
    """
    Compute the initial distribution.
    """
    distribution = np.zeros(num_states)
    distribution[:interval] = 1.0 / interval
    return distribution

def transition_distribution(num_states, interval, slope):
    """
    Compute the transition distribution.
    """
    distribution = np.zeros(num_states)
    for i in range(interval):
        distribution[i] = slope * (interval - i) / interval + (1 - slope) / interval
    return distribution
