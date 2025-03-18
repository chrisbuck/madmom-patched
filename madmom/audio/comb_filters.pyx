# encoding: utf-8
from __future__ import absolute_import, division, print_function
import numpy as np
cimport numpy as np
cimport cython
from madmom.processors import Processor

# feed forward comb filter
def feed_forward_comb_filter(signal, tau, alpha):
    """
    Filter the signal with a feed forward comb filter.

    Parameters
    ----------
    signal : numpy array
        Signal to be filtered.
    tau : int
        Delay in samples.
    alpha : float
        Scaling factor.

    Returns
    -------
    filtered_signal : numpy array
        Filtered signal.

    """
    filtered_signal = np.zeros_like(signal)
    filtered_signal[tau:] = signal[:-tau] + alpha * signal[tau:]
    return filtered_signal


# backward comb filter
def feed_backward_comb_filter(signal, tau, alpha):
    """
    Filter the signal with a feed backward comb filter.

    Parameters
    ----------
    signal : numpy array
        Signal to be filtered.
    tau : int
        Delay in samples.
    alpha : float
        Scaling factor.

    Returns
    -------
    filtered_signal : numpy array
        Filtered signal.

    """
    filtered_signal = np.zeros_like(signal)
    filtered_signal[:-tau] = signal[tau:] + alpha * signal[:-tau]
    return filtered_signal


# comb filterbank processor
class CombFilterbankProcessor(Processor):
    """
    Comb filterbank processor.

    Parameters
    ----------
    mode : {'forward', 'backward'}
        Mode of the comb filterbank.
    taus : list
        List of delays.
    alpha : float
        Scaling factor.

    """
    def __init__(self, mode, taus, alpha):
        self.mode = mode
        self.taus = taus
        self.alpha = alpha

    def process(self, activations):
        """
        Process the activations.

        Parameters
        ----------
        activations : numpy array
            Activations to be processed.

        Returns
        -------
        processed_activations : numpy array
            Processed activations.

        """
        processed_activations = np.zeros_like(activations)
        for tau in self.taus:
            if self.mode == 'forward':
                processed_activations += feed_forward_comb_filter(activations, tau, self.alpha)
            elif self.mode == 'backward':
                processed_activations += feed_backward_comb_filter(activations, tau, self.alpha)
            else:
                raise ValueError("Unknown mode: %s" % self.mode)
        return processed_activations
