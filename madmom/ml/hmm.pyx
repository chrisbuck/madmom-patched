# encoding: utf-8
from __future__ import absolute_import, division, print_function
import numpy as np
cimport numpy as np
cimport cython
from numpy.math cimport INFINITY
np.import_array()
ctypedef np.uint32_t uint32_t

class TransitionModel(object):
    def __init__(self, transitions, num_states):
        self.transitions = transitions
        self.num_states = num_states

    def get_transitions(self, state):
        return self.transitions[state]

class ObservationModel(object):
    def __init__(self, observations, num_states):
        self.observations = observations
        self.num_states = num_states

    def get_observations(self, state):
        return self.observations[state]

class HiddenMarkovModel(object):
    def __init__(self, transition_model, observation_model, initial_distribution=None):
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.initial_distribution = initial_distribution

    def decode(self, activations):
        return viterbi(activations, self.transition_model, self.initial_distribution)

def viterbi(activations, transition_model, initial_distribution=None):
    num_frames, num_states = activations.shape
    v = np.zeros((num_frames, num_states))
    backpointer = np.zeros((num_frames, num_states), dtype=np.int32)
    if initial_distribution is None:
        v[0, :] = activations[0, :]
    else:
        v[0, :] = activations[0, :] * initial_distribution
    for t in range(1, num_frames):
        for s in range(num_states):
            transition_probs = transition_model.get_transitions(s)
            trans_scores = v[t - 1, :] * transition_probs
            best_state = np.argmax(trans_scores)
            v[t, s] = trans_scores[best_state] * activations[t, s]
            backpointer[t, s] = best_state
    best_path = np.zeros(num_frames, dtype=np.int32)
    best_path[-1] = np.argmax(v[-1, :])
    for t in range(num_frames - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]
    return best_path
