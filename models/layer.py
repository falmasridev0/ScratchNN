import numpy as np
from models.neuron import Neuron
from activations.activation_functions import relu


class Layer:

    def __init__(self, units, activation=relu):
        self.is_initialized = False
        self._neurons = None
        self._bias = None
        self._weights = None
        self.n_features = 0
        self.activation = activation
        self._units = units  # Number of neurons

    def initialize_params(self):
        self._weights = np.random.random_sample((self.n_features, self._units))
        self._bias = np.zeros((self._units,))
        self._neurons = np.array([Neuron(self._weights[:, i], self._bias[i]) for i in range(self._units)])
        self.is_initialized = True
        return self.is_initialized

    def forward_pass(self,x):
        return self.activation(np.dot(x,self._weights) + self._bias)

    def get_weights(self):
        return self._weights

    def get_bias(self):
        return self._bias

    def set_weights(self, weights):
        self._weights = weights

    def set_bias(self, bias):
        self._bias = bias

    def set_n_features(self, n_features):
        self.n_features = n_features

    def get_n_features(self):
        return self.n_features

    def get_units(self):
        return self._units
