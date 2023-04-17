import numpy as np


class Neuron:
    def __init__(self, weights,bias=0):
        self._weights = weights
        self._bias = bias

    def get_weights(self):
        return self._weights

    def get_bias(self):
        return self._bias

    def set_weights(self,weights):
        self._weights = weights

    def set_bias(self, bias):
        self._bias = bias



