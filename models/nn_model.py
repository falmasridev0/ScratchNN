import numpy as np
from models.layer import Layer


class NNModel:
    def __init__(self, n_features: int, *layers: Layer):
        self.n_features = n_features
        self.layers = layers
        self.layers[0].n_features = self.n_features
        self.layers[0].initialize_params()
        for i in range(1,len(self.layers)):
            layers[i].n_features = layers[i-1].get_units()
            layers[i].initialize_params()

    def predict(self, inputs: np.ndarray):
        current_input = inputs  # LAYER 0
        for i in range(len(self.layers)):
            current_input = self.layers[i].forward_pass(current_input)
            print(f"Values: {current_input} LAYER #{i+1}")
        return current_input
