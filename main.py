import numpy as np
from models.nn_model import NNModel
from models.layer import Layer
import activations.activation_functions as af

model = NNModel(2, Layer(3), Layer(1 ))
model.predict(np.array([[1, 2],[4,5]]))

