from numpy import exp,maximum


def sigmoid(z):
    return 1 / (1 + exp(z))


def relu(z):
    return maximum(0, z)
