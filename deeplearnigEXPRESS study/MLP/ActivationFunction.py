import numpy as np

def step(x):
    result = x > 0.000001
    return result.astype(np.int)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def ReLU(x):
    return np.maximun(x,0)
