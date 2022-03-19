import numpy as np
def relu(Z):
  return Z * (Z > 0)

def softmax(Z):
  return np.exp(Z)/np.sum(np.exp(Z))