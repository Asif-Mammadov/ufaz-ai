import numpy as np
class Relu:

  def calc(Z):
    return Z * (Z > 0)

  def grad(Z):
    return Z > 0

class Softmax:

  def calc(Z):
    return np.exp(Z)/np.sum(np.exp(Z))

  def grad(Z):
    gradient = np.zeros(Z.shape)
    for i in range(Z.shape[0]):
      for j in range(Z.shape[1]):
        if i == j:
            gradient[i,j] = Z[i, j] * (1-Z[i, j])
        else: 
            gradient[i,j] = -Z[i, j] * Z[i, j]
    return gradient

class Sigmoid:

  def calc(Z):
    return 1/(1 + np.exp(-Z))

  def grad(Z):
    return Sigmoid.calc(Z) * (1 - Sigmoid.calc(Z))
