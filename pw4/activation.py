from os import stat
import numpy as np
class Relu:
  @staticmethod 
  def calc(Z):
    return Z * (Z > 0)

  @staticmethod
  def grad(Z):
    return Z > 0

class Softmax:
  @staticmethod
  def calc(Z):
    return np.exp(Z)/np.sum(np.exp(Z))

  @staticmethod
  def grad(Z):
    softmax = Softmax.calc(Z)
    return softmax * np.identity(softmax.size) - np.dot(softmax.T, softmax)

class Sigmoid:
  def calc(Z):
    return 1/(1 + np.exp(-Z))
  def grad(Z):
    return Sigmoid.calc(Z) * (1 - Sigmoid.calc(Z))
# def relu(Z):
#   return Z * (Z > 0)

# def softmax(Z):
#   return np.exp(Z)/np.sum(np.exp(Z))

# def d_relu(Z):
#   return Z > 0

# def d_softmax(Z):
#   softmax = softmax(Z)
#   return softmax * np.identity(softmax.size) - np.dot(softmax.T, softmax)

# SM = self.value.reshape((-1,1))
# jac = np.diagflat(self.value) - np.dot(SM, SM.T)
