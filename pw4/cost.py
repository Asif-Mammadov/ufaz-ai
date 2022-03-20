import numpy as np

class MSE:

  def calc(Yhat, Y):
    return np.sum((Yhat - Y) ** 2) / (2 * Y.shape[1])
    
  def grad(Yhat, Y):
    return np.sum((Yhat - Y)) / Y.shape[1]

class CrossEntropy:
  def calc(Yhat, Y):
    Yhat[Yhat==0] = 1e-3
    Y[Y==0] = 1e-3
    Yhat[Yhat==1] -= 1e-3
    Y[Y==1] -= 1e-3
    cost = -np.sum(np.dot(Y, np.log(Yhat).T)) / Y.shape[1]
    return cost

  def grad(Yhat, Y):
    return - np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat)
