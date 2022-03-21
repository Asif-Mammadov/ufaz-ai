import numpy as np

class MSE:
  def calc(Yhat, Y):
    return np.sum((Y - Yhat) ** 2 / 2, axis=1, keepdims=True) / Y.shape[1]
    
  def grad(Yhat, Y):
    return -(Y - Yhat) 

class CrossEntropy:
  def calc(Yhat, Y):
    cost = -(np.dot(Y, np.log(Yhat).T) + np.dot(1-Y, np.log(1-Yhat).T)) / Y.shape[1]
    return cost

  def grad(Yhat, Y):
    return -(np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))
