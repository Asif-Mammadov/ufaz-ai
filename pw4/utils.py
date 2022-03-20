import numpy as np
import pandas as pd
from activation import Relu, Sigmoid, Softmax
def split_train_test(X: np.ndarray, Y: np.ndarray, test_portion:float=0.2, seed:int=None)->np.ndarray:
  """Splits the data into training and testing sets.

  Args:
      X (np.ndarray): Input values.
      Y (np.ndarray): Output values.
      test_portion (float, optional): Portion of data to be used for testing. Defaults to 0.2.
      seed (int, optional): Seed. Defaults to None.

  Returns:
      X_train, Y_train, X_test, Y_test (np.ndarray): Split values into training and testing.
  """
  np.random.seed(seed) 
  np.random.shuffle(X)
  np.random.shuffle(Y)
  p = np.random.permutation(X.shape[1])
  X = X[:, p]
  Y = Y[:, p]

  data_size = X.shape[1]
  train_size = np.int64(np.floor( (1 - test_portion) * data_size))
  X_train = X[:, :train_size]
  X_test = X[:, train_size:]
  Y_train = Y[:, :train_size]
  Y_test = Y[:, train_size:]
  return X_train, Y_train, X_test, Y_test

def createOneHot(dataset: pd.DataFrame, column: str) -> np.ndarray:
  """Represents given column in a oneHot format

  Args:
      dataset (pd.DataFrame) : Dataset
      column (str) : column to represent in OneHot format

  Returns:
      np.ndarray: OneHot representations of labels
  """
  Y = dataset.loc[:, column]
  unique = Y.unique()
  unique_size = len(Y.unique())
  size = len(Y)
  oneHot = np.zeros((unique_size,size))
  for i in range(unique_size):
    oneHot[i] = np.array([Y == unique[i]])
  return oneHot

def loadAttributes(dataset: pd.DataFrame, attributes: list) -> np.array:
  """Loads attribute values in a matrix. Represents an input of neural network.

  Args:
      dataset (pd.DataFrame): Dataset
      attributes (list): list of attributes (strings) to extract from dataset
  Returns:
      np.array: Maxtrix with rows (attributes) and columns (instances)
  """ 
  return dataset.loc[:, attributes].to_numpy().T

def shuffleTrainingData(X_train: np.ndarray, Y_train: np.ndarray, seed:int=None):
  """Shuffles the matrix of training data using Fisher-Yates algorithm
  https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm

  Args:
      X_train (np.ndarray): Input of training data
      Y_train (np.ndarray): Output of training data
      seed (int, optional): Seed. Default is None.
  """
  np.random.seed(seed)
  ncols = X_train.shape[1]
  if (ncols != Y_train.shape[1]):
    raise ValueError("Number of instances of X_train and Y_train should be the same.")

  for i in range(1, ncols)[::-1]:
    j = np.random.randint(0, i)
    X_train[:, [i, j]] = X_train[:, [j, i]]
    Y_train[:, [i, j]] = Y_train[:, [j, i]]

def cross_entropy_loss(Yhat, Y):
  Y[Y == 0] = 1e-16
  Y[Y == 1] -= 1e-16 
  return -np.sum(Yhat * np.log2(Y) + (1 - Yhat) * np.log2(1 - Y))

def mse(Yhat, Y):
  return np.sum((Yhat - Y) ** 2) / (2 * Y.shape[1])

def d_mse(Yhat, Y):
  return np.sum((Yhat - Y)) / Y.shape[1]

def cross_entropy(Yhat, Y):
  cost = (-1/Y.shape[1]) * np.dot(Y, np.log(Yhat).T) + np.dot((1-Y), np.log(1-Yhat).T)
  return cost

def trainingEpoch(X_train, Y_train, hidden_layers=[3], activations=[Sigmoid, Sigmoid], cost=mse, batch_size=5, alpha=0.1):
  shuffleTrainingData(X_train, Y_train)
  input_layer = X_train.shape[0]
  final_layer = Y_train.shape[0]
  layers = [input_layer, *hidden_layers, final_layer]
  W = []
  B = []
  A = []
  Z = []
  errors = []
  n_instances = X_train.shape[1]
  shuffleTrainingData(X_train, Y_train)
  i_epoch = 0
  # Generate weights and biases
  for i in range(1, len(layers)):
    w = np.random.random((layers[i], layers[i-1])) * 0.01
    b = np.zeros((layers[i], 1))
    W.append(w)
    B.append(b)

  # Launch epochs
  # for _ in range(10):
  for i in range(0, n_instances, batch_size):
    print("Epoch {}:".format(i_epoch))
    x_train = X_train[:, i:i+batch_size]
    y_train = Y_train[:, i:i+batch_size]
    # Forward propagation
    a = x_train
    A.append(a)
    for j in range(len(W)):
      z = np.dot(W[j], a) + B[j]
      a = activations[j].calc(z)
      Z.append(z)
      A.append(a)
    error= cost(a, y_train)
    errors.append(error)

    # dA = - np.divide(y_train, a) - np.divide(1 - y_train, 1 - a)
    dA = d_mse(a, y_train)
    # Backwards propagation
    for j in range(len(W))[::-1]:
      dZ = dA * activations[j].grad(Z[j])
      dw = np.dot(dZ, A[j].T)
      db = np.sum(dZ, axis=1, keepdims=True) / batch_size
      if j != 0:
        dA = np.dot(W[j].T, dZ)

      W[j] -= alpha * dw
      B[j] -= alpha * db

    i_epoch += 1
  # import matplotlib.pyplot as plt
  # plt.plot(np.arange(i_epoch), errors)
  # plt.show()
  return {"W": W, "B": B, "activations": activations}
  