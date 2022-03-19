import numpy as np
import pandas as pd
import activation
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


def trainingEpoch(X_train, Y_train, hidden_layers=[5, 5, 3], batch_size=5):
  input_layer = X_train.shape[0]  
  final_layer = Y_train.shape[0]
  layers = [input_layer] + hidden_layers + [final_layer]
  print("Layers: {}".format(layers))
  W = []
  B = []
  n_instances = X_train.shape[1]
  shuffleTrainingData(X_train, Y_train)
  i_epoch = 0
  # Generate weights and biases
  for i in range(1, len(layers)):
    w = np.random.random((layers[i-1], layers[i]))
    b = np.zeros((layers[i], 1))
    W.append(w)
    B.append(b)

  # Launch epochs
  for i in range(0, n_instances, batch_size):
    print("Epoch {}:".format(i_epoch))
    x_train = X_train[:, i:i+batch_size]
    y_train = Y_train[:, i:i+batch_size]
    # Forward propagation
    A = x_train
    for i in range(len(W)):
      Z = np.dot(W[i].T, A) + B[i]
      if i == len(W) - 1:
        A = activation.softmax(Z)
      else:
        A = activation.relu(Z)
    print(A)
    #TODO : Back propagation
    i_epoch += 1