import numpy as np
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









