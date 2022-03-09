import numpy as np
def split_train_test(X, Y, test_portion=0.2, seed=42):
  # share should be in [0, 1]
  np.random.seed(seed) 
  np.random.shuffle(X)
  np.random.shuffle(Y)
  data_size = X.shape[1]
  train_size = np.int64(np.floor( (1 - test_portion) * data_size))
  X_train = X[:, :train_size]
  X_test = X[:, train_size:]
  Y_train = Y[:train_size]
  Y_test = Y[train_size:]
  return X_train, Y_train, X_test, Y_test
