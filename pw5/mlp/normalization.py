import numpy as np
def linear_scaling_max(X: np.ndarray)->np.ndarray:
  """
  X' = X / X_max

  Args:
    X (np.ndarray): 2D matrix to scale.

  Returns:
    np.ndarray : Normalized 2D matrix X.
  """
  X = divide_ignored(X, X.max(axis=1)[:, np.newaxis])
  return X

def linear_scaling(X: np.ndarray)->np.ndarray:
  """
  Linear scaling normalization:
  X' = (X - X_min) / (X_max - X_min)

  Args:
    X (np.ndarray): 2D matrix to scale.

  Returns:
    np.ndarray : Normalized 2D matrix X.
  """
  X_min = X.min(axis=1)[:, np.newaxis]
  X_max = X.max(axis=1)[:, np.newaxis]
  X = divide_ignored(X - X_min, X_max - X_min)
  return X

def z_score(X:np.ndarray)->np.ndarray:
  """
  Z-score scaling normalization:
  X' = (X - mean) / std

  Args:
    X (np.ndarray): 2D matrix to scale.

  Returns:
    np.ndarray : Normalized 2D matrix X.
  """
  mean = X.mean(axis=1)[:, np.newaxis]
  std = X.std(axis=1)[:, np.newaxis]
  X = divide_ignored(X - mean, std)
  return X

def divide_ignored(a:np.ndarray, b:np.ndarray)->np.ndarray:
  """
  Divides 2 matrices ignoring division by 0.
  Args:
    a (np.ndarray): 2D matrix.
    b (np.ndarray): 2D matrix.

  Returns:
    np.ndarray : New matrix a / b.
  """
  with np.errstate(divide='ignore', invalid='ignore'):
    tmp = np.true_divide(a, b)
    tmp[tmp == np.inf] = 0
    tmp = np.nan_to_num(tmp)
  return tmp
