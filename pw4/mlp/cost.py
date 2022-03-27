import numpy as np

class MSE:
  """Mean Squared Error Cost Function.
  """
  def calc(Yhat:np.ndarray, Y:np.ndarray)->np.ndarray:
    """Calculates the error.

    Args:
        Yhat (np.ndarray): Predicted values.
        Y (np.ndarray): Target values.

    Returns:
        np.ndarray: Error value.
    """
    return np.sum((Y - Yhat) ** 2 / 2, axis=1, keepdims=True) / Y.shape[1]
    
  def grad(Yhat:np.ndarray, Y:np.ndarray)->np.ndarray:
    """Finds gradient value.

    Args:
        Yhat (np.ndarray): Predicted values.
        Y (np.ndarray): Target values.

    Returns:
        np.ndarray: Gradient matrix.
    """
    return -(Y - Yhat) 

class CrossEntropy:
  """Cross Entropy Cost Function.
  """  
  def calc(Yhat:np.ndarray, Y:np.ndarray)->np.ndarray:
    """Calculates the error.

    Args:
        Yhat (np.ndarray): Predicted values.
        Y (np.ndarray): Target values.

    Returns:
        np.ndarray: Error value.
    """
    cost = -(np.dot(Y, np.log(Yhat).T) + np.dot(1-Y, np.log(1-Yhat).T)) / Y.shape[1]
    return cost

  def grad(Yhat:np.ndarray, Y:np.ndarray)->np.ndarray:
    """Finds gradient value.

    Args:
        Yhat (np.ndarray): Predicted values.
        Y (np.ndarray): Target values.

    Returns:
        np.ndarray: Gradient matrix.
    """
    return -(np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))
