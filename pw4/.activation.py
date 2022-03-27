import numpy as np

class Sigmoid:
  """Sigmoid activation function.
  """
  def calc(Z:np.ndarray)->np.ndarray:
    """Calculates the activation of Z.

    Args:
        Z (np.ndarray): A matrix.

    Returns:
        np.ndarray: New matrix after activation function.
    """    
    return 1/(1 + np.exp(-Z))

  def grad(Z):
    return Sigmoid.calc(Z) * (1 - Sigmoid.calc(Z))

class Relu:
  """ReLU (Rectified Linear Unit) activation function.
  """
  def calc(Z:np.ndarray)->np.ndarray:
    """Calculates the activation of Z.

    Args:
        Z (np.ndarray): A matrix.

    Returns:
        np.ndarray: New matrix after activation function.
    """    
    return Z * (Z > 0)

  def grad(Z:np.ndarray)->np.ndarray:
    """Finds the gradient.

    Args:
        Z (np.ndarray): A matrix.

    Returns:
        np.ndarray: New matrix after the gradient of activation function.
    """
    return Z > 0

class Softmax:
  """Softmax activation function.
  """

  def calc(Z):
    """Calculates the activation of Z.

    Args:
        Z (np.ndarray): A matrix.

    Returns:
        np.ndarray: New matrix after activation function.
    """    
    return np.exp(Z)/np.sum(np.exp(Z))

  def grad(Z):
    """Finds the gradient.

    Args:
        Z (np.ndarray): A matrix.

    Returns:
        np.ndarray: New matrix after the gradient of activation function.
    """
    gradient = np.zeros(Z.shape)
    for i in range(Z.shape[0]):
      for j in range(Z.shape[1]):
        if i == j:
            gradient[i,j] = Z[i, j] * (1-Z[i, j])
        else: 
            gradient[i,j] = -Z[i, j] * Z[i, j]
    return gradient

