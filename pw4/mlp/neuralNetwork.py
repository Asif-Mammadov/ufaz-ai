import numpy as np
from mlp.activation import Sigmoid, Softmax, Relu
from mlp.cost import MSE
from mlp import utils, normalization
import matplotlib.pyplot as plt

from mlp.utils import get_accuracy


class NeuralNetwork:
  """Class for designing, training and testing custom Neural Network.

    Attributes
    ----------
    X_train : np.ndarray
        Input data for training.
    Y_train : np.ndarray
        Labels of training data (output).
    X_test : np.ndarray
        Input data for test.
    Y_test : np.ndarray
        Labels of test data.
    n_instances: int
        Number of instances in a traing set.
    n_attributes: int
        Number of attributes to train with.
    lr : float
        Learning rate.
    lr_reduce : float
        Decrease rate of learning rate after each epoch.
    cost : <<cost.Class>>. Defaults to MSE. Refer to .cost.py
      Cost function used.
    batch_size : int. Defaults to None.
        Batch size for training. If None, batch size equals to n_instances.
    normalizeData : bool. Defaults to Fasle.
      Defines whether to normalize the data for dataset.

    Methods
    -------
  """  
  def __init__(self, X_train:np.ndarray, Y_train:np.ndarray, lr:float, lr_reduce:float=1, cost=MSE, batch_size:int=None, normalization=None):
    """

    Args:
        X_train (np.ndarray): Input data for training.
        Y_train (np.ndarray): Labels of training data (output).
        lr (float): Learning rate.
        lr_reduce (float): Decrease rate of learning rate after each epoch. lr /= lr_reduce
        cost (_type_, optional): Cost function used. Defaults to MSE. Refer to .cost.py
        batch_size (int, optional): Batch size for training. Defaults to None. If None, batch size equals to n_instances.
        normalization (_type_, optional): Defines the normalization function. Defaults to None.
    """
    self.X_train = X_train.astype('float')
    self.Y_train = Y_train.astype('float')
    self.normalization = normalization
    if callable(self.normalization):
      self.X_train = self.normalization(self.X_train)
      self.Y_train = self.normalization(self.Y_train)
    self.n_instances = self.X_train.shape[1]
    self.n_attributes = self.X_train.shape[0]
    self.batch_size = batch_size if batch_size else self.n_instances
    self.activations = [Sigmoid]
    self.cost = cost
    self.lr = lr
    self.lr_reduce = lr_reduce
    self.W = []
    self.B = []
    self.A = []
    self.Z = []
    self.layers = [X_train.shape[0], Y_train.shape[0]]
    self.costs = []
    self.avg_costs = []
    self.accuracies = []
    self.i_iter = 0

  def add_hidden_layer(self, n_nodes:int, activation):
    """Appends a layer to the network architecture.

      The layer is put just before the output layer.
    Args:
        n_nodes (int): Number of nodes(perceptrons) for this layer.
        activation (activation.Class): Activation function to be used. Refer to .activation.py
    """
    self.layers.insert(len(self.layers) - 1, n_nodes)
    self.activations.insert(len(self.activations)-1, activation)
  
  def set_layer(self, index:int, n_nodes:int=None, activation=None):
    """Changes layer properties.

    Args:
        index (int): Index of the layer in neural network.
        n_nodes (int): Number of nodes (perceptrons) to set for this layer. Defaults to None.
        activation (activation.Class): Activation function to be used. Refer to .activation.py. Defaults to None.

        If parameter is None, the value won't change.
    """    
    if n_nodes:
      self.layers[index] = n_nodes
    if activation:
      self.activations[index] = activation

  def info(self):
    """Displays information about the neural network.
    """
    print("N Training instances: {}\nN attributes: {}\nBatch size: {} \nLearning rate: {}\nCost function:{}\nNormalization:{}"
      .format(self.n_instances, self.n_attributes, self.batch_size, self.lr, self.cost, self.normalization))
    print("\nArchitecutre:")
    print("Layer {}: {} nodes".format(0, self.layers[0]))
    for i in range(1, len(self.layers)):
      print("Layer {}: {} nodes | {}".format(i, self.layers[i], self.activations[i-1]))

  def generate_parameters(self):
    """Generate parameters and their dimensions for neural network.
    """
    for i in range(1, len(self.layers)):
      w = np.random.random((self.layers[i], self.layers[i-1])) * 0.01
      b = np.zeros((self.layers[i], 1))
      z = np.zeros((self.layers[i], 1))
      a = np.zeros((self.layers[i], 1))
      self.W.append(w)
      self.B.append(b)
      self.Z.append(z)
      self.A.append(a)
    self.A.append(a)

  def train(self, n_epochs=None, verbose=True):
    infFlag = False
    min_error = np.inf
    if not n_epochs:
      infFlag = True
      # Dynamic change
      n_epochs = 100
    self.generate_parameters()
    pos_slope = 0
    i = 0
    error_i = 0
    errors = None
    while i < n_epochs:
      if verbose:
        print("-" * 20, "Epoch {}:".format(i), "-" * 20)
      self.training_epoch(verbose=verbose)
      conf_matrix = self.testPrediction(self.X_train, self.Y_train, verbose=verbose)
      accuracy = get_accuracy(conf_matrix)
      self.accuracies.append(accuracy)
      self.lr /= self.lr_reduce
      if infFlag:
        n_epochs += 1
        new_error_i = len(self.costs)
        if errors:
          new_avr_error = self.avg(self.costs[error_i:new_error_i])
          if new_avr_error > min_error:
            pos_slope += 1
          else:
            min_error = new_avr_error
            pos_slope = 0
        if pos_slope >= 10:
          break
        errors = self.avg(self.costs[error_i:new_error_i])
        self.avg_costs.append([new_error_i, errors])
        error_i = new_error_i
      i += 1


  def training_epoch(self, verbose:bool=True):
    """Trains the network.

    Args:
        verbose (bool, optional): Verbose mode with errors displayed for each iteration. Defaults to True.
    """
    utils.shuffleTrainingData(self.X_train, self.Y_train)
    for i in range(0, self.n_instances, self.batch_size):
      if verbose:
        print("Batch {}:".format(self.i_iter), end=" ")
      x_train = self.X_train[:, i:i+self.batch_size]
      y_train = self.Y_train[:, i:i+self.batch_size]

      # Forward propagation
      self.A[0] = x_train 
      for j in range(len(self.W)):
        self.Z[j] = np.dot(self.W[j], self.A[j]) + self.B[j]
        self.A[j+1] = self.activations[j].calc(self.Z[j])

      error= self.cost.calc(self.A[-1], y_train)
      error_general = np.sum(error)
      if verbose:
        print("Error : {}".format(error_general))
      self.costs.append(error_general)

      # Backwards propagation
      dA = self.cost.grad(self.A[-1], y_train)
      for j in range(len(self.W)) [::-1]:
        dz = dA * self.activations[j].grad(self.Z[j])
        dw = np.dot(dz, self.A[j].T)
        db = np.sum(dz, axis=1, keepdims=True) / self.batch_size
        dA = np.dot(self.W[j].T, dz)

        self.W[j] -= self.lr * dw
        self.B[j] -= self.lr * db

      self.i_iter += 1
  
  def predict(self, X:np.ndarray)->np.ndarray:
    """Predicts the Y (output) value given the X (input).


    Args:
        X (np.ndarray): Input data.

    Returns:
        Y (np.ndarray): Output (label) value.
    """
    a = X.copy()
    if callable(self.normalization):
        a = self.normalization(a)
    for j in range(len(self.W)):
      z = np.dot(self.W[j], a) + self.B[j]
      a = self.activations[j].calc(z)
    if a.shape[0] == 1:
      return a > 0.5
    else:
      for i in range(a.shape[1]):
        index = a[:, i].argmax()
        a[:, i] = np.zeros(a.shape[0])
        a[:, i][index] = 1
      return a

  def testPrediction(self, X_test:np.ndarray, Y_test:np.ndarray, verbose:bool=True)->tuple:
    """Tests the predicted data with expected (target) data.

    Args:
        X_test (np.ndarray) : Input test data.
        Y_test (np.ndarray) : Label (output) test data.
        verbose (bool) : Get the verbose output. Defautls to True.

    Returns:
        tuple: (true, false) predictions
    """    
    tp = fp = tn = fn = 0
    Yhat = self.predict(X_test)
    conf_matrix = np.zeros((Y_test.shape[0], Y_test.shape[0]))
    for i in range(Yhat.shape[1]):
      conf_matrix[Y_test[:, i].argmax(), Yhat[:, i].argmax()] += 1
    if verbose:
      print("Accuracy : {}".format(get_accuracy(conf_matrix)))
    return conf_matrix

  def avg(self, l):
    """Finds average of an array l
    """
    return sum(l) / len(l)

  def plot_stats(self):
    """Plots the error and accuracy of a model on a given data.
    """
    avg_costs = np.array(self.avg_costs)
    fig, axs = plt.subplots(2)
    axs[0].plot(self.costs, c='r', label="error")
    if len(avg_costs) > 0:
      axs[0].plot(avg_costs.T[0], avg_costs.T[1], c='black', label="average error")
    axs[0].set_xlabel("Iterations")
    axs[0].set_ylabel("Error")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(self.accuracies, marker='.', c='b')
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid(True)
    plt.show()