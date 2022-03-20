import numpy as np
from activation import Sigmoid
from cost import MSE
import utils
class NeuralNetwork:
  def __init__(self, X_train, Y_train, lr, cost=MSE, batch_size=None, normalizeData=False):
    self.X_train = X_train
    self.Y_train = Y_train
    if normalizeData:
      self.X_train /= self.X_train.max()
      self.Y_train /= self.Y_train.max()
    self.n_instances = self.X_train.shape[1]
    self.n_attributes = self.X_train.shape[0]
    self.batch_size = batch_size if batch_size else self.n_instances
    self.activations = [Sigmoid]
    self.cost = cost
    self.lr = lr
    self.W = []
    self.B = []
    self.A = []
    self.Z = []
    self.layers = [X_train.shape[0], Y_train.shape[0]]
    self.costs = []
    self.i_epoch = 0
    self.normalizeData=normalizeData

  def add_hidden_layer(self, n_nodes, activation):
    self.layers.insert(len(self.layers) - 1, n_nodes)
    self.activations.insert(len(self.activations)-1, activation)
  
  def set_layer(self, index, n_nodes, activation):
    self.layers[index] = n_nodes
    self.activations[index] = activation

  def info(self):
    print("N Training instances: {}\n \
      N attributes {} \n \
      Batch size: {} \n \
      Learning rate: {}".format(self.n_instances, self.n_attributes, self.batch_size, self.lr))
    print("Architecutre:")
    print("{}: {}".format(0, self.layers[0]))
    for i in range(1, len(self.layers)):
      print("{}: {}  {}".format(i, self.layers[i], self.activations[i-1]))

  def training_epoch(self, n_iter=1):
    print("Training started")
    # Generate weights and biases
    for i in range(1, len(self.layers)):
      w = np.random.random((self.layers[i], self.layers[i-1])) * 0.01
      b = np.zeros((self.layers[i], 1))
      self.W.append(w)
      self.B.append(b)

    for _ in range(n_iter):
      print("Iteration {}:".format(_))
      utils.shuffleTrainingData(self.X_train, self.Y_train)
      for i in range(0, self.n_instances, self.batch_size):
        print("Epoch {}:".format(self.i_epoch), end=" ")
        x_train = self.X_train[:, i:i+self.batch_size]
        y_train = self.Y_train[:, i:i+self.batch_size]

        # Forward propagation
        a = x_train
        self.A.append(a)
        for j in range(len(self.W)):
          z = np.dot(self.W[j], a) + self.B[j]
          a = self.activations[j].calc(z)
          self.Z.append(z)
          self.A.append(a)
        error= self.cost.calc(a, y_train)
        print("Error : {}".format(error))
        self.costs.append(error)

        # Backwards propagation
        dA = self.cost.grad(a, y_train)
        for j in range(len(self.W))[::-1]:
          dZ = dA * self.activations[j].grad(self.Z[j])
          dw = np.dot(dZ, self.A[j].T)
          db = np.sum(dZ, axis=1, keepdims=True) / self.batch_size
          if j != 0:
            dA = np.dot(self.W[j].T, dZ)
           
          self.W[j] -= self.lr * dw
          self.B[j] -= self.lr * db

        self.i_epoch += 1
  
  def predict(self, X):
    a = X.copy()
    if self.normalizeData:
      a /= a.max()
    for j in range(len(self.W)):
      z = np.dot(self.W[j], a) + self.B[j]
      a = self.activations[j].calc(z)
    
    for i in range(a.shape[1]):
      index = a[:, i].argmax()
      a[:, i] = np.zeros(a.shape[0])
      a[:, i][index] = 1
    return a

  def testPrediction(self, X_test, Y_test):
    t = f = 0
    Yhat = self.predict(X_test)
    for i in range(Yhat.shape[1]):
      if Yhat[:, i].argmax() == Y_test[:, i].argmax():
        t += 1
      else:
        f += 1
    print("Correct: {}\nFalse: {}\nAccuracy:{}".format(t, f, t / (t + f)))
    return t, f

  def plot_costs(self):
    import matplotlib.pyplot as plt
    plt.plot(np.arange(self.i_epoch), self.costs)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.show()