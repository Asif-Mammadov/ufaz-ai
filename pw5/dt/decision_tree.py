import numpy as np
from dt.node import Node

class DecisionTree:
  def __init__(self, X_train:np.ndarray, Y_train:np.ndarray, max_depth:int=4, attr_names:list=None):
    """Initialize decision tree.

    Args:
      X_train (np.ndarray): 2D matrix with rows as attributes and columns as instances.
      Y_train (np.ndarray): 1D array with labeled values.
      max_depth (int): Maximum depth of a tree.
      attr_names (list, optional): Names of attributes in the same order as it is in X_train. Defaults to None.
    """
    self.X_train = X_train
    self.Y_train = Y_train
    self.data = np.concatenate([self.X_train, self.Y_train], axis=0)
    self.max_depth = 4
    self.root_node = None
    self.attr_names = list(attr_names)

  def separate_by_value(self, data:np.ndarray, attr_index:int, value:float):
    """
    Separates an array in two parts according the given value.

    Args:
      data (np.ndarray): 2D input matrix with rows as attributes and columns as instances.
      attr_index (int): Index of the attribute in dataset.
      value (float): Separating value.

    Returns (np.ndarray, np.ndarray) : 2 data arrays (2D matrix each).

    """
    less = np.where(data[attr_index] <= value)[0]
    not_less = np.where(data[attr_index] > value)[0]
    return data[:, less], data[:, not_less]

  def get_entropy(self, data:np.ndarray):
    """
    Calculates the entropy of the given data.
    Args:
      data (np.ndarray): 2D input matrix with rows as attributes and columns as instances.

    Returns (float): Entropy value.

    """
    data_labels = data[-1]
    data_labels_length = len(data_labels)
    s = 0
    for unique_value in np.unique(data_labels):
      p = len(data_labels[data_labels == unique_value]) / data_labels_length
      s += p * np.log2(p)
    return -1 * s

  def get_best_decision(self, data:np.ndarray):
    """
    Finds attribute and value with the highest discriminative power.

    Args:
      data (np.ndarray): 2D input matrix with rows as attributes and columns as instances.

    Returns (dict):
      dict["attr_index"] : index of an attribute.
      dict["attr_name"] : name of the attribute. Defaults to None.
      dict["value"] : a threshold value of the attribute.
      dict["left"] : first part of data separated to the left.
      dict["right"] : second part of data separated to the right.
    """
    best_disc_p = -np.inf
    best = {}
    for attr_index in range(len(data) - 1):
      data_attr = data[attr_index]
      data_attr_length = len(data_attr)
      for value in np.unique(data_attr):
        left, right = self.separate_by_value(data, attr_index, value)
        left_entropy = self.get_entropy(left)
        right_entropy = self.get_entropy(right)
        disc_p = self.get_entropy(data) - (
          (len(left[-1]) / data_attr_length) * left_entropy + (len(right[-1]) / data_attr_length) * right_entropy)

        if (disc_p > best_disc_p):
          best["attr_index"] = attr_index
          best["attr_name"] = self.attr_names[attr_index] if self.attr_names else None
          best["value"] = value
          best["left"] = left
          best["right"] = right
          best_disc_p = disc_p

    return best

  def isPure(self, data:np.ndarray):
    """
    Checks if the node is pure node.

    Args:
      data (np.ndarray): 2D input matrix with rows as attributes and columns as instances.

    Returns (bool): True if the node is pure, False otherwise.
    """
    if self.get_entropy(data) == 0:
      return True
    return False

  def fit(self):
    """
    Trains the model.
    """
    self.root_node = self.fill_node(self.data, 0)

  def test(self, X_test:np.ndarray, Y_test:np.ndarray, verbose:bool=True):
    """
    Tests the model on a given data.
    Args:
      X_test (np.ndarray): 2D matrix with rows as attributes and columns as instances.
      Y_test (np.ndarray): 1D array with labels.
      verbose (bool): Verbose mode. Defaults to True.

    Returns (np.ndarray): Confusion matrix.
    """
    tp = tn = fp = fn = 0
    Yhat = self.predict(X_test)
    Y_test = Y_test.flatten()
    comparison = (Y_test == Yhat)
    for y, _comp in zip(Y_test, comparison):
      if _comp:
        if y == 1:
          tp += 1
        else:
          tn += 1
      else:
        if y == 1:
          fp += 1
        else:
          fn += 1
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if verbose:
      print("TN:{} FN:{} FP:{} TP{}".format(tn, fn, fp, tp))
    return np.array([[tn, fn], [fp, tp]])

  def predict(self, X_test:np.ndarray):
    """
    Predicts the output on a given data.
    Args:
      X_test (np.ndarray): 2D matrix with rows as attributes and columns as instances.

    Returns (np.ndarray): 1D array of predicted values.

    """
    a = []
    for i in range(len(X_test[-1])):
      node = self.root_node
      a.append(self.traverse(X_test[:, i], node))
    return np.array(a)

  def traverse(self, x:np.ndarray, node:Node):
    """
    Traverses the tree.
    Args:
      x (np.ndarray): 2D matrix with rows as attributes and columns as instances.
      node (Node): Node of a tree.

    Returns (int): The output of a node (when it is a leaf).
    """
    if node.isLeaf():
      return node.get_output()
    if x[node.get_attr_index()] <= node.get_value():
      return self.traverse(x, node.get_left_child())
    else:
      return self.traverse(x, node.get_right_child())

  def fill_node(self, data:np.ndarray, i:int):
    """
    Fills the node with the most suitable values for the decision tree.

    Args:
      data (np.ndarray): 2D matrix with rows as attributes and columns as instances.
      i (int): Level of a node in a tree.

    Returns (Node): Filled node.
    """
    print("Level", i, "| Data length", len(data[-1]))
    if self.isPure(data):
      node = Node()
      node.set_output(data[-1][0])
      print("It is clean with output {}".format(data[-1][0]))
      return node
    elif i == 4:
      node = Node()
      a = np.bincount(data[-1].astype(int))
      node.set_output(np.bincount(data[-1].astype(int)).argmax())
      print("Index is equal to {} which is maximum depth. I took most probable output {}. All(0, 1) are {}".format(i, node.get_output(), a))
      return node

    best = self.get_best_decision(data)
    node = Node(best["attr_index"], best["value"])
    # print("Best left:{}\nBest right{}".format(best["left"], best["right"]))
    print("Separated on attribute index {}(Called: {})  with value of {}".format(best["attr_index"], best["attr_name"], best["value"]))
    print("\nGoing to left node")
    node.set_left_child(self.fill_node(best["left"], i+1))
    print("\nGoing to right node")
    node.set_right_child(self.fill_node(best["right"], i+1))
    return node
