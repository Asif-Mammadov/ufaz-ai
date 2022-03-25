import numpy as np
from dt.node import Node

class DecisionTree:
  def __init__(self, X_train, Y_train, max_depth=4, attr_names=None):
    self.X_train = X_train
    self.Y_train = Y_train
    self.data = np.concatenate([self.X_train, self.Y_train], axis=0)
    self.max_depth = 4
    self.root_node = None
    self.attr_names = list(attr_names)

  def separate_by_value(self, data, attr_index, value):
    less = np.where(data[attr_index] <= value)[0]
    not_less = np.where(data[attr_index] > value)[0]
    return data[:, less], data[:, not_less]

  def get_entropy(self, data):
    data_labels = data[-1]
    data_labels_length = len(data_labels)
    s = 0
    for unique_value in np.unique(data_labels):
      p = len(data_labels[data_labels == unique_value]) / data_labels_length
      s += p * np.log2(p)
    return -1 * s

  def get_best_decision(self, data):
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

  def isClean(self, data):
    if self.get_entropy(data) == 0:
      return True
    return False

  def fit(self):
    self.root_node = self.fill_node(self.data, 0)

  def test_accuracy(self, X_test, Y_test):
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
    print("TN:{} FN:{} FP:{} TP{}".format(tn, fn, fp, tp))
    print("Accuracy: {}".format(accuracy))
    return np.array([[tn, fn], [fp, tp]])

  def predict(self, X_test):
    a = []
    for i in range(len(X_test[-1])):
      node = self.root_node
      a.append(self.traverse(X_test[:, i], node))
    return np.array(a)

  def traverse(self, x, node):
    if node.isLeaf():
      return node.get_output()
    if x[node.get_attr_index()] <= node.get_value():
      return self.traverse(x, node.get_left_child())
    else:
      return self.traverse(x, node.get_right_child())

  def fill_node(self, data, i):
    print("Level", i, "| Data length", len(data[-1]))
    if self.isClean(data):
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
