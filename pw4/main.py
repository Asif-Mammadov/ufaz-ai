import numpy as np
import pandas as pd
from mlp import utils
from mlp.neuralNetwork import NeuralNetwork
from mlp.activation import Sigmoid, Relu, Softmax
from mlp.cost import MSE, CrossEntropy
from mlp.normalization import linear_scaling_max, linear_scaling
from mlp.utils import get_accuracy


def main():
  df = pd.read_csv('Iris.csv')
  X = utils.loadAttributes(df, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
  Y = utils.createOneHot(df, "Species")

  X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y, test_portion=0.3)
  nn = NeuralNetwork(X_train, Y_train, lr=0.05, cost=MSE, batch_size=4, normalization=None)
  nn.add_hidden_layer(5, Relu)
  nn.info()
  nn.train(n_epochs=50,verbose=True)
  print("-" * 20, "Recap", "-" * 20)
  nn.info()
  conf_matrix = nn.testPrediction(X_test, Y_test, verbose=False)
  print("-" * 20, "Test set prediction result", "-" * 20)
  print("Confusion matrix:\n {}".format(conf_matrix))
  print("Accuracy: {}".format(get_accuracy(conf_matrix)))
  nn.plot_stats()

if __name__ == '__main__':
  main()