import numpy as np
import pandas as pd
import utils
from neuralNetwork import NeuralNetwork
from activation import Sigmoid, Relu, Softmax
from cost import MSE, CrossEntropy


def main():
  df = pd.read_csv('Iris.csv')
  X = utils.loadAttributes(df, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
  Y = utils.createOneHot(df, "Species")

  X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y, test_portion=0.2)

  nn = NeuralNetwork(X_train, Y_train, lr=1, batch_size=20, cost=MSE, normalizeData=True)
  nn.add_hidden_layer(5, Relu)
  nn.info()
  nn.training_epoch(n_iter=5)
  nn.testPrediction(X_test, Y_test)
  nn.plot_costs()

if __name__ == '__main__':
  main()