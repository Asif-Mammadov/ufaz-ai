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

  X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y, test_portion=0.3)

  nn = NeuralNetwork(X_train, Y_train, X_test, Y_test, lr=0.1, cost=MSE, batch_size=20)
  nn.add_hidden_layer(5, Sigmoid)
  nn.info()
  nn.train(n_epochs=50, verbose=True)
  nn.plot_stats()

if __name__ == '__main__':
  main()