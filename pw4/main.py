import numpy as np
import pandas as pd
from mlp import utils
from mlp.neuralNetwork import NeuralNetwork
from mlp.activation import Sigmoid, Relu, Softmax
from mlp.cost import MSE, CrossEntropy
from mlp.normalization import linear_scaling_max


def main():
  df = pd.read_csv('Iris.csv')
  X = utils.loadAttributes(df, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
  Y = utils.createOneHot(df, "Species")

  X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y, test_portion=0.3)
  nn = NeuralNetwork(X_train, Y_train, X_test, Y_test, lr=0.05, cost=MSE, batch_size=4)
  nn.add_hidden_layer(5, Relu)
  nn.info()
  nn.train(n_epochs=50,verbose=True)
  nn.plot_stats()

if __name__ == '__main__':
  main()