from mlp.neuralNetwork import NeuralNetwork
from mlp.cost import MSE, CrossEntropy
from mlp.activation import Sigmoid, Relu, Tanh
from mlp import utils
import pandas as pd
import numpy as np
import metrics
from mlp.normalization import z_score, linear_scaling, linear_scaling_max



def main():
    data = pd.read_csv('heart.csv')
    attr = data.columns[:-1]
    X = utils.loadAttributes(data, attr)
    Y = data['target'].to_numpy().reshape((1, -1))
    X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y, test_portion=0.3)

    nn = NeuralNetwork(X_train, Y_train, X_test, Y_test, lr=0.01, cost=MSE, batch_size=4, normalization=linear_scaling_max)
    nn.add_hidden_layer(5, Tanh)
    nn.info()
    nn.train(verbose=True)
    print("-"*20, "Results", "-"*20)
    conf_matrix = nn.testPrediction(X_test, Y_test, verbose=True)
    print("Precision:", metrics.get_precision(conf_matrix))
    print("Accuracy:", metrics.get_accuracy(conf_matrix))
    print("Sensitivity:", metrics.get_sensitivity(conf_matrix))
    print("Specificity:", metrics.get_specificity(conf_matrix))
    nn.plot_stats()

if __name__ == "__main__":
    main()