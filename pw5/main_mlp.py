from mlp.neuralNetwork import NeuralNetwork
from mlp.cost import MSE, CrossEntropy
from mlp.activation import Sigmoid, Relu
from mlp import utils
import pandas as pd
import numpy as np



def main():
    data = pd.read_csv('heart.csv')
    attr = data.columns[:-1]
    X = utils.loadAttributes(data, attr)
    Y = data['target'].to_numpy().reshape((1, -1))

    tmp = np.concatenate([X, Y], axis=0)
    # return
    X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y, test_portion=0.3)

    nn = NeuralNetwork(X_train, Y_train, X_test, Y_test, lr=0.01, cost=MSE, batch_size=4, normalizeData=True)
    nn.add_hidden_layer(5, Relu)
    nn.info()
    nn.train(verbose=True)
    nn.plot_stats()

if __name__ == "__main__":
    main()