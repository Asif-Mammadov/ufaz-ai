import numpy as np
from mlp.neuralNetwork import NeuralNetwork
from mlp.activation import Relu, Sigmoid
from mlp import utils


def main():
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    Y = np.array([[0, 1, 1, 0]])

    X = np.tile(X, 120)
    Y = np.tile(Y, 120)
    X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y)

    # return
    nn = NeuralNetwork(X_train, Y_train, X_test, Y_test, lr=0.1, batch_size=4)
    nn.add_hidden_layer(5, Sigmoid)
    nn.info()
    nn.train(n_epochs=4, verbose=True)
    nn.plot_stats()


if __name__ == '__main__':
    main()