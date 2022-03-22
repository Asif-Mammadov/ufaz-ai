import numpy as np
from neuralNetwork import NeuralNetwork
from activation import Relu
import utils


def main():
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    Y = np.array([[0, 1, 1, 0]])

    X = np.tile(X, 600)
    Y = np.tile(Y, 600)
    X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y)

    # return
    nn = NeuralNetwork(X, Y, lr=0.1, batch_size=50)
    # nn.add_hidden_layer(5, Relu)
    nn.info()
    nn.training_epoch(n_epochs=5, verbose=False)
    nn.testPrediction(X_test, Y_test)
    nn.plot_costs()


if __name__ == '__main__':
    main()