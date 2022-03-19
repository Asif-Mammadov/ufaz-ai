import numpy as np
import pandas as pd
import utils
import activation

df = pd.read_csv('Iris.csv')
X = utils.loadAttributes(df, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
Y = utils.createOneHot(df, "Species")
# training_data = np.append(X, Y)
# print()
X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y, test_portion=0.2)
print(X_train.shape, Y_train.shape)
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((2, 6))
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape((2, 6))
utils.trainingEpoch(X_train, Y_train, 1)
