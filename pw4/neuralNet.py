import numpy as np
import pandas as pd
from utils import split_train_test

df = pd.read_csv('Iris.csv')

x = df.iloc[:, 1:-1]
y = df['Species']
X = x.to_numpy().T
Y = np.array([y == "Iris-setosa", y == "Iris-versicolor", y == "Iris-virginica"]).astype(int)



X_train, Y_train, X_test, Y_test = split_train_test(X, Y, test_portion=0.2)

# print(X_train.shape)

# x1 = X_train[:, 0].reshape(4, 1) 

def relu(Z):
  return Z * (Z > 0)

def softmax(Z):
  return np.exp(Z)/np.sum(np.exp(Z))

weights = []
biases = []
A = X_train
n_input = A.shape[0]
layers = [5, 5, 3, 3]
for i, n_nodes in enumerate(layers):
  W = np.random.random((n_nodes, n_input))
  b = np.zeros((n_nodes, 1))
  weights.append(W)
  biases.append(b)
  Z = np.dot(W, A)
  if i == len(layers) -1:
    A = softmax(Z)
  else:
    A = relu(Z)
  n_input = A.shape[0]
  print(A.shape)

print(weights[-1])
print("A shape", A.shape)
print("A[:, 0]", A[:, 0])

# print("Y_train shape", Y_train.shape)