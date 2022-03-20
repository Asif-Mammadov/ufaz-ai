import numpy as np
import pandas as pd
import utils
from activation import Relu, Softmax

df = pd.read_csv('pw4/Iris.csv')
X = utils.loadAttributes(df, ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
Y = utils.createOneHot(df, "Species")
# training_data = np.append(X, Y)
# print()
X_train, Y_train, X_test, Y_test = utils.split_train_test(X, Y, test_portion=0.2)

print("Y_train", Y_train.T)
print("Y_test", Y_test.T)

# print(np.dot(a, b.T))
parameters = utils.trainingEpoch(X_train, Y_train, batch_size=40)


X = X_test[:, 2].reshape((-1, 1))
# print(Y_test[:, 2])
def predict(parameters, X):
  W, B = parameters["W"], parameters["B"]
  activations = parameters["activations"]
  a = X.copy()
  for j in range(len(W)):
    z = np.dot(W[j], a) + B[j]
    a = activations[j].calc(z)
  
  for i in range(a.shape[1]):
    index = a[:, i].argmax()
    a[:, i] = np.zeros(a.shape[0])
    a[:, i][index] = 1
  return a

def testPrediction(X_test, Y_test, parameters):
  t = f = 0
  Yhat = predict(parameters, X_test)
  print(Yhat, Y_test)
  for i in range(Yhat.shape[1]):
    if Yhat[:, i].argmax() == Y_test[:, i].argmax():
      t += 1
    else:
      f += 1
  print("Correct: {}\nFalse: {}\nAccuracy:{}".format(t, f, t / (t + f)))
  return t, f

testPrediction(X_train, Y_train, parameters)