import numpy as np
import pandas as pd
from dt.utils import split_train_test
from dt.decision_tree import DecisionTree




def main():
  data = pd.read_csv("heart.csv")
  X = data.iloc[:, :-1].to_numpy().T
  Y = data.iloc[:, -1].to_numpy().T.reshape(1, -1)
  X_train, Y_train, X_test, Y_test = split_train_test(X, Y, test_portion=0.2)
  dt = DecisionTree(X_train, Y_train, max_depth=4, attr_names=data.columns[:-1])
  dt.fit()
  confusion_matrix, accuracy = dt.test_accuracy(X_test, Y_test)

if __name__ == "__main__":
  main()