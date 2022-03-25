import numpy as np
def sum_diag(matrix):
  return sum([matrix[i][i] for i in range(len(matrix))])

def get_accuracy(conf_matrix):
  return np.trace(conf_matrix) / conf_matrix.sum()

def get_precision(conf_matrix):
  return conf_matrix[1][1] / sum(conf_matrix[1])

def get_specificity(conf_matrix):
  return conf_matrix[0][0] / conf_matrix.sum(axis=0)[0]

def get_sensitivity(conf_matrix):
  return conf_matrix[1][1] / conf_matrix.sum(axis=0)[1]