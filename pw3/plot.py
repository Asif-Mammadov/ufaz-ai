from cProfile import label
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('Iris.csv')
def plot(ax, df, x_attr, y_attr, class_attr, title=None, axis_labels=None):
  color =  ['blue', 'red', 'green']
  for i, _class_attr in enumerate(df[class_attr].unique()):
    filter = (df[class_attr] == _class_attr)
    ax.scatter(df[x_attr][filter], df[y_attr][filter], c=color[i], s=100, alpha=0.4, marker='o', label=_class_attr)
  if axis_labels:
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
  else:
    ax.set_xlabel(x_attr)
    ax.set_ylabel(y_attr)
  if title:
    ax.set_title(title)
  # ax.legend(loc='lower left', ncol=len(df.columns))
  # plt.show()


# print(df.columns[1:-1])

# fig, ax = plt.subplots(nrows=4, ncols=4)
# print(ax)
# for i, column1 in enumerate(df.columns[1:-1]):
#   for j, column2 in enumerate(df.columns[1:-1]):
#     plot(ax[i][j], df, column1, column2, 'Species')
# plt.legend()
# fig.show()
# plt.show()  