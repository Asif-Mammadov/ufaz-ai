from operator import imod

from plot import plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('Iris.csv')

# kmeans 1d

sl = df['SepalLengthCm']
# print(sl)

k = 3
init = np.array(np.random.randint(len(df), size=k))
# print(init)

c = init 
# print(c - 3)
# for row in df:
#   tmp_c = np.abs(c - row[0])
#   print(tmp_c)