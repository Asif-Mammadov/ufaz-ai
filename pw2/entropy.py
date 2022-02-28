import copy
import numpy as np
class Entropy:
  def __init__(self, string):
    self.count = self.__decompose(string)
    self.length = len(string)
    self.freq = self.__caclulate_freq()

  def __decompose(self, string):
    tmp_dict = {}
    for nextchar in string:
      if not nextchar in tmp_dict:
        tmp_dict[nextchar] = 1
      else:
        tmp_dict[nextchar] += 1
    return tmp_dict

  def __caclulate_freq(self):
    freq = copy.deepcopy(self.count)
    for c in freq:
      freq[c] /= self.length
    return freq

  def compute_entropy(self):
    h = 0
    for _freq in self.freq.values():
      h += _freq * np.log2(_freq)
    h *= -1
    return h

  def get_count(self):
    return self.count

  def set_count(self, count):
    self.count = count
  
  def get_length(self):
    return self.length
  
  def set_length(self, length):
    self.length = length

  def get_freq(self):
    return self.freq
  
  def set_freq(self, freq):
    self.freq = freq
