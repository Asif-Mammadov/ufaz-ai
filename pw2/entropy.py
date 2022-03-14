import copy
import numpy as np
class Entropy:
  def __init__(self, string:str):
    """Initialize Entropy object.

    Args:
        string (str): String to find entropy.
    """
    self.count = self.__decompose(string)
    self.length = len(string)
    self.freq = self.__caclulate_freq()

  def __decompose(self, string:str) -> dict:
    """Decompose a string into a number of occurences of each letter.

    Args:
        string (str): String to decompose.

    Returns:
        dict: with key - a letter, value - number of occurences.

        Example: "hello" 
        dict = {'h' : 1, 'e': 1, 'l': 2, 'o': 1}
    """
    tmp_dict = {}
    for nextchar in string:
      if not nextchar in tmp_dict:
        tmp_dict[nextchar] = 1
      else:
        tmp_dict[nextchar] += 1
    return tmp_dict

  def __caclulate_freq(self) -> dict:
    """Finds frequency of each letter in a range of [0, 1].

    Returns:
        dict: representing freq of each letter.
    """
    freq = copy.deepcopy(self.count)
    for c in freq:
      freq[c] /= self.length
    return freq

  def compute_entropy(self) -> float:
    """Finds entropy of a given `string`.

    Returns:
        float: Entropy value.
    """
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
