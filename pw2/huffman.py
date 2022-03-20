from html import entities

from numpy import string_
from node import Node
from entropy import Entropy

class Huffman:
  def __init__(self, entropy:Entropy):
    """Initialize an object for Huffman coding.

    Args:
        entropy (Entropy): Entropy object.
    """
    nodes = []
    for key in entropy.count:
      nodes.append(Node(key, freq=entropy.count[key]))

    nodes.sort(key=lambda x: x.freq)

    while len(nodes) > 1:
      nodes.sort(key=lambda x: x.freq)
      least = nodes[:2]
      left, right = least
      newNode = Node(left.data + right.data, freq=left.freq + right.freq, left_child=left, right_child=right)
      nodes.append(newNode)
      nodes.remove(left)
      nodes.remove(right)

    self.string = entropy.string
    self.root = nodes[0]
    self.encoded = None
  
  def bfs_print(self):
    """Prints a Huffman tree in a visually appealing format.
    """
    root = self.root
    q = [] 
    explored = []
    explored.append(root)
    q.append(root)
    q.append(None)
    print("<" + str(root.freq) + " " + root.data + ">")
    while len(q) > 1:
      v = q.pop(0)
      if (v == None):
        q.append(None)
        print()
        continue
      for child in v.get_children():
        if child and not child in explored:
          explored.append(child)
          q.append(child)
          print("<" + str(child.freq) + " " + child.data + ">", end=" ")

  def encoding_of(self, chr:str)->str:
    """Finds an encoding of a given character in Huffman coding.

    Args:
        chr (str): character

    Returns:
        str: Encoding of a character.
    """    
    if not chr in self.encoded:
      raise ValueError("No such character in the dictionary.") 
    else:
      return self.encoded[chr]
  
  def to_encoding(self, string:str)->str:
    """Converts string to encoding

    Args:
        string (str): Human readable string.

    Returns:
        str: Encoded value.
    """
    encoded_str = ""
    for c in string:
      encoded_str += self.encoding_of(c)
    return encoded_str

  def to_string(self, encoded_str:str)->str:
    """Converts encoding to string

    Args:
        encoded_str (str): Encoded value.

    Returns:
        str: Decoded string.
    """
    new_string = ""
    i = 0
    while i < len(encoded_str):
      node = self.root
      while not node.isLeaf():
        if encoded_str[i] == "0":
          node = node.get_left_child()
        elif encoded_str[i] == "1":
          node = node.get_right_child()
        i += 1
      new_string += node.get_data()
    return new_string

  def get_encoded(self):
    if not self.encoded:
      self.encoded = {}
      self.root.generate_dict(dict=self.encoded)
    return self.encoded

  def set_encoded(self, dict):
    self.encoded = dict

  def get_root(self):
    return self.root
  
  def set_root(self, root):
    self.root = root
  