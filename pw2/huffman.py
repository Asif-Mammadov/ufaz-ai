from html import entities
from node import Node
from entropy import Entropy

class Huffman:
  def __init__(self, entropy):
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

    self.root = nodes[0]
    self.encoded = None
  
  def bfs_print(self):
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

  def encode_of(self, chr):
    if not chr in self.encoded:
      raise "Error"
    else:
      return self.encoded[chr]

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
  