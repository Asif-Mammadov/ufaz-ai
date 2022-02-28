from cmath import exp


class Node:
  def __init__(self, data, freq=None, left_child=None, right_child=None):
    self.data = data
    self.freq = freq
    # self.encoded = encoded
    self.left_child = left_child
    self.right_child = right_child

  def print_node(self):
    print("|", self.data, self.freq, "|", end=" ") 
  def display(self):
    leftdata = self.left_child.data if self.left_child != None else None
    rightdata = self.right_child.data if self.right_child != None else None
    print("Data: {}\nFreq: {}\nLeft Child: {}\nRight Child: {}".format(self.data, self.freq, leftdata, rightdata))

  def get_data(self):
    return self.data
  
  def set_data(self, data):
    self.data = data
  
  def get_freq(self):
    return self.freq
  
  def set_freq(self, freq):
    self.freq = freq
  
  def get_left_child(self):
    return self.left_child
  
  def set_left_child(self, left_child):
    self.left_child = left_child

  def get_right_child(self):
    return self.right_child
  
  def set_right_child(self, right_child):
    self.right_child = right_child


  def display_propagate(self, node):
    print(node.data, node.freq)
    if (node.left_child):
      self.display_propagate(node.left_child)
    if (node.right_child):
      self.display_propagate(node.right_child)
  
  def print_children(self):
    if (self.left_child):
      self.left_child.print_node()
    if (self.right_child):
      self.right_child.print_node()

  def get_children(self):
    return [self.left_child, self.right_child]

     
  def display_propagate_inline(self):
    self.print_children()
    # if self.get_left_child():
    #   self.get_left_child().display_propagate_inline()
    # if self.get_right_child():
    #   self.get_right_child().display_propagate_inline()

  # def display_bfs(self, node):
  #   q = []

  # def find(self, number):
  #   if number == self.freq:
  #     print("Done")
  #     return number
  #   elif number < self.freq:
  #     if self.get_left_child():
  #       self.get_left_child().find(number)
  #     else:
  #       print("no such number")
  #       return -1
  #   elif number > self.freq:
  #     if self.get_right_child():
  #       self.get_right_child().find(number)
  #     else:
  #       print("No such number")
  #       return -1

  def isLeaf(self):
    if not self.get_right_child() and not self.get_left_child():
      return True
    return False

  def generate_dict(self, code="", dict={}):
    if self.isLeaf():
      dict[self.data] = code
      return
    if self.get_left_child():
      self.get_left_child().generate_dict(code + "0", dict)
    if self.get_right_child():
      self.get_right_child().generate_dict(code + "1", dict)
    



