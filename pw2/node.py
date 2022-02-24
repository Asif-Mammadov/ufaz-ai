class Node:
  def __init__(self, data, freq=None, left_child=None, right_child=None):
    self.data = data
    self.freq = freq
    self.left_child = left_child
    self.right_child = right_child
  
  def display(self):
    leftdata = self.left.data if self.left != None else None
    rightdata = self.right.data if self.right != None else None
    print("Data: {}\nFreq: {}\nLeft Child: {}\nRight Child: {}".format(self.data, self.freq, self.left_child, self.right_child))

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

