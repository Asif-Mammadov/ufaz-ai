from cmath import exp


class Node:
  def __init__(self, data:str, freq:int=None, left_child=None, right_child=None):
    """Node initializer. 

    Args:
        data (str): value saved in a node.
        freq (float, optional): number of occurences (children nodes included). Defaults to None.
        left_child (Node, optional): left child of a node. Defaults to None.
        right_child (Node, optional): right child of a node. Defaults to None.
    """
    self.data = data
    self.freq = freq
    # self.encoded = encoded
    self.left_child = left_child
    self.right_child = right_child

  def generate_dict(self, code:str="", dict:dict={}):
    """Generates dictionary of encoded values according to Huffman coding.

    Args:
        code (str, optional): Code value. Defaults to "".
        dict (dict, optional): Dictionary to save. Defaults to {}.
    """
    if self.isLeaf():
      dict[self.data] = code
      return
    if self.get_left_child():
      self.get_left_child().generate_dict(code + "0", dict)
    if self.get_right_child():
      self.get_right_child().generate_dict(code + "1", dict)

  def isLeaf(self) -> bool:
    """Checks if the a node is a leaf.

    Returns:
        bool: True if a leaf, otherwise False.
    """
    if not self.get_right_child() and not self.get_left_child():
      return True
    return False

  def info(self):
    """Gives short information about the node: data and frequency.
    """
    print("|", self.data, self.freq, "|", end=" ") 

  def info_ext(self):
    """Gives extensive information about the node: data, frequency, left and right children.
    """
    leftdata = self.left_child.data if self.left_child != None else None
    rightdata = self.right_child.data if self.right_child != None else None
    print("Data: {}\nFreq: {}\nLeft Child: {}\nRight Child: {}".format(self.data, self.freq, leftdata, rightdata))

  def display_propagate(self, node):
    """Prints all the descendants of a given node.

    Args:
        node (Node): starting node. 
    """
    print(node.data, node.freq)
    if (node.left_child):
      self.display_propagate(node.left_child)
    if (node.right_child):
      self.display_propagate(node.right_child)
  
  def info_children(self):
    """Get short info about children of a node.
    """    
    if (self.left_child):
      self.left_child.info()
    if (self.right_child):
      self.right_child.info()

  def get_children(self):
    return [self.left_child, self.right_child]

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


    



