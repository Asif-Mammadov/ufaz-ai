class Node:
  def __init__(self, attr_index:int=None, value=None):
    """

    Args:
      attr_index (int): Index of an attribute in a dataset.
      value (float): Threshold value of a node.
    """
    self.attr_index = attr_index
    self.value = value
    self.left_child = None
    self.right_child = None
    # Output value (prediction) of a node. Available only if the node is leaf node.
    self.output = None

  def isLeaf(self):
    """

    Returns: True if node is leaf node, False otherwise.

    """
    if self.get_left_child() is None and self.get_right_child() is None:
      return True
    return False

  def get_attr_index(self):
    return self.attr_index

  def set_attr_index(self, attr_index):
    self.attr_index = attr_index

  def get_value(self):
    return self.value

  def set_value(self, value):
    self.value = value

  def get_left_child(self):
    return self.left_child

  def set_left_child(self, left_child):
    self.left_child = left_child

  def get_right_child(self):
    return self.right_child

  def set_right_child(self, right_child):
    self.right_child = right_child

  def get_output(self):
    return self.output

  def set_output(self, output):
    self.output = output
