import copy
import utils
import numpy as np
class Decision_Node:
  def __init__(self, dataset, N_groups=3, distinctionColumn='Species'):
    self.dataset = dataset
    self.N_groups = N_groups
    self.distinctionColumn = distinctionColumn
    self.attr_groups = self.separate_in_attr_groups()
    self.attr_groups_counted = self.count_attr_groups()
    self.dataset_entropy = utils.entropy(self.dataset, self.dataset[distinctionColumn].unique())
    self.attr_groups_entropy = self.find_attr_groups_entropy()

  def separate_in_attr_groups(self):
    dataset = self.get_dataset()
    attributes = dataset.columns[1:-1]
    groups_by_attr = {}
    N = dataset.shape[0]
    for attribute in attributes:
        sorted_data = dataset.sort_values(attribute).loc[:, [attribute, self.distinctionColumn]]
        groups = []
        for i in range(self.N_groups):
            groups.append(sorted_data[int(i*N/3):int((i+1)*N/3)])
        groups_by_attr[attribute] = groups
    return groups_by_attr

  def count_group(self, group_df, species=None):
      if not species:
        species=self.dataset[self.distinctionColumn].unique()
      counted_df = group_df.groupby(['Species']).count()
      for specimen in species:
          if not counted_df.index.isin([specimen]).any():
              counted_df.loc[specimen, :] = 0
              counted_df = counted_df.astype(np.int64)
      counted_df.reset_index(level=0,inplace=True)
      return counted_df

  def count_attr_groups(self):
      attr_groups = self.get_attr_groups()
      groups_by_species_counted = copy.deepcopy(attr_groups)
      for attr in groups_by_species_counted.keys():
          for group in range(len(groups_by_species_counted[attr])):
              groups_by_species_counted[attr][group] = self.count_group(groups_by_species_counted[attr][group])
      return groups_by_species_counted

  def print_attr_groups_counted(self):
      groups = self.get_attr_groups_counted()
      for attr in groups.keys():
          print("Attribute: ", attr)
          for group in range(len(groups[attr])):
              print("\nGroup {}:".format(group))
              print(groups[attr][group], end='\n')
          print("-" * 100)

  def find_attr_groups_entropy(self):
    attr_groups_counted = self.attr_groups_counted
    attr_groups_entropy = {}
    for attr in attr_groups_counted:
        groups = []
        for group in range(len(attr_groups_counted[attr])):
            groups.append(utils.entropy_group(attr_groups_counted[attr][group]))
        attr_groups_entropy[attr] = groups
    return attr_groups_entropy

  def print_attr_groups_entropy(self):
    attr_groups_entropy = self.attr_groups_entropy
    for attr in attr_groups_entropy:
        print(attr)
        for group in range(len(attr_groups_entropy[attr])):
            print("Group {} : {}".format(group, attr_groups_entropy[attr][group]))
        print("-" * 50)

  def get_group_size(self, group):
    return group.iloc[:, 1].sum()

  def get_discriminative_power(self, columnName, distinctionColumn='Species'):
    dataset = self.dataset
    single_attr_groups_counted = {columnName: self.get_attr_groups_counted()[columnName]}
    dataset_entropy = utils.entropy(dataset, dataset[distinctionColumn].unique())
    single_attr_groups_enropy = self.get_attr_groups_entropy()
    N = 0
    s = dataset_entropy
    for i_group in range(len(single_attr_groups_counted[columnName])):
      N += self.get_group_size(single_attr_groups_counted[columnName][i_group])
    for i_group in range(len(single_attr_groups_counted[columnName])):
      s -= (self.get_group_size(single_attr_groups_counted[columnName][i_group]) / N ) * single_attr_groups_enropy[columnName][i_group]
    return s
  def print_discriminative_powers(self):
    for attr_name in self.get_attr_groups_counted():
        print(attr_name, ":", self.get_discriminative_power(attr_name))

  def get_dataset(self):
    return self.dataset 
  def set_dataset(self, dataset):
    self.dataset = dataset
  
  def get_N_groups(self):
    return self.N_groups
  def set_N_groups(self, N_groups):
    self.N_groups = N_groups

  def get_attr_groups(self):
    return self.attr_groups
  def set_attr_groups(self, attr_groups):
    self.attr_groups = attr_groups

  def get_attr_groups_counted(self):
    return self.attr_groups_counted
  def set_attr_groups_counted(self, attr_groups_counted):
    self.attr_groups_counted = attr_groups_counted
  
  def get_dataset_entropy(self):
    return self.dataset_entropy
  def set_dataset_entropy(self, dataset_entropy):
    self.dataset_entropy = dataset_entropy
  
  def get_attr_groups_entropy(self):
    return self.attr_groups_entropy
  def set_attr_groups_entropy(self, attr_groups_entropy):
    self.attr_groups_entropy = attr_groups_entropy