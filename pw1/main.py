import pandas as pd
import numpy as np
import copy
import utils
dataset = pd.read_csv("Iris.csv", delimiter=',')


def separate_in_attr_groups(dataset, N_groups=3):
    attributes = dataset.columns[1:-1]
    groups_by_attr = {}
    N = dataset.shape[0]
    for attribute in attributes:
        sorted_data = dataset.sort_values(attribute).loc[:, [attribute, 'Species']]
        groups = []
        for i in range(N_groups):
            groups.append(sorted_data[int(i*N/3):int((i+1)*N/3)])
        groups_by_attr[attribute] = groups
    return groups_by_attr


def count_group(group_df, species=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']):
    counted_df = group_df.groupby(['Species']).count()
    for specimen in species:
        if not counted_df.index.isin([specimen]).any():
            counted_df.loc[specimen, :] = 0
            counted_df = counted_df.astype(np.int64)
    counted_df.reset_index(level=0,inplace=True)
    return counted_df

def count_attr_groups(attr_groups):
    groups_by_species_counted = copy.deepcopy(attr_groups)
    for attr in groups_by_species_counted.keys():
        for group in range(len(groups_by_species_counted[attr])):
            groups_by_species_counted[attr][group] = count_group(groups_by_species_counted[attr][group])
    return groups_by_species_counted


attr_groups = separate_in_attr_groups(dataset)
attr_groups_counted = count_attr_groups(attr_groups)
print(attr_groups_counted['SepalLengthCm'][0])

def print_attr_groups_counted(groups):
    
    for attr in groups.keys():
        print("Attribute: ", attr)
        for group in range(len(groups[attr])):
            print("\nGroup {}".format(group))
            print(groups[attr][group], end='\n')
        print("-" * 100)
print_attr_groups_counted(attr_groups_counted)


print("Dataset entropy : {}".format(utils.entropy(dataset, dataset['Species'].unique())))

def entropy_group(group_counted_df, distinctionColumn='Species', N=50):
    s = 0
    for specimen in group_counted_df[distinctionColumn]:
        n_specimen = group_counted_df.set_index(distinctionColumn).loc[specimen, :][0]
        if n_specimen != 0:
            s += (n_specimen / N) * np.log2(n_specimen / N)
    s *= -1
    return s

def get_attr_groups_entropy(attr_groups_counted):
  attr_groups_entropy = {}
  for attr in attr_groups_counted:
      groups = []
      for group in range(len(attr_groups_counted[attr])):
          groups.append(entropy_group(attr_groups_counted[attr][group]))
      attr_groups_entropy[attr] = groups
  return attr_groups_entropy
# print(entropy_group(attr_groups_counted['SepalLengthCm'][0]))
attr_groups_entropy = get_attr_groups_entropy(attr_groups_counted)

def print_attr_groups_entropy(attr_groups_entropy):
  for attr in attr_groups_entropy:
      print(attr)
      for group in range(len(attr_groups_entropy[attr])):
          print("Group {} : {}".format(group, attr_groups_entropy[attr][group]))
      print("-" * 50)


def get_prob(group_counted_df, specimen, distinctionColumn='Species'):
  n_specimen = group_counted_df.set_index(distinctionColumn).loc[specimen, :][0]
  return n_specimen / group_counted_df.iloc[:, 1].sum()

def get_group_size(group):
  return group.iloc[:, 1].sum()

def get_discriminative_power(dataset, single_attr_groups_counted, distinctionColumn='Species'):
  dataset_entropy = utils.entropy(dataset, dataset[distinctionColumn].unique())
  N = 0
  single_attr_groups_enropy = get_attr_groups_entropy(single_attr_groups_counted)
  attr_name = list(single_attr_groups_enropy.keys())[0]
  s = dataset_entropy
  for i_group in range(len(single_attr_groups_counted[attr_name])):
    N += get_group_size(single_attr_groups_counted[attr_name][i_group])
  for i_group in range(len(single_attr_groups_counted[attr_name])):
    s -= (get_group_size(single_attr_groups_counted[attr_name][i_group]) / N ) * single_attr_groups_enropy[attr_name][i_group]
  return s

for attr_name, attr_value in attr_groups_counted.items():
  print(attr_name, ":", get_discriminative_power(dataset, {attr_name : attr_value}))
# print(get_discriminative_power(dataset, {'SepalLengthCm': attr_groups_counted['SepalLengthCm']}))



import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(dataset.iloc[:, 1:], hue='Species', size=2)
plt.show()