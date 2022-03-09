from pw1.decision_node import Decision_Tree
import pandas as pd

dataset = pd.read_csv("Iris.csv", delimiter=',')
dt = Decision_Tree(dataset)


dt.print_attr_groups_entropy()

for attr_name in dt.get_attr_groups_counted():
  print(attr_name, ":", dt.get_discriminative_power(attr_name))


new_dataset = dataset[dataset['PetalLengthCm'] > dataset.sort_values('PetalLengthCm')['PetalLengthCm'].reset_index(drop=True)[50]].reset_index(drop=True)
print(new_dataset.sort_values('PetalLengthCm'))
# print(new_dataset)
dt2 = Decision_Tree(new_dataset, N_groups=2)

for attr_name in dt2.get_attr_groups_counted():
  print(attr_name, ":", dt2.get_discriminative_power(attr_name))

# # dt2.print_attr_groups_entropy()
dt2.print_attr_groups_counted()

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(new_dataset.iloc[:, 1:], hue='Species', size=2)
plt.show()