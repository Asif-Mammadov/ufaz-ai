import numpy as np
import pandas as pd
def sort_species(dataset, specimen):
    return dataset[dataset['Species'] == specimen]

def entropy(dataset, instances):
    s = 0
    N = dataset.shape[0]
    for specimen in instances:
        n_specimen = sort_species(dataset, specimen).shape[0]
        s +=  n_specimen / N * np.log2(n_specimen / N)
    s *= -1
    return s

def entropy_group(group_counted_df, distinctionColumn='Species', N=50):
    s = 0
    for specimen in group_counted_df[distinctionColumn]:
        n_specimen = group_counted_df.set_index(distinctionColumn).loc[specimen, :][0]
        if n_specimen != 0:
            s += (n_specimen / N) * np.log2(n_specimen / N)
    s *= -1
    return s

def exclude(left_df, right_df):
    return pd.merge(left_df,right_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)