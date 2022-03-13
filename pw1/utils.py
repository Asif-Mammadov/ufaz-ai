import numpy as np
import pandas as pd
def sort_species(dataset:pd.DataFrame, specimen:str) -> pd.DataFrame:
    """Filters `dataset` according to `specimen`.

    Args:
        dataset (pd.DataFrame): Iris dataset.
        specimen (str): a unique class in a 'Species' column.

    Returns:
        pd.DataFrame: Filtered to a certain `specimen`.
    """
    return dataset[dataset['Species'] == specimen]

def entropy(dataset: pd.DataFrame, instances: list) -> float:
    """Calculates the entropy of a `dataset` with `instances`.

    Args:
        dataset (pd.DataFrame): Iris dataset.
        instances (list): unique classes in a dataset.

    Returns:
        float: Entropy value.
    """
    s = 0
    N = dataset.shape[0]
    for specimen in instances:
        n_specimen = sort_species(dataset, specimen).shape[0]
        s +=  n_specimen / N * np.log2(n_specimen / N)
    s *= -1
    return s

def get_group_size(group:pd.DataFrame) -> int:
    """Gets number of instances in a group.

    Args:
        group (pd.DataFrame): Consists of instances of classes. 

    Returns:
        int: Number of total instances in a group.
    """
    return group.iloc[:, 1].sum()

def entropy_group(group_counted_df: pd.DataFrame, distinctionColumn:str='Species') -> float:
    """Calculates the group entropy.

    Args:
        group_counted_df (pd.DataFrame): Group with counted instances according to a class (`distinctionColumn`)
        distinctionColumn (str, optional): Column to differentiate between classes in a dataset. Defaults to 'Species'.
        N (int, optional): Number of instances. Defaults to 50.

    Returns:
        float: _description_
    """
    s = 0
    N = get_group_size(group_counted_df)
    for specimen in group_counted_df[distinctionColumn]:
        n_specimen = group_counted_df.set_index(distinctionColumn).loc[specimen, :][0]
        if n_specimen != 0:
            s += (n_specimen / N) * np.log2(n_specimen / N)
    s *= -1
    return s

def exclude(left_df, right_df):
    return pd.merge(left_df,right_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)