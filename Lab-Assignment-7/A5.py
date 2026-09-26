import pandas as pd
import numpy as np
import math
from graphviz import Digraph


def load_data():
    df=pd.read_csv("features.csv")
    return df


def entropy(df,y):
    class_counts = y.value_counts()
    total = len(df)
    sum = 0
    for i in class_counts.index:
        p_i = class_counts[i] / total
        sum += p_i * math.log(p_i, 2)
    return (-1)*sum



def equal_width_binning(column, bins):
    min = column.min()
    max = column.max()
    width = (max - min) / bins
    binned_values = []
    for i in column:
        if i == max:
            bin_number = bins - 1
        else:
            bin_number = int((i - min) / width)
        binned_values.append(bin_number)
    return binned_values


def information_gain(df,feature,y):
    total_entropy = entropy(df,y)
    weighted_entropy = 0
    total_samples = len(df)
    values = df[feature].unique()
    for value in values:
        subset = df[df[feature] == value]
        subset_probability = len(subset) / total_samples
        subset_entropy = entropy(subset,subset['person_id'])
        weighted_entropy += subset_probability * subset_entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

def best_feature(X,y,df):
    ig_values = {}
    for feature in X:
        ig_values[feature] = information_gain(df,feature,y)
    root = max(ig_values, key=ig_values.get)

    return root

def build_tree(df, features, y):
    if len(y.unique()) == 1:
        return y.iloc[0]

    if len(features) == 0:
        return y.value_counts().idxmax()
    feature = best_feature(features, y, df)
    
    tree = {
        feature: {}
    }

    for value in df[feature].unique():
        subset = df[df[feature] == value]
        remaining_features = features.drop(feature)
        tree[feature][value] = build_tree(subset,remaining_features,subset['person_id'])

    return tree



df=load_data()
X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']

bins=4
for i in X:
        df[i] = equal_width_binning(df[i], bins)

tree_root = build_tree(df, X.columns,y)
print(tree_root)


