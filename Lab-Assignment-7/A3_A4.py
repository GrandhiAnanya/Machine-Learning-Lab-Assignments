import pandas as pd
import numpy as np
import math


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

def best_feature(y,df):
    ig_values = {}
    X=df.drop(columns=['person_id', 'image_name'])
    for feature in X:
        ig_values[feature] = information_gain(df,feature,y)
    print(ig_values)
    root = max(ig_values, key=ig_values.get)

    return root




df=load_data()
X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']

bins=4
for i in X:
        df[i] = equal_width_binning(df[i], bins)

tree_root = best_feature(y,df)
print("root = ",tree_root)
