import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import math

def load_data():
    df = pd.read_csv("features.csv")
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


df = load_data()
X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']

ig_values = {}
for feature in X:
    ig_values[feature] = information_gain(df,feature,y)

top_2_features = sorted(
    ig_values,
    key=ig_values.get,
    reverse=True
)[:2]

print("Selected features:", top_2_features)
X_2 = X[top_2_features]

clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_2, y)
fig, ax = plt.subplots(figsize=(10, 7))


disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X_2,
    response_method="predict",
    ax=ax,
    xlabel=top_2_features[0],
    ylabel=top_2_features[1],
    alpha=0.5
)


scatter = disp.ax_.scatter(
    X_2.iloc[:, 0],
    X_2.iloc[:, 1],
    c=pd.factorize(y)[0],
    edgecolor="black",
    s=30
)


plt.title("Decision Surface of Decision Tree")
plt.tight_layout()
plt.show()