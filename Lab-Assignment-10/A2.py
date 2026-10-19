
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    df = pd.read_csv("features.csv")
    return df


df=load_data()
features = df.drop(columns=['person_id', 'image_name'])
correlation_matrix = features.corr(method='pearson')
print(correlation_matrix.round(2))

plt.figure(figsize=(16, 12))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    linewidths=0.5
)


plt.show()
