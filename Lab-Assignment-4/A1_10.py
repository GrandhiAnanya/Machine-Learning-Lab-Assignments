import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#chatgpt used
def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )

    return df

df = load_data()
income = df["Income"].dropna()
hist, bin_edges = np.histogram(income, bins=10)
plt.figure(figsize=(8,5))
plt.bar(bin_edges[:-1],
        hist,
        width=np.diff(bin_edges),
        align='edge',
        edgecolor='black')

plt.title("Histogram of Income")
plt.xlabel("Income")
plt.ylabel("Frequency")

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()