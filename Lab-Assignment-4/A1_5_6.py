import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import minkowski

#chatgpt used
def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )

    return df

def preprocess(df):
        # Drop unnecessary columns
    df.drop(columns=["ID", "Z_CostContact", "Z_Revenue"], inplace=True)

    # Label Encoding
    le = LabelEncoder()
    df["Education"] = le.fit_transform(df["Education"])

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=["Marital_Status"], dtype=int)

    # Convert Date column into numerical values
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    df["Customer_Year"] = df["Dt_Customer"].dt.year
    df["Customer_Month"] = df["Dt_Customer"].dt.month
    df["Customer_Day"] = df["Dt_Customer"].dt.day
    df.drop(columns=["Dt_Customer"], inplace=True)
    return df

def minkowski_distance(x, y, p):
    return (np.sum(np.abs(x - y) ** p)) ** (1 / p)



df=load_data()
df=preprocess(df)
x = df.iloc[0].values
y = df.iloc[1].values


p_values = range(1, 11)
distances = []
scipy_distances = []

for p in p_values:
    d = minkowski_distance(x, y, p)
    scipy_distance = minkowski(x, y, p)
    distances.append(d)
    scipy_distances.append(scipy_distance)
    print(f"p = {p}")
    print(f"My Function    : {d:.6f}")
    print(f"SciPy Function : {scipy_distance:.6f}")
    print(f"Difference     : {abs(d - scipy_distance):.10f}\n")

plt.figure(figsize=(8,5))

plt.plot(p_values, distances, marker='o', label="My Function")
plt.plot(p_values, scipy_distances, marker='s', linestyle='--', label="SciPy")

plt.xlabel("Order (p)")
plt.ylabel("Distance")
plt.title("Comparison of Minkowski Distance")
plt.legend()
plt.grid(True)

plt.show()