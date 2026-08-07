import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

    df["Income"].fillna(df["Income"].mean(), inplace=True)

    return df

def calculate_mean(data):
    total = 0
    for value in data:
        total += value
    return total / len(data)

def calculate_variance(data):
    mean = calculate_mean(data)
    sum_squared = 0
    for value in data:
        sum_squared += (value - mean) ** 2
    return sum_squared / len(data)  

def calculate_std(data):
    variance = calculate_variance(data)
    return variance ** 0.5


df=load_data()
df=preprocess(df)
print(f"{'Feature':<30}{'Manual Mean':>15}{'NumPy Mean':>15}"
      f"{'Manual Var':>15}{'NumPy Var':>15}"
      f"{'Manual Std':>15}{'NumPy Std':>15}")

for column in df.columns:
    values = df[column].tolist()
    mean = calculate_mean(values)
    variance = calculate_variance(values)
    std = calculate_std(values)
    np_mean = np.mean(values)
    np_var = np.var(values)
    np_std = np.std(values)
    print(f"{column:<30}"
          f"{mean:>15.4f}"
          f"{np_mean:>15.4f}"
          f"{variance:>15.4f}"
          f"{np_var:>15.4f}"
          f"{std:>15.4f}"
          f"{np_std:>15.4f}")