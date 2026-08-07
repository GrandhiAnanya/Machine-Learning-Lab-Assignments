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
    return df

def dot_product(A, B):
    return np.sum(A * B)

def euclidean_norm(vector):
    return np.sqrt(np.sum(vector ** 2))




df=load_data()
df=preprocess(df)
A = df.iloc[0].values
B = df.iloc[1].values

custom_dot = dot_product(A, B)
numpy_dot = np.dot(A, B)

print("Dot Product Comparison")
print("----------------------")
print("Custom Function :", custom_dot)
print("NumPy dot()     :", numpy_dot)
print("Difference      :", abs(custom_dot - numpy_dot))


custom_norm_A = euclidean_norm(A)
numpy_norm_A = np.linalg.norm(A)

custom_norm_B = euclidean_norm(B)
numpy_norm_B = np.linalg.norm(B)

print("\nEuclidean Norm Comparison")
print("-------------------------")
print("Vector A")
print("Custom Function :", custom_norm_A)
print("NumPy norm()    :", numpy_norm_A)
print("Difference      :", abs(custom_norm_A - numpy_norm_A))

print("\nVector B")
print("Custom Function :", custom_norm_B)
print("NumPy norm()    :", numpy_norm_B)
print("Difference      :", abs(custom_norm_B - numpy_norm_B))