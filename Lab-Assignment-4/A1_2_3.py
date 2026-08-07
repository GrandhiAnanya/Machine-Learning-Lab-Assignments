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


def label(df):
    le = LabelEncoder()
    df["Education"] = le.fit_transform(df["Education"])

    return df

def one_hot(df):
    df = pd.get_dummies(df,
                    columns=["Marital_Status"],
                    dtype=int)
    return df


def drop_cols(df):
    df.drop(columns=["ID", "Z_CostContact", "Z_Revenue"], inplace=True)
    return df

def data_conv(df):
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])

# Extract useful features
    df["Customer_Year"] = df["Dt_Customer"].dt.year
    df["Customer_Month"] = df["Dt_Customer"].dt.month
    df["Customer_Day"] = df["Dt_Customer"].dt.day

    # Drop original date column
    df.drop(columns=["Dt_Customer"], inplace=True)
    return df
         


df=load_data()
df=label(df)
df=one_hot(df)
df=drop_cols(df)
df=data_conv(df)
print(df.head())


