import pandas as pd
import numpy as np

#chatgpt used
def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )

    return df

def numeric(df):
    return df.select_dtypes(include=['int64', 'float64']).columns

def categoric(df):
    return df.select_dtypes(include=['object', 'category', 'bool']).columns


df=load_data()
numerical_cols = numeric(df)
categorical_cols = categoric(df)

print("Numerical:", numerical_cols)
print("Categorical:", categorical_cols)