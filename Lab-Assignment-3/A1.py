import pandas as pd
import numpy as np

def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )

    return df

def category(df):
    return df.select_dtypes(include=['object','str']).columns

def numerics(df):
    return df.select_dtypes(include=np.number).columns



df=load_data()

print("\ncategorical columns")
categorical_columns = category(df)
print(categorical_columns)

print("\nnumerical columns")
numerical_columns = numerics(df)
print(numerical_columns)

print(df.dtypes)
print(df.head)

'''
nominal-marital_status,dt_customer  ; these columns cant be measured
ordinal-education ; this column has an order, for eg:- graduation < PHD
'''


      