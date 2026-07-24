import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

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

def label_encode(df):
    encoder=LabelEncoder()
    df["Education"]=encoder.fit_transform(df['Education'])
    return df

def one_hot_encode(df):
    encoder=OneHotEncoder(sparse_output=False)

    encoded_data=encoder.fit_transform(df[['Marital_Status']])
    encoded_df=pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(['Marital_Status'])
    )

    final_df = pd.concat(
    [df.drop(columns='Marital_Status'), encoded_df],
    axis=1
    )

    return final_df


df=load_data()

print("before encoding:")
print(df)
'''
print("\ncategorical columns")
categorical_columns = category(df)
print(categorical_columns)

print("\nnumerical columns")
numerical_columns = numerics(df)
print(numerical_columns)


label encoding is used for scalar to scalar mapping , after analysing the dataset ordinal data such as education can be encoded using label encoding
one hot encoding is used for vector to scalar mapping , nominal data such as marital status and dt_customers can be encoded using one hot encoding

'''
df=label_encode(df)
df=one_hot_encode(df)

print("after encoding:")
print(df)



