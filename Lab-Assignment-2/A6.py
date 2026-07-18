import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



def load_data():
    df = pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI"
    )

    return df

def preprocess_data(df):
    df=df.replace("?",pd.NA)
    binary_columns = [
    "on thyroxine",
    "query on thyroxine",
    "on antithyroid medication",
    "sick",
    "pregnant",
    "thyroid surgery",
    "I131 treatment",
    "query hypothyroid",
    "query hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH measured",
    "T3 measured",
    "TT4 measured",
    "T4U measured",
    "FTI measured",
    "TBG measured"
    ]
    df[binary_columns]=df[binary_columns].replace({
        "t":1,
        "f":0
    })

    numeric_columns = [
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI",
    "TBG"
    ]

    for col in numeric_columns:
        df[col]=pd.to_numeric(df[col])
    for col in numeric_columns:
        df[col]=df[col].fillna(df[col].median())  

    encoder=LabelEncoder()
    df["sex"]=encoder.fit_transform(df["sex"])  
    df["referral source"] = encoder.fit_transform(df["referral source"])
    df["Condition"]=encoder.fit_transform(df["Condition"])

    return df

def get_vector(df):

    vec1=df.iloc[0]
    vec2=df.iloc[1]

    return vec1,vec2

def cosine_similarity(vec1,vec2):
    dot_prod=np.dot(vec1,vec2)
    mag1=np.linalg.norm(vec1)
    mag2=np.linalg.norm(vec2)
    similarity=dot_prod/(mag1*mag2)

    return similarity

    



df=load_data()
df=preprocess_data(df)
vec1,vec2=get_vector(df)
similarity=cosine_similarity(vec1,vec2)
print(f"Cosine Similarity : {similarity}")
