import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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

def load_data():
    df = pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI"
    )

    return df

def preprocess_data(df):
    df=df.replace("?",pd.NA)
   
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

def calculate_jc_smc(vector1,vector2):
    f11 = ((vector1 == 1) & (vector2 == 1)).sum()
    f10 = ((vector1 == 1) & (vector2 == 0)).sum()
    f01 = ((vector1 == 0) & (vector2 == 1)).sum()
    f00 = ((vector1 == 0) & (vector2 == 0)).sum()
    jc = f11/(f11+f10+f01)
    smc=(f11+f00)/(f11+f10+f01+f00)

    return jc,smc

def cosine_similarity(vec1,vec2):
    dot_prod=np.dot(vec1,vec2)
    mag1=np.linalg.norm(vec1)
    mag2=np.linalg.norm(vec2)
    similarity=dot_prod/(mag1*mag2)

    return similarity

def get_first_20_vectors(df):
    df=df.drop(columns=["Record ID"])

    return df.iloc[:20]


def create_similarity_matrices(df):
    
    n=len(df)

    jc_m = np.zeros((n,n))
    smc_m = np.zeros((n,n))
    cos_m = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            vec1=df.iloc[i]
            vec2=df.iloc[j]

            binary_vec1 = vec1[binary_columns]
            binary_vec2 = vec2[binary_columns]
            jc, smc = calculate_jc_smc(binary_vec1, binary_vec2)

            cos=cosine_similarity(vec1,vec2)
            jc_m[i , j]=jc
            smc_m[i , j]=smc
            cos_m[i , j]=cos
    
    return jc_m,smc_m,cos_m

def plot_heatmap(matrix,title):
    plt.figure(figsize=(8,6))

    sns.heatmap(matrix,
                annot=True,
                cmap="viridis")
    
    plt.title(title)
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    plt.show()



df = load_data()
df=preprocess_data(df)
df=get_first_20_vectors(df)

jc_m, smc_m, cos_m=create_similarity_matrices(df)

plot_heatmap(jc_m,"Jaccard Coefficient")
plot_heatmap(smc_m,"Simple Matching Coefficient")
plot_heatmap(cos_m,"Cosine Similarity")

