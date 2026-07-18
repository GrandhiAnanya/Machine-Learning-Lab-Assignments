import pandas as pd



def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="thyroid0387_UCI")
    return df

def binary_vectors(df):
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
    binary_df = df[binary_columns].replace({"t": 1, "f": 0})
    vector1 = binary_df.iloc[0]
    vector2 = binary_df.iloc[1]

    return vector1,vector2

def calculate_jc_smc(vector1,vector2):
    f11 = ((vector1 == 1) & (vector2 == 1)).sum()
    f10 = ((vector1 == 1) & (vector2 == 0)).sum()
    f01 = ((vector1 == 0) & (vector2 == 1)).sum()
    f00 = ((vector1 == 0) & (vector2 == 0)).sum()
    jc = f11/(f11+f10+f01)
    smc=(f11+f00)/(f11+f10+f01+f00)

    return jc,smc



df=load_data()
print(df.columns)
print(df.head()) #for identifying the binary columns

vec1,vec2=binary_vectors(df)
jc,smc=calculate_jc_smc(vec1,vec2)

print(f"\n JC: {jc} and SMC: {smc}")
""" SMC is higer than JC because smc considers matching values 
for the thyroid dataset JC would be more appropriate because it 
would be more useful and informative to observ pateints with medical conditions"""