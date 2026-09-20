import pandas as pd
import math

def load_data():
    df=pd.read_csv("features.csv")
    return df

def gini(y,df):
    classes =y.unique()
    class_counts = y.value_counts()
    total=len(df)
    sum=0
    for i in (classes):
        p_i=(class_counts[i])/total
        sum+=(p_i)**2

    return 1-sum
         






df=load_data()
X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']
classes =y.unique()
class_counts = y.value_counts()
gini=gini(y,df)
print("gini index =",gini)

