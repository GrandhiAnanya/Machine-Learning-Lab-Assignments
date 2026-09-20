import pandas as pd
import math

def load_data():
    df=pd.read_csv("features.csv")
    return df

def entropy(y,df):
    classes =y.unique()
    class_counts = y.value_counts()
    total=len(df)
    sum=0
    for i in (classes):
        p_i=(class_counts[i])/total
        sum+=(p_i)*(math.log(p_i,2))

    return sum*(-1)
         






df=load_data()
X = df.drop(columns=['person_id', 'image_name'])
y = df['person_id']
classes =y.unique()
class_counts = y.value_counts()
entropy=entropy(y,df)
print("entory =",entropy)
#print(classes)
#print(class_counts)
