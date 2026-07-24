import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )

    return df

def minkowski_distance(p1,p2):
    distance=[]
    for p in range(1,11):
        dist=0
        for j in range (len(p1)):
           dist+=abs(p1[j]-p2[j])**p
        dist=pow(dist,1/p)
        distance.append(dist)

    return distance

def minkowski_distance_lib(p1,p2):
    distance=[]
    for i in range(1,11):
        distance.append(minkowski(p1,p2,p=i))

    return distance





df=load_data()
numeric_df=df.select_dtypes(include=np.number)

p1=numeric_df.iloc[0].values
p2=numeric_df.iloc[1].values

distance=minkowski_distance(p1,p2)
distance2=minkowski_distance_lib(p1,p2)

plt.plot(range(1,11),distance,marker="o",label="without lib")
plt.plot(range(1,11),distance2,marker="x",label="with lib")
plt.xlabel("p value")
plt.ylabel("Minowski values")
plt.legend()
plt.grid(True)
plt.show()