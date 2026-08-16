import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier  


def load_data():
    df=pd.read_csv("features.csv")
    return df
    


df=load_data()
df = df.drop(columns='image_name')
X = df.drop(columns="person_id").to_numpy()
y = df["person_id"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
p=neigh.predict(X_test)
print("test\tpv\tav")
for i in range(len(p)):
    print(f"{i}\t{p[i]}\t{y_test[i]}")
print (f"Acuuracy score:{neigh.score(X_test, y_test)}")