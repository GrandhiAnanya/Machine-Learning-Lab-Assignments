import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
import time

def load_data():
    df=pd.read_csv("features.csv")
    return df
    

times=[]
for i in range(10):
    start_time = time.perf_counter()
    df=load_data()
    selected_classes = ["A", "Z"]
    df = df[df["person_id"].isin(selected_classes)]
    df = df.drop(columns='image_name')
    X = df.drop(columns="person_id").to_numpy()
    y = df["person_id"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    p=neigh.predict(X_test)
    print("test\tpv\tav")
    for i in range(len(p)):
        print(f"{i}\t{p[i]}\t{y_test[i]}")
    print (f"Acuuracy score:{neigh.score(X_test, y_test)}")
    end_time = time.perf_counter()
        
    times.append(end_time - start_time)
        
average_time = sum(times) / len(times)

accuracy = accuracy_score(y_test, p)

precision = precision_score(
    y_test, p,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test, p,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test, p,
    average="weighted",
    zero_division=0
)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)
print("Avg Time :", average_time, "seconds")