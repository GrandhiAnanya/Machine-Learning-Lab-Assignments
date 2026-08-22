import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import time  

def load_data():
    df=pd.read_csv("features.csv")
    return df

def label_encoding(column): 
    unique_values = column.dropna().unique() 
    mapping = {value: index for index, value in enumerate(unique_values)} 
    encoded_column = column.map(mapping) 
    return encoded_column, mapping 

def one_hot_encoding(column): 
    encoded_df = pd.get_dummies(column, prefix=column.name) 
    return encoded_df

def fill_values(numeric_columns,df):
    for col in numeric_columns:
        df[col]=pd.to_numeric(df[col])
    for col in numeric_columns:
        df[col]=df[col].fillna(df[col].median())

def distance_calculation(p1, p2, o):
    dist = abs(p1 - p2) ** o
    return dist

def sort(a):
    keys=list(a.keys())
    for i in range(1,len(a)):
        key=keys[i]
        j=i-1
        while j>=0 and a[keys[j]][0] > a[key][0]:
            keys[j+1]=keys[j]
            j-=1
        keys[j+1]=key   
    return keys

def identify_class(dist,distances,y_train,k):
    k_nearest=[]
    dist_k=[]
    for i in range(0,k):
        k_nearest.append(y_train[dist[i]])
        dist_k.append(distances[dist[i]][2])

    mode={}
    for i in range(len(k_nearest)):
        if k_nearest[i] in mode:
            mode[k_nearest[i]]+=dist_k[i]
        else:
            mode[k_nearest[i]]=dist_k[i]

    highest_class=max(mode.values())
    for key in mode:
            if highest_class==mode[key]:
                return key
    



def wknn(X_test,X_train,y_train,k_nn):
    classes={}
    for i in range(0,len(X_test)):
        distances={}
        for k in range(0,len(X_train)):
            d=[]
            for j in range(0,len(X_test[i])):
                d.append(distance_calculation(X_test[i][j],X_train[k][j],2))
            d_final=pow(np.sum(d),1/2)
            weight=1/d_final
            distances[k] = ([d_final,y_train[k],weight])
        dist=sort(distances)
        ic=identify_class(dist,distances,y_train,k_nn)
        classes[i] = (ic)
        
    return classes

def score(classes,y_test):
    total_right=0
    for i in range(len(classes)):
        if classes[i]==y_test[i]:
            total_right+=1

    accuracy=total_right/len(y_test)
    return accuracy


times=[]
for i in range(10):
    start_time = time.perf_counter()
    df=load_data()
    df = df.drop(columns='image_name')
    X = df.drop(columns="person_id").to_numpy()
    y = df["person_id"].to_numpy()
    k=3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  
    classes=wknn(X_test,X_train,y_train,k)
    end_time = time.perf_counter()
    times.append(end_time - start_time)
    
average_time = sum(times) / len(times)
print("test\tpv\tav")
for i in range(len(X_test)):
    print(f"{i}\t{classes[i]}\t{y_test[i]}")

correct = 0
tp = 0
fp = 0
fn = 0

for i in range(len(y_test)):

    predicted = classes[i]
    actual = y_test[i]

    # Accuracy
    if predicted == actual:
        correct += 1

    # Assuming A is the positive class
    if predicted == "A" and actual == "A":
        tp += 1

    elif predicted == "A" and actual != "A":
        fp += 1

    elif predicted != "A" and actual == "A":
        fn += 1


accuracy = correct / len(y_test)

precision = tp / (tp + fp) if (tp + fp) != 0 else 0

recall = tp / (tp + fn) if (tp + fn) != 0 else 0

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) != 0
    else 0
)

print("Average computational time: ",average_time)
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1_score)

'''ascore=score(classes,y_test)
print(f"\n Accuracy score:{ascore}")'''
 



