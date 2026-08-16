import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier  

def load_data():
    df=pd.read_csv("features.csv")
    return df

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

def identify_class(a,y_train,k):
    k_nearest=[]
    for i in range(0,k):
        k_nearest.append(y_train[a[i]])

    mode={}
    for i in k_nearest:
        if i in mode:
            mode[i]+=1
        else:
            mode[i]=1

    highest_class=max(mode.values())
    keys=[]
    for key in mode:
        if highest_class==mode[key]:
            keys.append(key)

    if len(keys)>1:
        for i in a:
            if y_train[i]in keys:
                return y_train[i]
    else:
        return keys[0]



def knn(X_test,X_train,y_train,k_nn):
    classes={}
    for i in range(0,len(X_test)):
        distances={}
        for k in range(0,len(X_train)):
            d=[]
            for j in range(0,len(X_test[i])):
                d.append(distance_calculation(X_test[i][j],X_train[k][j],2))
            d_final=pow(np.sum(d),1/2)
            distances[k] = ([d_final,y_train[k]])
        dist=sort(distances)
        ic=identify_class(dist,y_train,k_nn)
        classes[i] = (ic)
        
    return classes

def fit(X_train,y_train,k):
    X_train=X_train
    y_train=y_train
    k=k

def score(classes,y_test):
    total_right=0
    for i in range(len(classes)):
        if classes[i]==y_test[i]:
            total_right+=1

    accuracy=total_right/len(y_test)
    return accuracy



df=load_data()
df = df.drop(columns='image_name')
X = df.drop(columns="person_id").to_numpy()
y = df["person_id"].to_numpy()
k=3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
fit(X_train,y_train,k) 
classes=knn(X_test,X_train,y_train,k) #predict
print("test\tpv\tav")
for i in range(len(X_test)):
    print(f"{i}\t{classes[i]}\t{y_test[i]}")

ascore=score(classes,y_test)
print(f"\n Accuracy score:{ascore}")


 
 



