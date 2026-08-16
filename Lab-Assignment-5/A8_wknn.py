import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


df=load_data()
df = df.drop(columns='image_name')
X = df.drop(columns="person_id").to_numpy()
y = df["person_id"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
scores=[]
scores_lib=[]
for i in range(1,11):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    p=neigh.predict(X_test)
    scores_lib.append(neigh.score(X_test, y_test))

    classes=wknn(X_test,X_train,y_train,i)
    ascore=score(classes,y_test)
    scores.append(ascore)

plt.plot(range(1,11),scores,marker="o",label="without lib")
plt.plot(range(1,11),scores_lib,marker="x",label="with lib")
plt.xlabel("k values")
plt.ylabel("Accuracy scores")
plt.legend()
plt.grid(True)
plt.show()



    